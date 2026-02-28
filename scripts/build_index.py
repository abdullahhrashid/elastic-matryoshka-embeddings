import argparse
import json
import os
import time
import sys
import faiss
import numpy as np
import redis
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from src.models.embedding_model import EmbeddingModel
from src.utils.logger import get_logger

logger = get_logger(__file__)

DIMS = [768, 512, 256, 128, 64]
INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'indexes')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_docs', type=int, default=100_000)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument(
        '--checkpoint',
        default='checkpoints/v1_fb_contriever/checkpoint_epoch_2.pt',
    )
    p.add_argument('--redis_host', default='localhost')
    p.add_argument('--redis_port', type=int, default=6379)
    return p.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    logger.info(f'Loading model from {checkpoint_path}')
    model = EmbeddingModel('facebook/contriever')
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info('Model loaded successfully')
    return model


def load_passages(n_docs: int):
    logger.info(f'Loading {n_docs:,} MS MARCO passages from BEIR (streaming)...')
    dataset = load_dataset(
        'BeIR/msmarco',
        'corpus',
        split='corpus',
        streaming=True,
        trust_remote_code=True,
    )
    passages = []
    for i, row in enumerate(dataset):
        if i >= n_docs:
            break
        title = row.get('title', '') or ''
        text = (title + ' ' + row['text']).strip() if title else row['text']
        passages.append((i, text))  # use sequential int IDs
    logger.info(f'Loaded {len(passages):,} passages')
    return passages


@torch.no_grad()
def encode_passages(model, tokenizer, passages, batch_size: int, device: torch.device):
    all_embeddings = []
    all_ids = []
    all_texts = []

    for start in tqdm(range(0, len(passages), batch_size), desc='Encoding passages'):
        batch = passages[start : start + batch_size]
        batch_ids = [p[0] for p in batch]
        batch_texts = [p[1] for p in batch]

        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)

        emb = model(encoded)  
        all_embeddings.append(emb.cpu().float().numpy())
        all_ids.extend(batch_ids)
        all_texts.extend(batch_texts)

    embeddings = np.concatenate(all_embeddings, axis=0)
    ids = np.array(all_ids, dtype=np.int64)
    return embeddings, ids, all_texts

def store_in_redis(r: redis.Redis, ids, embeddings, texts):
    logger.info(f'Storing {len(ids):,} documents in Redis...')
    pipe = r.pipeline(transaction=False)
    for i, (doc_id, text) in enumerate(zip(ids, texts)):
        payload = {
            'text': text,
            'embedding': embeddings[i].tolist(),
        }
        pipe.set(f'doc:{int(doc_id)}', json.dumps(payload))
        if (i + 1) % 5000 == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
    pipe.execute()
    logger.info('Redis storage complete')


def build_faiss_indexes(embeddings, ids):
    os.makedirs(INDEX_DIR, exist_ok=True)

    for dim in DIMS:
        logger.info(f'Building FAISS index for dim={dim}...')
        t0 = time.time()

        #slice + normalise
        vecs = embeddings[:, :dim].copy().astype(np.float32)
        faiss.normalize_L2(vecs)

        #IndexIDMap wraps a flat inner product index so we can store custom IDs
        base = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(base)
        index.add_with_ids(vecs, ids)

        out_path = os.path.join(INDEX_DIR, f'faiss_{dim}.index')
        faiss.write_index(index, out_path)
        logger.info(f'  Saved {index.ntotal:,} vectors - {out_path}  ({time.time()-t0:.1f}s)')

    #saving the sequential ID list so we can cross check if needed
    id_map_path = os.path.join(INDEX_DIR, 'id_map.npy')
    np.save(id_map_path, ids)
    logger.info(f'ID map saved - {id_map_path}')

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
        logger.info(f'Connected to Redis at {args.redis_host}:{args.redis_port}')
    except redis.ConnectionError:
        logger.error(
            f'Cannot connect to Redis at {args.redis_host}:{args.redis_port}. '
            'Start Redis with: docker run -d -p 6379:6379 redis:alpine'
        )
        sys.exit(1)

    model = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

    passages = load_passages(args.n_docs)

    logger.info('Starting encoding...')
    t_enc = time.time()
    embeddings, ids, texts = encode_passages(model, tokenizer, passages, args.batch_size, device)
    logger.info(f'Encoding done in {time.time()-t_enc:.1f}s  |  shape: {embeddings.shape}')

    store_in_redis(r, ids, embeddings, texts)

    build_faiss_indexes(embeddings, ids)

    logger.info('Indexing complete')

if __name__ == '__main__':
    main()

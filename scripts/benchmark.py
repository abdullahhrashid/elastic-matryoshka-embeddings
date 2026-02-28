import argparse
import json
import os
import sys
import time
import faiss
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from src.models.embedding_model import EmbeddingModel
from src.utils.logger import get_logger

logger = get_logger(__file__)

DIMS = [768, 512, 256, 128, 64]
INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'indexes')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__),
    '..', 'checkpoints', 'v1_fb_contriever', 'checkpoint_epoch_2.pt',
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_queries', type=int, default=200,
                   help='Number of dev queries to benchmark per dim')
    p.add_argument('--top_k', type=int, default=100,
                   help='Number of FAISS candidates to retrieve per query')
    p.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    p.add_argument('--warmup', type=int, default=10,
                   help='Warmup queries to run before timing (discarded)')
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
    return model

def load_dev_queries(n: int) -> list[str]:
    logger.info(f'Loading {n} MS MARCO dev queries...')
    ds = load_dataset('BeIR/msmarco', 'queries', split='queries', streaming=True, trust_remote_code=True,)
    queries = []
    for row in ds:
        queries.append(row['text'])
        if len(queries) >= n + 50:
            break
    rng = np.random.default_rng(42)
    rng.shuffle(queries)
    selected = queries[:n]
    logger.info(f'Loaded {len(selected)} queries')
    return selected

@torch.no_grad()
def encode_query(model, tokenizer, text: str, device: torch.device) -> np.ndarray:
    enc = tokenizer(
        text, padding=True, truncation=True, max_length=512, return_tensors='pt'
    ).to(device)
    emb = model(enc)  # (1, 768)
    return emb[0].cpu().float().numpy()

def percentile(data: list[float], p: float) -> float:
    arr = sorted(data)
    idx = (len(arr) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(arr) - 1)
    return arr[lo] + (arr[hi] - arr[lo]) * (idx - lo)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    model = load_model(args.checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    queries = load_dev_queries(args.n_queries + args.warmup)

    for dim in DIMS:
        path = os.path.join(INDEX_DIR, f'faiss_{dim}.index')
        if not os.path.exists(path):
            logger.error(f'Index not found: {path}. Run build_index.py first.')
            sys.exit(1)

    results = {}

    for dim in DIMS:
        logger.info(f'Benchmarking dim={dim}')
        index = faiss.read_index(os.path.join(INDEX_DIR, f'faiss_{dim}.index'))

        latencies_ms = []
        for i, query_text in enumerate(tqdm(queries, desc=f'dim={dim}')):
            #encode query
            query_768 = encode_query(model, tokenizer, query_text, device)

            #truncate + normalise to dim
            q_d = query_768[:dim].copy().astype(np.float32)
            q_d /= (np.linalg.norm(q_d) + 1e-9)
            q_d = q_d.reshape(1, -1)

            #time only the FAISS search (this is what scales with corpus size)
            t0 = time.perf_counter()
            index.search(q_d, args.top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            #skip warmup queries
            if i >= args.warmup:
                latencies_ms.append(elapsed_ms)

        p50 = percentile(latencies_ms, 50)
        p95 = percentile(latencies_ms, 95)
        p99 = percentile(latencies_ms, 99)
        mean = float(np.mean(latencies_ms))

        results[str(dim)] = {
            'n_queries': len(latencies_ms),
            'top_k': args.top_k,
            'mean_ms': round(mean, 4),
            'p50_ms': round(p50, 4),
            'p95_ms': round(p95, 4),
            'p99_ms': round(p99, 4),
        }

        logger.info(
            f'dim={dim:>4} | mean={mean:.3f}ms | '
            f'P50={p50:.3f}ms | P95={p95:.3f}ms | P99={p99:.3f}ms'
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'latency_benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Benchmark complete → {out_path}')


if __name__ == '__main__':
    main()

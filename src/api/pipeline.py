import json
import os
import time
import faiss
import numpy as np
import redis
import torch
from transformers import AutoTokenizer
from src.models.embedding_model import EmbeddingModel
from src.utils.logger import get_logger

logger = get_logger(__file__)

DIMS = [768, 512, 256, 128, 64]
INDEX_DIR = os.path.join(os.path.dirname(__file__), '../../data/indexes')

DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(__file__), '../../checkpoints/v1_fb_contriever/checkpoint_epoch_2.pt',)

class MRLSearchPipeline:
    def __init__(self, checkpoint_path = DEFAULT_CHECKPOINT, redis_host = 'localhost', redis_port = 6379, device = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f'Initialising pipeline on device: {self.device}')

        self.model = EmbeddingModel('facebook/contriever')
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        logger.info('Model loaded')

        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')

        self.indexes = {}
        for dim in DIMS:
            path = os.path.join(INDEX_DIR, f'faiss_{dim}.index')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f'FAISS index not found: {path}\n'
                    'Run `python scripts/build_index.py` first.'
                )
            self.indexes[dim] = faiss.read_index(path)
            logger.info(f'Loaded FAISS index dim={dim}  ({self.indexes[dim].ntotal:,} vectors)')

        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        try:
            self.redis.ping()
            logger.info(f'Connected to Redis at {redis_host}:{redis_port}')
        except redis.ConnectionError as e:
            raise ConnectionError(
                f'Cannot connect to Redis at {redis_host}:{redis_port}. '
                'Start with: docker run -d -p 6379:6379 redis:alpine'
            ) from e

    @torch.no_grad()
    def encode(self, text: str):
        enc = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        ).to(self.device)
        emb = self.model(enc)       
        return emb[0].cpu().float().numpy()

    def search(self, query, dim = 768, top_k_candidates = 100, top_n_results = 10):
        if dim not in self.indexes:
            raise ValueError(f'dim={dim} not supported. Choose from {DIMS}')

        timings = {}

        t0 = time.perf_counter()
        query_768 = self.encode(query)              
        timings['encode_ms'] = (time.perf_counter() - t0) * 1000

        query_d = query_768[:dim].copy().astype(np.float32)
        query_d /= (np.linalg.norm(query_d) + 1e-9)
        query_d = query_d.reshape(1, -1)            

        t1 = time.perf_counter()
        _scores, doc_ids = self.indexes[dim].search(query_d, top_k_candidates)
        timings['faiss_ms'] = (time.perf_counter() - t1) * 1000
        doc_ids = doc_ids[0].tolist()                 

        t2 = time.perf_counter()
        pipe = self.redis.pipeline(transaction=False)
        for doc_id in doc_ids:
            pipe.get(f'doc:{doc_id}')
        raw_values = pipe.execute()
        timings['redis_ms'] = (time.perf_counter() - t2) * 1000

        q_norm = query_768 / (np.linalg.norm(query_768) + 1e-9)

        results = []
        for doc_id, raw in zip(doc_ids, raw_values):
            if raw is None:
                continue
            payload = json.loads(raw)
            doc_emb = np.array(payload['embedding'], dtype=np.float32)
            doc_norm = doc_emb / (np.linalg.norm(doc_emb) + 1e-9)
            score = float(np.dot(q_norm, doc_norm))
            results.append({'doc_id': doc_id, 'text': payload['text'], 'score': score})

        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_n_results]
        for rank, r in enumerate(results, 1):
            r['rank'] = rank

        logger.info(
            f'Search complete | dim={dim} | '
            f"encode={timings['encode_ms']:.1f}ms "
            f"faiss={timings['faiss_ms']:.1f}ms "
            f"redis={timings['redis_ms']:.1f}ms"
        )
        return results

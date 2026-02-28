import argparse
import json
import os
import sys
import numpy as np
import redis
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger(__file__)

INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'indexes')
EMBED_PATH = os.path.join(INDEX_DIR, 'embeddings_768.npy')
IDS_PATH   = os.path.join(INDEX_DIR, 'doc_ids.npy')
TEXTS_PATH = os.path.join(INDEX_DIR, 'texts.json')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--redis_host', default='localhost')
    p.add_argument('--redis_port', type=int, default=6379)
    p.add_argument('--batch_size', type=int, default=5000,
                   help='Number of docs per Redis pipeline flush')
    return p.parse_args()


def main():
    args = parse_args()

    for path in [EMBED_PATH, IDS_PATH, TEXTS_PATH]:
        if not os.path.exists(path):
            logger.error(f'Missing file: {path}')
            sys.exit(1)

    logger.info('Loading embeddings, doc_ids, and texts...')
    embeddings = np.load(EMBED_PATH)          
    doc_ids    = np.load(IDS_PATH)             
    with open(TEXTS_PATH) as f:
        texts = json.load(f)

    N = len(doc_ids)
    assert embeddings.shape[0] == N == len(texts), \
        f'Shape mismatch: embeddings={embeddings.shape[0]}, ids={N}, texts={len(texts)}'
    logger.info(f'Loaded {N:,} documents  |  embedding shape: {embeddings.shape}')

    r = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    try:
        r.ping()
        logger.info(f'Connected to Redis at {args.redis_host}:{args.redis_port}')
    except redis.ConnectionError:
        logger.error(
            f'Cannot connect to Redis at {args.redis_host}:{args.redis_port}\n'
            'Start with: docker run -d -p 6379:6379 redis:alpine'
        )
        sys.exit(1)

    logger.info('Inserting documents into Redis...')
    pipe = r.pipeline(transaction=False)
    for i in tqdm(range(N), desc='Loading Redis'):
        payload = {
            'text':      texts[i],
            'embedding': embeddings[i].tolist(),
        }
        pipe.set(f'doc:{int(doc_ids[i])}', json.dumps(payload))
        if (i + 1) % args.batch_size == 0:
            pipe.execute()
            pipe = r.pipeline(transaction=False)
    pipe.execute()

    final_count = r.dbsize()
    logger.info(f'Redis now contains {final_count:,} keys')


if __name__ == '__main__':
    main()

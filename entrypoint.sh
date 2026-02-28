#!/bin/bash
set -e

export PYTHONPATH=/app

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

echo "[entrypoint] Waiting for Redis at ${REDIS_HOST}:${REDIS_PORT}..."
until python - <<EOF
import redis, sys
try:
    redis.Redis(host="${REDIS_HOST}", port=${REDIS_PORT}).ping()
    sys.exit(0)
except Exception:
    sys.exit(1)
EOF
do
    echo "[entrypoint] Redis not ready yet — retrying in 2s..."
    sleep 2
done
echo "[entrypoint] Redis is ready."

DOC_COUNT=$(python - <<EOF
import redis
r = redis.Redis(host="${REDIS_HOST}", port=${REDIS_PORT})
print(r.dbsize())
EOF
)

if [ "$DOC_COUNT" -eq "0" ]; then
    echo "[entrypoint] Redis is empty — loading 100k documents (first boot, ~90s)..."
    python scripts/load_redis.py --redis_host "${REDIS_HOST}" --redis_port "${REDIS_PORT}"
    echo "[entrypoint] Redis loaded."
else
    echo "[entrypoint] Redis already contains ${DOC_COUNT} documents — skipping load."
fi

echo "[entrypoint] Starting uvicorn..."
exec uvicorn src.api.app:app --host 0.0.0.0 --port 8000

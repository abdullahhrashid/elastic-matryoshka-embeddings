import os
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.api.pipeline import MRLSearchPipeline, DIMS

#shared pipeline instance (loaded once at startup)
pipeline: Optional[MRLSearchPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    checkpoint = os.environ.get('CHECKPOINT_PATH', 'checkpoints/v1_fb_contriever/checkpoint_epoch_2.pt',)
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    pipeline = MRLSearchPipeline(checkpoint_path=checkpoint, redis_host=redis_host, redis_port=redis_port,)
    yield
    
app = FastAPI(
    title='MRL Elastic Embeddings API',
    description=(
        'Semantic search over MS MARCO passages using a Matryoshka fine-tuned '
        'Contriever model. Select any embedding dimension at inference time — '
        'smaller dims trade accuracy for speed.'
    ),
    version='0.1.0',
    lifespan=lifespan,
)

class SearchRequest(BaseModel):
    query: str = Field(..., description='Natural language query string')
    dim: int = Field(
        768,
        description=f'Embedding dimension to use for ANN search. One of {DIMS}.',
    )
    top_k: int = Field(
        100,
        ge=1,
        le=1000,
        description='Number of FAISS candidates to retrieve before reranking.',
    )
    top_n: int = Field(
        10,
        ge=1,
        le=100,
        description='Number of results to return after full-dim cosine reranking.',
    )

class SearchResult(BaseModel):
    rank: int
    doc_id: int
    score: float = Field(..., description='Cosine similarity at full 768-dim (reranked)')
    text: str

class SearchResponse(BaseModel):
    query: str
    dim: int
    results: List[SearchResult]

class HealthResponse(BaseModel):
    status: str
    dims_loaded: List[int]
    model: str

@app.get('/health', response_model=HealthResponse, tags=['Ops'])
def health():
    if pipeline is None:
        raise HTTPException(status_code=503, detail='Pipeline not initialised')
    return HealthResponse(
        status='ok',
        dims_loaded=list(pipeline.indexes.keys()),
        model='facebook/contriever (MRL fine-tuned)',
    )

@app.post('/search', response_model=SearchResponse, tags=['Search'])
def search(req: SearchRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail='Pipeline not initialised')
    if req.dim not in DIMS:
        raise HTTPException(
            status_code=400,
            detail=f'Invalid dim={req.dim}. Must be one of {DIMS}.',
        )
    try:
        raw_results = pipeline.search(
            query=req.query,
            dim=req.dim,
            top_k_candidates=req.top_k,
            top_n_results=req.top_n,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResponse(
        query=req.query,
        dim=req.dim,
        results=[SearchResult(**r) for r in raw_results],
    )

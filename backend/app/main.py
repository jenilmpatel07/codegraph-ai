import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db, CodeNode
from app.ingestion.repo_ingestor import RepoIngestor
from app.retrieval.hybrid_search import HybridCodeSearcher
from app.retrieval.qa_engine import CitationQAEngine

app = FastAPI(title="CodeGraph AI Engine", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ingestor = RepoIngestor()


@app.on_event("startup")
def on_startup():
    init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class IngestRequest(BaseModel):
    repo_url: HttpUrl


class SearchRequest(BaseModel):
    repo_name: str
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    repo_name: str
    question: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/ingest")
def ingest_repo(payload: IngestRequest, db: Session = Depends(get_db)):
    try:
        stats = ingestor.ingest_github_repo(str(payload.repo_url), db)
        return {"message": "Repository indexed successfully", **stats}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/graph/{repo_name}")
def get_dependency_graph(repo_name: str, db: Session = Depends(get_db)):
    nodes = db.query(CodeNode).filter(CodeNode.repo_name == repo_name).all()
    if not nodes:
        raise HTTPException(status_code=404, detail="Repository not found")

    symbol_to_ids = {}
    for n in nodes:
        symbol_to_ids.setdefault(n.symbol_name, []).append(str(n.id))

    graph_nodes = [
        {
            "id": str(n.id),
            "label": n.symbol_name,
            "type": n.node_type,
            "file_path": n.file_path,
            "lines": f"{n.start_line}-{n.end_line}",
            "language": n.language,
        }
        for n in nodes
    ]

    graph_edges = []
    for n in nodes:
        for called_symbol in (n.calls or []):
            for target_id in symbol_to_ids.get(called_symbol, []):
                if target_id != str(n.id):
                    graph_edges.append(
                        {"source": str(n.id), "target": target_id, "label": "calls"}
                    )

    return {"repo_name": repo_name, "nodes": graph_nodes, "edges": graph_edges}


@app.post("/api/search")
def hybrid_search(payload: SearchRequest, db: Session = Depends(get_db)):
    searcher = HybridCodeSearcher(ingestor.embedder)
    matches = searcher.search(payload.repo_name, payload.query, db, top_k=payload.top_k)
    if not matches:
        raise HTTPException(status_code=404, detail="No matching code nodes found")
    return {"repo_name": payload.repo_name, "query": payload.query, "results": matches}


@app.post("/api/ask")
def ask_codebase(payload: AskRequest, db: Session = Depends(get_db)):
    searcher = HybridCodeSearcher(ingestor.embedder)
    qa = CitationQAEngine(searcher)
    return qa.ask(payload.repo_name, payload.question, db, top_k=payload.top_k)

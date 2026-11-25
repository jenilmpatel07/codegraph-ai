from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session

from app.core.database import SessionLocal, init_db, CodeNode
from app.ingestion.repo_ingestor import RepoIngestor

app = FastAPI(title="CodeGraph AI Engine", version="0.1.0")

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

import shutil
from pathlib import Path
from typing import Dict, Any, List
from git import Repo
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

from app.ingestion.ast_parser import ASTCodeParser, EXTENSION_TO_LANG
from app.core.database import CodeNode

IGNORE_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next"}
CLONE_BASE = Path("cloned_repos")


class RepoIngestor:
    def __init__(self) -> None:
        self.parser = ASTCodeParser()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def ingest_github_repo(self, repo_url: str, db: Session) -> Dict[str, Any]:
        repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
        target_dir = CLONE_BASE / repo_name

        if target_dir.exists():
            shutil.rmtree(target_dir)
        CLONE_BASE.mkdir(parents=True, exist_ok=True)

        Repo.clone_from(repo_url, target_dir, depth=1)

        db.query(CodeNode).filter(CodeNode.repo_url == repo_url).delete()

        all_chunks = []
        files_scanned = 0

        for path in target_dir.rglob("*"):
            if any(part in IGNORE_DIRS for part in path.parts):
                continue
            if path.is_file() and path.suffix.lower() in EXTENSION_TO_LANG:
                files_scanned += 1
                all_chunks.extend(self.parser.parse_file(path, repo_root=target_dir))

        if not all_chunks:
            return {"repo_name": repo_name, "files_scanned": files_scanned, "nodes_indexed": 0}

        texts_to_embed = [
            f"{c.language} {c.node_type} {c.symbol_name} in {c.file_path}:\n{c.code[:1500]}"
            for c in all_chunks
        ]
        embeddings = self.embedder.encode(texts_to_embed, show_progress_bar=False).tolist()

        db_nodes: List[CodeNode] = []
        for chunk, emb in zip(all_chunks, embeddings):
            db_nodes.append(
                CodeNode(
                    repo_url=repo_url,
                    repo_name=repo_name,
                    file_path=chunk.file_path,
                    language=chunk.language,
                    node_type=chunk.node_type,
                    symbol_name=chunk.symbol_name,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    code=chunk.code,
                    calls=chunk.calls,
                    embedding=emb,
                )
            )

        db.bulk_save_objects(db_nodes)
        db.commit()

        return {
            "repo_name": repo_name,
            "files_scanned": files_scanned,
            "nodes_indexed": len(db_nodes),
        }

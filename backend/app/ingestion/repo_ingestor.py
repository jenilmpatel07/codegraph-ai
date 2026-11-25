import ast
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer

from app.core.database import CodeNode
from app.ingestion.ast_parser import ASTCodeParser

CLONE_DIR = Path("/tmp/codegraph_repos")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".tox", ".eggs"}


class RepoIngestor:
    def __init__(self) -> None:
        self.parser = ASTCodeParser()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        CLONE_DIR.mkdir(parents=True, exist_ok=True)

    def _extract_doc_chunks(self, repo_name: str, file_path: Path, rel_path: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return chunks

        if file_path.name.lower().startswith("readme"):
            lines = content.splitlines()
            step = 45
            for idx in range(0, min(len(lines), 180), step):
                block = "\n".join(lines[idx : idx + step]).strip()
                if len(block) > 40:
                    chunks.append(
                        {
                            "repo_name": repo_name,
                            "file_path": rel_path,
                            "language": "markdown",
                            "node_type": "documentation",
                            "symbol_name": f"{file_path.name}#section_{idx // step + 1}",
                            "start_line": idx + 1,
                            "end_line": min(idx + step, len(lines)),
                            "code": block,
                            "calls": [],
                        }
                    )
        elif file_path.suffix == ".py":
            try:
                tree = ast.parse(content)
                doc = ast.get_docstring(tree)
                if doc and len(doc.strip()) > 30:
                    doc_lines = doc.strip().splitlines()
                    chunks.append(
                        {
                            "repo_name": repo_name,
                            "file_path": rel_path,
                            "language": "python",
                            "node_type": "module_docstring",
                            "symbol_name": f"{file_path.stem}.__doc__",
                            "start_line": 1,
                            "end_line": min(len(doc_lines) + 2, len(content.splitlines())),
                            "code": f'"""\n{doc.strip()}\n"""',
                            "calls": [],
                        }
                    )
            except Exception:
                pass
        return chunks

    def ingest_github_repo(self, repo_url: str, db: Session) -> Dict[str, Any]:
        clean_url = repo_url.rstrip("/")
        if clean_url.endswith(".git"):
            clean_url = clean_url[:-4]
        repo_name = clean_url.split("/")[-1]
        target_dir = CLONE_DIR / repo_name

        if target_dir.exists():
            shutil.rmtree(target_dir)

        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

        db.query(CodeNode).filter(CodeNode.repo_name == repo_name).delete()
        db.commit()

        all_chunks: List[Dict[str, Any]] = []
        files_scanned = 0

        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                fpath = Path(root) / fname
                rel_path = os.path.relpath(str(fpath), str(target_dir))

                if fname.lower().startswith("readme") or fpath.suffix == ".py":
                    doc_chunks = self._extract_doc_chunks(repo_name, fpath, rel_path)
                    if doc_chunks:
                        all_chunks.extend(doc_chunks)

                parsed = self.parser.parse_file(fpath, rel_path)
                if parsed:
                    files_scanned += 1
                    for item in parsed:
                        row = item if isinstance(item, dict) else (item.model_dump() if hasattr(item, "model_dump") else vars(item))
                        row["repo_name"] = repo_name
                        all_chunks.append(row)

        if not all_chunks:
            return {"repo_name": repo_name, "files_scanned": files_scanned, "nodes_indexed": 0}

        texts = [
            f"{c['node_type']} {c['symbol_name']} in {c['file_path']}:\n{c['code'][:1200]}"
            for c in all_chunks
        ]
        embeddings = self.embedder.encode(texts, batch_size=32, show_progress_bar=False)

        db_records = [
            CodeNode(
                repo_url=repo_url,
                repo_name=chunk["repo_name"],
                file_path=chunk["file_path"],
                language=chunk["language"],
                node_type=chunk["node_type"],
                symbol_name=chunk["symbol_name"],
                start_line=chunk["start_line"],
                end_line=chunk["end_line"],
                code=chunk["code"],
                calls=chunk["calls"],
                embedding=emb.tolist(),
            )
            for chunk, emb in zip(all_chunks, embeddings)
        ]

        db.bulk_save_objects(db_records)
        db.commit()

        return {
            "repo_name": repo_name,
            "files_scanned": files_scanned,
            "nodes_indexed": len(db_records),
        }

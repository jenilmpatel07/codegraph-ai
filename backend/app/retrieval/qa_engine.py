import hashlib
import json
import os
import urllib.request
from typing import Dict, Any, List, Optional
import redis
from sqlalchemy.orm import Session
from app.retrieval.hybrid_search import HybridCodeSearcher

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class CitationQAEngine:
    def __init__(self, searcher: HybridCodeSearcher) -> None:
        self.searcher = searcher
        try:
            self.cache = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.cache.ping()
        except Exception:
            self.cache = None

    def _cache_key(self, repo_name: str, question: str) -> str:
        digest = hashlib.sha256(f"{repo_name}:{question.strip().lower()}".encode()).hexdigest()[:16]
        return f"codegraph:qa:{repo_name}:{digest}"

    def _try_ollama(self, prompt: str) -> Optional[str]:
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip() or None
        except Exception:
            return None

    def _try_openai_compatible(self, prompt: str) -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def _build_graph_extractive_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        doc_chunks = [c for c in chunks if c["node_type"] in ("documentation", "module_docstring")]
        code_chunks = [c for c in chunks if c["node_type"] not in ("documentation", "module_docstring")]

        sections: List[str] = []
        if doc_chunks:
            lead = doc_chunks[0]
            summary_lines = [
                ln.strip().lstrip("#").strip()
                for ln in lead["code"].splitlines()
                if ln.strip() and not ln.strip().startswith(("===", "---", ".. ", "::", '"""'))
            ][:4]
            sections.append(
                f"Overview (`[{lead['citation']}]`):\n" + " ".join(summary_lines)
            )

        if code_chunks:
            lines = ["Core AST Definitions & 1-Hop Call Graph:"]
            for idx, c in enumerate(code_chunks, start=1):
                neighbors = c.get("graph_neighbors") or []
                neighbor_str = (
                    ", ".join(f"{nb['symbol_name']} [{nb['citation']}]" for nb in neighbors[:3])
                    if neighbors
                    else (", ".join(c["calls"][:4]) or "leaf symbol")
                )
                lines.append(
                    f"{idx}. `{c['symbol_name']}` ({c['node_type']}) at `[{c['citation']}]`\n"
                    f"   -> 1-Hop Dependencies: {neighbor_str}"
                )
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def ask(self, repo_name: str, question: str, db: Session, top_k: int = 5) -> Dict[str, Any]:
        key = self._cache_key(repo_name, question)
        if self.cache:
            cached = self.cache.get(key)
            if cached:
                result = json.loads(cached)
                result["cached"] = True
                return result

        chunks = self.searcher.search(repo_name, question, db, top_k=top_k)
        if not chunks:
            return {
                "repo_name": repo_name,
                "question": question,
                "answer": "No indexed symbols matched your query in this repository.",
                "citations": [],
                "cached": False,
                "engine": "none",
            }

        context_blocks = []
        for c in chunks:
            neighbors = c.get("graph_neighbors") or []
            hop_text = ""
            if neighbors:
                hop_text = "\n1-Hop Callees:\n" + "\n".join(
                    f"  - {nb['symbol_name']} [{nb['citation']}]: {nb['signature']}" for nb in neighbors
                )
            context_blocks.append(
                f"[{c['citation']}] ({c['node_type']} {c['symbol_name']})\n{c['code'][:1600]}{hop_text}"
            )

        prompt = (
            "You are a principal software architect answering questions strictly from the provided AST snippets, documentation, and 1-hop call-graph neighbors.\n"
            "Every claim MUST cite exact file paths and line numbers using `[file_path:start-end]`.\n"
            "Do not invent symbols or behavior not shown in the context.\n\n"
            f"Question: {question}\n\n"
            f"GraphRAG Context:\n" + "\n\n".join(context_blocks) + "\n\nAnswer:"
        )

        engine = "ollama"
        answer = self._try_ollama(prompt)
        if not answer:
            engine = "openai_compatible"
            answer = self._try_openai_compatible(prompt)
        if not answer:
            engine = "graphrag_extractive"
            answer = self._build_graph_extractive_answer(question, chunks)

        response_payload = {
            "repo_name": repo_name,
            "question": question,
            "answer": answer,
            "citations": [
                {
                    "citation": c["citation"],
                    "symbol_name": c["symbol_name"],
                    "node_type": c["node_type"],
                    "file_path": c["file_path"],
                    "lines": c["lines"],
                    "rrf_score": c["rrf_score"],
                    "code": c["code"],
                }
                for c in chunks
            ],
            "cached": False,
            "engine": engine,
        }

        if self.cache:
            self.cache.setex(key, 3600, json.dumps(response_payload))

        return response_payload

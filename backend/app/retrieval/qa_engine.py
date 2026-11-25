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
            with urllib.request.urlopen(req, timeout=15) as resp:
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
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None

    def _build_extractive_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        lines = [
            f"Based on static AST and hybrid retrieval for `{question}`, here are the verified code definitions:\n"
        ]
        for idx, c in enumerate(chunks, start=1):
            calls_str = ", ".join(c["calls"][:6]) if c["calls"] else "none"
            first_doc = ""
            for raw_line in c["code"].splitlines()[1:6]:
                cleaned = raw_line.strip().strip('"\'')
                if cleaned and not cleaned.startswith(("def ", "class ", "@")):
                    first_doc = f" — {cleaned}"
                    break
            lines.append(
                f"{idx}. **`{c['symbol_name']}`** (`{c['node_type']}`) at `[{c['citation']}]`{first_doc}\n"
                f"   - Outgoing calls: `{calls_str}`"
            )
        return "\n".join(lines)

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

        context_blocks = "\n\n".join(
            f"[{c['citation']}] ({c['node_type']} {c['symbol_name']})\n{c['code'][:1800]}"
            for c in chunks
        )
        prompt = (
            "You are a senior software architect answering questions strictly from the provided AST code snippets.\n"
            "Every claim MUST cite exact file paths and line numbers using `[file_path:start-end]`.\n"
            "Do not invent symbols or behavior not shown in the snippets.\n\n"
            f"Question: {question}\n\n"
            f"Code Context:\n{context_blocks}\n\n"
            "Answer:"
        )

        engine = "ollama"
        answer = self._try_ollama(prompt)
        if not answer:
            engine = "openai_compatible"
            answer = self._try_openai_compatible(prompt)
        if not answer:
            engine = "ast_extractive"
            answer = self._build_extractive_answer(question, chunks)

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
                }
                for c in chunks
            ],
            "cached": False,
            "engine": engine,
        }

        if self.cache:
            self.cache.setex(key, 3600, json.dumps(response_payload))

        return response_payload

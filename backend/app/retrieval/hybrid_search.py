import re
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from app.core.database import CodeNode

STOPWORDS = {
    "what", "is", "are", "how", "does", "do", "the", "a", "an", "in", "of",
    "to", "and", "for", "with", "on", "this", "repo", "repository", "code", "work"
}


class HybridCodeSearcher:
    def __init__(self, embedder: SentenceTransformer) -> None:
        self.embedder = embedder

    def _path_multiplier(self, file_path: str, node_type: str, is_overview: bool) -> float:
        p = file_path.lower()
        if is_overview and node_type in ("documentation", "module_docstring"):
            return 1.65
        if p.startswith(("tests/", "test/", "examples/", "example/", "benchmarks/")):
            return 0.55
        if p.startswith(("src/", "lib/", "app/", "pkg/", "core/")):
            return 1.25
        return 1.0

    def search(self, repo_name: str, query: str, db: Session, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.embedder.encode([query], show_progress_bar=False)[0].tolist()
        q_lower = query.lower().strip()
        is_overview = any(
            phrase in q_lower
            for phrase in ("what is", "overview", "architecture", "purpose", "about", "how does this")
        )

        semantic_hits = (
            db.query(CodeNode)
            .filter(CodeNode.repo_name == repo_name)
            .order_by(CodeNode.embedding.cosine_distance(query_vec))
            .limit(top_k * 6)
            .all()
        )

        if not semantic_hits:
            return []

        tokens = [
            t.lower()
            for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
            if len(t) > 1 and t.lower() not in STOPWORDS and t.lower() != repo_name.lower()
        ]

        all_nodes = db.query(CodeNode).filter(CodeNode.repo_name == repo_name).all()
        symbol_index: Dict[str, CodeNode] = {}
        for n in all_nodes:
            if n.symbol_name not in symbol_index or n.file_path.startswith("src/"):
                symbol_index[n.symbol_name] = n

        lexical_scored = []
        for node in all_nodes:
            sym_lower = node.symbol_name.lower()
            code_lower = node.code.lower()
            path_lower = node.file_path.lower()
            score = 0.0
            for tok in tokens:
                if tok == sym_lower:
                    score += 5.0
                elif tok in sym_lower:
                    score += 2.5
                if tok in path_lower:
                    score += 1.2
                score += min(code_lower.count(tok) * 0.25, 2.0)
            if score > 0:
                lexical_scored.append((score, node))

        lexical_scored.sort(key=lambda x: x[0], reverse=True)
        lexical_hits = [node for _, node in lexical_scored[: top_k * 6]]

        rrf_k = 60
        rrf_scores: Dict[int, float] = {}
        node_map: Dict[int, CodeNode] = {}

        for rank, node in enumerate(semantic_hits, start=1):
            rrf_scores[node.id] = rrf_scores.get(node.id, 0.0) + 1.0 / (rrf_k + rank)
            node_map[node.id] = node

        for rank, node in enumerate(lexical_hits, start=1):
            rrf_scores[node.id] = rrf_scores.get(node.id, 0.0) + 1.0 / (rrf_k + rank)
            node_map[node.id] = node

        for nid, node in node_map.items():
            rrf_scores[nid] *= self._path_multiplier(node.file_path, node.node_type, is_overview)

        ranked_ids = sorted(rrf_scores.keys(), key=lambda nid: rrf_scores[nid], reverse=True)[:top_k]

        results = []
        for nid in ranked_ids:
            n = node_map[nid]
            # 1-hop graph neighbor resolution
            neighbor_defs = []
            for callee in (n.calls or [])[:4]:
                target = symbol_index.get(callee)
                if target and target.id != n.id:
                    sig = target.code.splitlines()[0].strip() if target.code else callee
                    neighbor_defs.append(
                        {
                            "symbol_name": target.symbol_name,
                            "citation": f"{target.file_path}:{target.start_line}-{target.end_line}",
                            "signature": sig[:120],
                        }
                    )

            results.append(
                {
                    "id": str(n.id),
                    "symbol_name": n.symbol_name,
                    "node_type": n.node_type,
                    "file_path": n.file_path,
                    "lines": f"{n.start_line}-{n.end_line}",
                    "citation": f"{n.file_path}:{n.start_line}-{n.end_line}",
                    "language": n.language,
                    "calls": n.calls or [],
                    "graph_neighbors": neighbor_defs,
                    "rrf_score": round(rrf_scores[nid], 5),
                    "code": n.code,
                }
            )
        return results

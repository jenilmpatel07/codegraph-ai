import re
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
from app.core.database import CodeNode


class HybridCodeSearcher:
    def __init__(self, embedder: SentenceTransformer) -> None:
        self.embedder = embedder

    def search(self, repo_name: str, query: str, db: Session, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.embedder.encode([query], show_progress_bar=False)[0].tolist()

        # 1. Semantic retrieval via pgvector cosine distance
        semantic_hits = (
            db.query(CodeNode)
            .filter(CodeNode.repo_name == repo_name)
            .order_by(CodeNode.embedding.cosine_distance(query_vec))
            .limit(top_k * 4)
            .all()
        )

        if not semantic_hits:
            return []

        # 2. Lexical / symbol scoring across repo nodes
        tokens = [t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query) if len(t) > 1]
        all_nodes = db.query(CodeNode).filter(CodeNode.repo_name == repo_name).all()

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
                    score += 1.0
                score += min(code_lower.count(tok) * 0.25, 2.0)
            if score > 0:
                lexical_scored.append((score, node))

        lexical_scored.sort(key=lambda x: x[0], reverse=True)
        lexical_hits = [node for _, node in lexical_scored[: top_k * 4]]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        rrf_scores: Dict[int, float] = {}
        node_map: Dict[int, CodeNode] = {}

        for rank, node in enumerate(semantic_hits, start=1):
            rrf_scores[node.id] = rrf_scores.get(node.id, 0.0) + 1.0 / (rrf_k + rank)
            node_map[node.id] = node

        for rank, node in enumerate(lexical_hits, start=1):
            rrf_scores[node.id] = rrf_scores.get(node.id, 0.0) + 1.0 / (rrf_k + rank)
            node_map[node.id] = node

        ranked_ids = sorted(rrf_scores.keys(), key=lambda nid: rrf_scores[nid], reverse=True)[:top_k]

        results = []
        for nid in ranked_ids:
            n = node_map[nid]
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
                    "rrf_score": round(rrf_scores[nid], 5),
                    "code": n.code,
                }
            )
        return results

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from app.core.database import SessionLocal, CodeNode
from app.ingestion.repo_ingestor import RepoIngestor
from app.retrieval.hybrid_search import HybridCodeSearcher

EVAL_SET = [
    {"query": "what is click", "expected_files": ["README.md", "README.rst", "src/click/__init__.py", "src/click/core.py"]},
    {"query": "how does Command parse arguments and invoke callbacks", "expected_files": ["src/click/core.py", "src/click/parser.py"]},
    {"query": "OptionParser option prefix parsing", "expected_files": ["src/click/parser.py"]},
    {"query": "terminal echo style and color formatting", "expected_files": ["src/click/termui.py", "src/click/_termui_impl.py"]},
    {"query": "ParamType convert type validation", "expected_files": ["src/click/types.py"]},
]


def evaluate():
    db = SessionLocal()
    ingestor = RepoIngestor()
    hybrid = HybridCodeSearcher(ingestor.embedder)

    vec_hits, hyb_hits = 0, 0
    vec_mrr, hyb_mrr = 0.0, 0.0

    for case in EVAL_SET:
        q = case["query"]
        expected = set(case["expected_files"])
        q_vec = ingestor.embedder.encode([q], show_progress_bar=False)[0].tolist()

        # Baseline: pure pgvector cosine similarity
        v_res = (
            db.query(CodeNode)
            .filter(CodeNode.repo_name == "click")
            .order_by(CodeNode.embedding.cosine_distance(q_vec))
            .limit(5)
            .all()
        )
        v_files = [r.file_path for r in v_res]
        if any(f in expected for f in v_files):
            vec_hits += 1
        for rank, f in enumerate(v_files, start=1):
            if f in expected:
                vec_mrr += 1.0 / rank
                break

        # CodeGraph Hybrid RRF + Path Priors
        h_res = hybrid.search("click", q, db, top_k=5)
        h_files = [r["file_path"] for r in h_res]
        if any(f in expected for f in h_files):
            hyb_hits += 1
        for rank, f in enumerate(h_files, start=1):
            if f in expected:
                hyb_mrr += 1.0 / rank
                break

    n = len(EVAL_SET)
    print(f"Benchmark Queries: {n} (repo: pallets/click)")
    print(f"Vector-Only Baseline  -> Recall@5: {vec_hits/n:.0%} | MRR: {vec_mrr/n:.3f}")
    print(f"CodeGraph Hybrid RRF  -> Recall@5: {hyb_hits/n:.0%} | MRR: {hyb_mrr/n:.3f}")
    db.close()


if __name__ == "__main__":
    evaluate()

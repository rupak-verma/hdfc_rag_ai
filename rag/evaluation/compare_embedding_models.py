"""Choose the embedding model by MEASUREMENT, not by guessing.  Run on Colab/laptop (needs sentence-transformers):
    python -m rag.evaluation.compare_embedding_models
    python -m rag.evaluation.compare_embedding_models --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
For each model: embed all chunks, run the 83 test questions through keyword / vector / hybrid search, record Recall@k, MRR,
vector size and embedding time.  The recommendation uses the DEV questions only; the TEST column is for honest reporting.
Does not need a database."""
import argparse, json, datetime as dt
from pathlib import Path
from rag.embeddings.generate_embeddings import build_embeddings
from rag.embeddings.embedding_model import EmbeddingModel
from rag.retrieval.stores import FileStore
from rag.retrieval.retriever import RagEngine
from rag.evaluation.retrieval_evaluation import compute, retrieval_metrics

DEFAULT = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-mpnet-base-v2"]
OUT = Path("rag/evaluation/results")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=DEFAULT); ap.add_argument("--backend")
    a = ap.parse_args(); items = json.loads(Path("rag/evaluation/test_questions.json").read_text()); res = []
    for name in a.models:
        print(f"== {name}", flush=True)
        out, man = build_embeddings(a.backend, name, verbose=False)
        model = EmbeddingModel(a.backend, name); eng = RagEngine(FileStore(str(out)), model)
        rows = compute(eng, items); dev = [r for r in rows if r["split"] == "dev"]; test = [r for r in rows if r["split"] == "test"]
        res.append(dict(model=man["model_id"], standin=man["is_standin"], dimension=man["dimension"], embed_seconds=man["seconds"],
            dev={m: retrieval_metrics(dev, f"rank_{m}") for m in ("keyword", "vector", "hybrid")},
            test={m: retrieval_metrics(test, f"rank_{m}") for m in ("keyword", "vector", "hybrid")}))
        h = res[-1]["dev"]["hybrid"]; print(f"   dim={man['dimension']} time={man['seconds']}s  dev hybrid MRR={h['mrr']} R@5={h['recall@5']}")
    best = max(res, key=lambda r: (r["dev"]["hybrid"]["mrr"], r["dev"]["hybrid"]["recall@5"], -r["dimension"]))
    lines = ["# Embedding model comparison", "", f"Date: {dt.date.today().isoformat()}  |  questions: dev {sum(1 for i in items if i['split']=='dev')} / test {sum(1 for i in items if i['split']=='test')}",
             "", "| Model | Dim | Embed time (s) | Dev hybrid MRR | Dev hybrid R@5 | Test vector R@5 | Test hybrid MRR | Test hybrid R@5 |", "|---|---|---|---|---|---|---|---|"]
    for r in res:
        lines.append(f"| {r['model']}{' (STAND-IN)' if r['standin'] else ''} | {r['dimension']} | {r['embed_seconds']} | {r['dev']['hybrid']['mrr']} | {r['dev']['hybrid']['recall@5']} | "
                     f"{r['test']['vector']['recall@5']} | {r['test']['hybrid']['mrr']} | {r['test']['hybrid']['recall@5']} |")
    lines += ["", f"**Recommendation (highest dev hybrid MRR, ties -> smaller vector): `{best['model']}` (dimension {best['dimension']}).**",
              "", "Next: set EMBEDDING_MODEL and EMBEDDING_DIMENSION in .env, then run generate_embeddings, db.migrate, ingestion_pipeline and `retrieval_evaluation --calibrate`."]
    if any(r["standin"] for r in res): lines.insert(2, "> **STAND-IN MODEL: test run of the script only. These numbers are NOT project results.**")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "embedding_model_comparison.md").write_text("\n".join(lines)); (OUT / "embedding_model_comparison.json").write_text(json.dumps(res, indent=2))
    print("\n".join(lines))

if __name__ == "__main__": main()

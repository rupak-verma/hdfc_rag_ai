"""Offline job: embed every chunk once and save vectors + a manifest.
Run from repo root (Colab is fine - no database needed):
    python -m rag.embeddings.generate_embeddings                      # real model from .env
    python -m rag.embeddings.generate_embeddings --backend spacy-standin   # test only
Output: rag/data/embeddings/<tag>/embeddings.npy, chunk_ids.json, manifest.json"""
import argparse, hashlib, json, re, sys, time, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
from rag.embeddings.embedding_model import EmbeddingModel

CHUNKS = Path("rag/data/approved/canonical_chunks.csv")

def tag_for(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", model_id).strip("-").lower()

def build_embeddings(backend=None, name=None, batch_size=64, limit=None, verbose=True):
    """Embed all chunks and save vectors + manifest. Returns (output_dir, manifest)."""
    m = EmbeddingModel(backend, name)
    df = pd.read_csv(CHUNKS, dtype=str).fillna("")
    if limit: df = df.head(limit)
    t0 = time.time(); parts = []
    for i in range(0, len(df), batch_size):
        parts.append(m.encode_passages(df.chunk_text.iloc[i:i + batch_size].tolist(), batch_size))
        if verbose and (i // batch_size) % 10 == 0: print(f"  {i + len(parts[-1])}/{len(df)} chunks", flush=True)
    E = np.vstack(parts).astype(np.float32)
    assert E.shape == (len(df), m.dimension) and np.isfinite(E).all()
    out = Path("rag/data/embeddings") / tag_for(m.model_id); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", E); (out / "chunk_ids.json").write_text(json.dumps(df.chunk_id.tolist()))
    manifest = dict(model_id=m.model_id, backend=m.backend, is_standin=m.is_standin, dimension=m.dimension,
        chunk_count=len(df), query_prefix=m.query_prefix, passage_prefix=m.passage_prefix,
        chunks_file=str(CHUNKS), chunks_sha256=hashlib.sha256(CHUNKS.read_bytes()).hexdigest(),
        seconds=round(time.time() - t0, 1), created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        python=sys.version.split()[0], numpy=np.__version__)
    if m.backend == "sentence-transformers":
        import sentence_transformers; manifest["sentence_transformers"] = sentence_transformers.__version__
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out, manifest

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backend"); ap.add_argument("--model")
    ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--limit", type=int, help="smoke test: first N chunks")
    a = ap.parse_args()
    out, manifest = build_embeddings(a.backend, a.model, a.batch_size, a.limit)
    print(json.dumps(manifest, indent=2)); print("saved to", out)

if __name__ == "__main__": main()

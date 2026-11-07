"""Load approved documents + chunks + vectors into PostgreSQL.  Idempotent: safe to run again.
    python -m rag.ingestion.ingestion_pipeline --embeddings rag/data/embeddings/<tag>
Checks before writing anything:
  1. manifest dimension == EMBEDDING_DIMENSION (if set) == the database column size
  2. chunk ids in the vectors file are exactly the chunk ids in canonical_chunks.csv, same order
  3. the chunks file has not changed since the vectors were made (SHA-256)
  4. stand-in (test) vectors are refused unless --allow-standin
Also removes any database document that is no longer in the approved set, so excluded records stay excluded."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np, pandas as pd, psycopg
from rag.db.migrate import dsn, column_dimension

APPR = Path("rag/data/approved")
DOC_COLS = ["document_id", "title", "text", "source_record_id", "category", "category_method", "source_name", "provenance",
            "verification_status", "answer_label", "version", "effective_date", "source_url", "source_page",
            "needs_fact_check", "fact_check_reasons", "product_ambiguous", "language"]

def vec_literal(v) -> str: return "[" + ",".join(f"{x:.7g}" for x in v) + "]"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--embeddings", required=True); ap.add_argument("--allow-standin", action="store_true")
    a = ap.parse_args(); emb = Path(a.embeddings)
    man = json.loads((emb / "manifest.json").read_text()); E = np.load(emb / "embeddings.npy")
    ids = json.loads((emb / "chunk_ids.json").read_text())
    D = pd.read_csv(APPR / "canonical_documents.csv", dtype=str).fillna("")
    C = pd.read_csv(APPR / "canonical_chunks.csv", dtype=str).fillna("")
    if man["is_standin"] and not a.allow_standin: raise SystemExit("Refusing stand-in (test) vectors. Use the real model, or pass --allow-standin for testing only.")
    if hashlib.sha256((APPR / "canonical_chunks.csv").read_bytes()).hexdigest() != man["chunks_sha256"]:
        raise SystemExit("canonical_chunks.csv changed after the vectors were made. Re-run generate_embeddings.")
    if ids != C.chunk_id.tolist() or E.shape != (len(C), man["dimension"]): raise SystemExit("Vectors do not line up with chunks.")
    if os.getenv("EMBEDDING_DIMENSION") and int(os.environ["EMBEDDING_DIMENSION"]) != man["dimension"]:
        raise SystemExit(f"EMBEDDING_DIMENSION={os.environ['EMBEDDING_DIMENSION']} but vectors are {man['dimension']}-dimensional.")
    D["needs_fact_check"] = D.needs_fact_check == "True"; D["product_ambiguous"] = D.product_ambiguous == "True"
    for c in ("effective_date", "source_url", "source_page"):      # empty text -> real NULL
        D[c] = D[c].astype(object).where(D[c] != "", None)

    with psycopg.connect(dsn()) as conn:
        dbdim = column_dimension(conn)
        if dbdim != man["dimension"]: raise SystemExit(f"Database column is vector({dbdim}), vectors are {man['dimension']}. Run rag.db.migrate (--recreate if changing model).")
        with conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge.documents WHERE NOT (document_id = ANY(%s))", (D.document_id.tolist(),))
            removed = cur.rowcount
            cur.executemany("""INSERT INTO knowledge.documents (document_id,title,body,source_record_id,category,category_method,source_name,
                provenance,verification_status,answer_label,version,effective_date,source_url,source_page,needs_fact_check,fact_check_reasons,
                product_ambiguous,language) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (document_id) DO UPDATE SET title=EXCLUDED.title, body=EXCLUDED.body, category=EXCLUDED.category,
                verification_status=EXCLUDED.verification_status, answer_label=EXCLUDED.answer_label, effective_date=EXCLUDED.effective_date,
                source_url=EXCLUDED.source_url, source_page=EXCLUDED.source_page, needs_fact_check=EXCLUDED.needs_fact_check,
                fact_check_reasons=EXCLUDED.fact_check_reasons, product_ambiguous=EXCLUDED.product_ambiguous, ingested_at=now()""",
                [tuple(r[c] for c in DOC_COLS) for r in D.to_dict("records")])
            cur.execute("DELETE FROM knowledge.document_chunks WHERE NOT (chunk_id = ANY(%s))", (C.chunk_id.tolist(),))
            rows = [(r.chunk_id, r.document_id, int(r.chunk_index), r.chunk_text, r.metadata, vec_literal(E[i])) for i, r in enumerate(C.itertuples())]
            cur.executemany("""INSERT INTO knowledge.document_chunks (chunk_id,document_id,chunk_index,chunk_text,metadata,embedding)
                VALUES (%s,%s,%s,%s,%s::jsonb,%s::vector) ON CONFLICT (chunk_id) DO UPDATE SET chunk_text=EXCLUDED.chunk_text,
                metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding""", rows)
            cur.execute("INSERT INTO knowledge.embedding_runs (model_id,backend,is_standin,dimension,chunk_count,manifest) VALUES (%s,%s,%s,%s,%s,%s)",
                        (man["model_id"], man["backend"], man["is_standin"], man["dimension"], len(C), json.dumps(man)))
        conn.commit()
        nd = conn.execute("SELECT count(*) FROM knowledge.documents").fetchone()[0]
        nc = conn.execute("SELECT count(*) FROM knowledge.document_chunks").fetchone()[0]
    print(f"ingested: {nd} documents, {nc} chunks, model {man['model_id']} (dim {man['dimension']}); removed stale documents: {removed}")
    assert nd == len(D) and nc == len(C)

if __name__ == "__main__": main()

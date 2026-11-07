"""Create (or verify) the knowledge schema with the right vector size.
    python -m rag.db.migrate --dimension 384            # explicit
    python -m rag.db.migrate --manifest rag/data/embeddings/<tag>/manifest.json
    add --recreate to drop and rebuild the knowledge tables (needed when you change the model/dimension)
Needs DATABASE_URL (see .env.example). Safe to re-run: it refuses to continue if the existing column has a different size."""
import argparse, json, os
from pathlib import Path
import psycopg

def dsn() -> str:
    url = os.environ["DATABASE_URL"]            # postgresql+psycopg://user:pass@host:5432/db
    return url.replace("postgresql+psycopg://", "postgresql://")

def column_dimension(conn):
    r = conn.execute("""SELECT a.atttypmod FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='knowledge' AND c.relname='document_chunks' AND a.attname='embedding' AND NOT a.attisdropped""").fetchone()
    return r[0] if r else None            # for pgvector, atttypmod is the dimension

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dimension", type=int); ap.add_argument("--manifest")
    ap.add_argument("--recreate", action="store_true"); a = ap.parse_args()
    dim = a.dimension or (json.loads(Path(a.manifest).read_text())["dimension"] if a.manifest else None) \
          or (int(os.environ["EMBEDDING_DIMENSION"]) if os.getenv("EMBEDDING_DIMENSION") else None)
    if not dim: raise SystemExit("Give --dimension, --manifest or EMBEDDING_DIMENSION. The size is never guessed.")
    with psycopg.connect(dsn(), autocommit=False) as conn:
        if a.recreate:
            conn.execute("DROP TABLE IF EXISTS knowledge.document_chunks, knowledge.documents, knowledge.embedding_runs CASCADE")
        existing = column_dimension(conn)
        if existing and existing != dim:
            raise SystemExit(f"Column is vector({existing}) but you asked for {dim}. Re-run with --recreate to rebuild.")
        sql = (Path(__file__).parent / "001_knowledge_schema.sql.tpl").read_text().replace("__DIM__", str(dim))
        conn.execute(sql); conn.commit()
        print(f"knowledge schema ready; embedding column = vector({column_dimension(conn)})")

if __name__ == "__main__": main()

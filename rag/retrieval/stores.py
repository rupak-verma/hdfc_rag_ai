"""Two interchangeable chunk stores with the same four methods:
   chunk_texts() -> (ids, texts) | vector_search(qvec, k) | keyword_fts(query, k) | fetch(ids) -> {id: record}
FileStore  = CSV + .npy files (no database; used by tests, Colab and quick demos)
PgStore    = PostgreSQL + pgvector (what the backend uses)
Both only ever expose documents whose status is DEMO_LABELLED or VERIFIED_PUBLIC."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd

ALLOWED = ("DEMO_LABELLED", "VERIFIED_PUBLIC")
REC_FIELDS = ["chunk_id", "document_id", "chunk_index", "chunk_text", "title", "category", "source_name",
              "verification_status", "answer_label", "source_url", "effective_date", "product_ambiguous", "needs_fact_check"]

class FileStore:
    kind = "file"
    def __init__(self, embeddings_dir: str, approved_dir: str = "rag/data/approved"):
        a, e = Path(approved_dir), Path(embeddings_dir)
        C = pd.read_csv(a / "canonical_chunks.csv", dtype=str).fillna("")
        D = pd.read_csv(a / "canonical_documents.csv", dtype=str).fillna("")
        self.manifest = json.loads((e / "manifest.json").read_text())
        E = np.load(e / "embeddings.npy"); ids = json.loads((e / "chunk_ids.json").read_text())
        assert ids == C.chunk_id.tolist(), "vectors do not match chunks"
        df = C.merge(D[["document_id", "title", "category", "source_name", "verification_status", "answer_label",
                        "source_url", "effective_date", "product_ambiguous", "needs_fact_check"]], on="document_id")
        keep = df.verification_status.isin(ALLOWED).to_numpy()
        self.df, self.E = df[keep].reset_index(drop=True), E[keep]
        self._by_id = {r["chunk_id"]: r for r in self.df.to_dict("records")}
    def chunk_texts(self): return self.df.chunk_id.tolist(), self.df.chunk_text.tolist()
    def vector_search(self, qvec, k):
        s = self.E @ qvec
        order = sorted(range(len(s)), key=lambda i: (-s[i], self.df.chunk_id.iat[i]))[:k]   # deterministic ties
        return [(self.df.chunk_id.iat[i], float(s[i])) for i in order]
    def keyword_fts(self, query, k): raise NotImplementedError("FTS needs PostgreSQL; use the BM25 keyword backend")
    def fetch(self, ids): return {i: self._by_id[i] for i in ids}

class PgStore:
    kind = "pg"
    def __init__(self, url: str | None = None):
        import psycopg
        self._dsn = (url or os.environ["DATABASE_URL"]).replace("postgresql+psycopg://", "postgresql://")
        self._conn = psycopg.connect(self._dsn, autocommit=True)
        r = self._conn.execute("SELECT model_id, dimension, is_standin FROM knowledge.embedding_runs ORDER BY run_id DESC LIMIT 1").fetchone()
        self.manifest = dict(model_id=r[0], dimension=r[1], is_standin=r[2]) if r else {}
    def _q(self, sql, params=()): return self._conn.execute(sql, params).fetchall()
    def chunk_texts(self):
        rows = self._q("""SELECT c.chunk_id, c.chunk_text FROM knowledge.document_chunks c JOIN knowledge.documents d USING (document_id)
                          WHERE d.verification_status = ANY(%s) ORDER BY c.chunk_id""", (list(ALLOWED),))
        return [r[0] for r in rows], [r[1] for r in rows]
    def vector_search(self, qvec, k):
        lit = "[" + ",".join(f"{x:.7g}" for x in qvec) + "]"
        rows = self._q("""SELECT c.chunk_id, 1 - (c.embedding <=> %s::vector) AS score FROM knowledge.document_chunks c
                          JOIN knowledge.documents d USING (document_id) WHERE d.verification_status = ANY(%s)
                          ORDER BY c.embedding <=> %s::vector, c.chunk_id LIMIT %s""", (lit, list(ALLOWED), lit, k))
        return [(r[0], float(r[1])) for r in rows]
    def keyword_fts(self, query, k):
        """PostgreSQL full-text search (ts_rank_cd). This is FTS, NOT BM25."""
        rows = self._q("""SELECT c.chunk_id, ts_rank_cd(c.fts, websearch_to_tsquery('english', %s)) AS score
                          FROM knowledge.document_chunks c JOIN knowledge.documents d USING (document_id)
                          WHERE c.fts @@ websearch_to_tsquery('english', %s) AND d.verification_status = ANY(%s)
                          ORDER BY score DESC, c.chunk_id LIMIT %s""", (query, query, list(ALLOWED), k))
        return [(r[0], float(r[1])) for r in rows]
    def fetch(self, ids):
        rows = self._q("""SELECT c.chunk_id, c.document_id, c.chunk_index, c.chunk_text, d.title, d.category, d.source_name,
                          d.verification_status, d.answer_label, coalesce(d.source_url,''), coalesce(d.effective_date::text,''),
                          d.product_ambiguous, d.needs_fact_check
                          FROM knowledge.document_chunks c JOIN knowledge.documents d USING (document_id) WHERE c.chunk_id = ANY(%s)""", (list(ids),))
        return {r[0]: dict(zip(REC_FIELDS, r)) for r in rows}

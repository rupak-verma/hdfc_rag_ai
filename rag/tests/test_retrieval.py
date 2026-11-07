import json, os, re, hashlib
import numpy as np, pandas as pd, pytest
from rag.retrieval.bm25_retriever import tokenize, BM25Retriever
from rag.retrieval.hybrid_retriever import rrf_fuse
from rag.retrieval.guards import named_phrases, unknown_phrases
from rag.retrieval.retriever import RagEngine, RagUnavailable
from rag.retrieval.stores import FileStore

# ---------- tiny fake embedding model: hashed bag of words (no downloads needed) ----------
class FakeModel:
    model_id, is_standin, dimension = "fake:hash64", True, 64
    def _v(self, t):
        v = np.zeros(64, dtype=np.float32)
        for w in tokenize(t): v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        return v / (np.linalg.norm(v) + 1e-9)
    def encode_query(self, t): return self._v(t)
    def encode_passages(self, ts, batch_size=64): return np.vstack([self._v(t) for t in ts])

DOCS = [("DOC-00001", "How do I open a savings account?", "Visit a branch with your KYC documents to open a savings account."),
        ("DOC-00002", "What is a fixed deposit?", "A fixed deposit locks your money for a fixed tenure at a fixed interest rate."),
        ("DOC-00003", "How do I block my debit card?", "Call the 24x7 helpline or use net banking to block your debit card immediately."),
        ("DOC-00004", "What is the Regalia ForexPlus Card?", "The Regalia ForexPlus Card is a prepaid card for foreign currency spending."),
        ("DOC-00005", "How is the home loan EMI calculated?", "EMI depends on the loan amount, interest rate and tenure. " * 40)]   # long -> several chunks

@pytest.fixture()
def engine(tmp_path):
    from rag.ingestion.chunking import make_chunks
    appr, emb = tmp_path / "approved", tmp_path / "emb"; appr.mkdir(); emb.mkdir()
    D = pd.DataFrame([dict(document_id=i, title=t, text=a, category="General Banking", source_name="test", verification_status="DEMO_LABELLED",
                           answer_label="Demo knowledge - not confirmed HDFC Bank policy", source_url="", effective_date="",
                           product_ambiguous="False", needs_fact_check="False") for i, t, a in DOCS])
    C = pd.DataFrame([dict(chunk_id=f"CH-{i[4:]}-{j:02d}", document_id=i, chunk_index=j, chunk_text=c, char_count=len(c), metadata="{}")
                      for i, t, a in DOCS for j, c in enumerate(make_chunks(t, a))])
    D.to_csv(appr / "canonical_documents.csv", index=False); C.to_csv(appr / "canonical_chunks.csv", index=False)
    m = FakeModel(); np.save(emb / "embeddings.npy", m.encode_passages(C.chunk_text.tolist()))
    (emb / "chunk_ids.json").write_text(json.dumps(C.chunk_id.tolist()))
    (emb / "manifest.json").write_text(json.dumps(dict(model_id=m.model_id, dimension=64, is_standin=True)))
    return RagEngine(FileStore(str(emb), str(appr)), m, thresholds=dict(min_cosine=0.2, min_coverage=0.2))

# ---------- components ----------
def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("Can I block my debit cards?") == ["block", "debit", "card"]
def test_bm25_finds_exact_words_and_is_deterministic():
    r = BM25Retriever(["a", "b", "c"], ["apple pie recipe", "savings account interest", "debit card block"])
    assert r.search("block debit card", 3)[0][0] == "c" and r.search("block debit card", 3) == r.search("block debit card", 3)
    assert r.search("zzz qqq", 3) == []
def test_rrf_rewards_agreement_and_breaks_ties_by_id():
    out = rrf_fuse({"keyword": [("x", 9), ("y", 5)], "vector": [("y", .9), ("z", .5)]}, {"keyword": 1, "vector": 1}, 3)
    assert out[0][0] == "y" and {c for c, _, _ in out} == {"x", "y", "z"}
    tie = rrf_fuse({"keyword": [("b", 1)], "vector": [("a", 1)]}, {}, 2); assert [c for c, _, _ in tie] == ["a", "b"]
def test_guard_phrases():
    assert named_phrases("What is the fee on the HDFC Platinum Elite Supreme credit card?") == ["Platinum Elite Supreme"]
    assert named_phrases("What is a savings account?") == []
    assert named_phrases("What Is My Balance") == []          # leading filler words are not product names
    assert unknown_phrases("Tell me about Regalia ForexPlus Card", "the regalia forexplus card is") == []
    assert unknown_phrases("Tell me about Mega Gold Card", "the regalia forexplus card is") == ["Mega Gold Card"]

# ---------- the public contract ----------
def test_contract_shape_and_labelled_context(engine):
    r = engine.retrieve("How can I block my debit card?", 3)
    assert set(r) == {"context", "sources", "retrieval_metadata"}
    assert r["sources"][0]["document_id"] == "DOC-00003" and {"document_id", "title", "chunk_id"} <= set(r["sources"][0])
    assert "Demo knowledge - not confirmed HDFC Bank policy" in r["context"] and "DOC-00003" in r["context"]
    md = r["retrieval_metadata"]; assert md["abstained"] is False and md["returned"] == len(r["sources"]) and md["top_k"] == 3
def test_abstains_on_empty_and_whitespace(engine):
    for q in ("", "   ", None):
        r = engine.retrieve(q); assert r["context"] == "" and r["sources"] == [] and r["retrieval_metadata"]["reason"] == "empty_question"
def test_abstains_on_unknown_named_product(engine):
    r = engine.retrieve("What is the annual fee of the HDFC Platinum Elite Supreme card?")
    assert r["retrieval_metadata"]["reason"] == "unknown_named_product" and r["sources"] == []
    assert engine.retrieve("Tell me about the Regalia ForexPlus Card")["retrieval_metadata"]["reason"] == "ok"
def test_abstains_when_evidence_weak(engine):
    engine.thresholds = dict(min_cosine=0.99, min_coverage=0.99)
    r = engine.retrieve("How do I block my debit card?"); assert r["retrieval_metadata"]["reason"] == "low_confidence" and r["context"] == ""
def test_top_k_is_clamped_and_max_two_chunks_per_doc(engine):
    assert engine.retrieve("home loan EMI interest rate tenure", 999)["retrieval_metadata"]["top_k"] == 20
    srcs = engine.retrieve("home loan EMI interest rate tenure", 10)["sources"]
    assert sum(1 for s in srcs if s["document_id"] == "DOC-00005") <= 2
def test_deterministic(engine):
    q = "how do I open a savings account"; assert engine.retrieve(q, 5) == engine.retrieve(q, 5)
def test_very_long_question_is_truncated_not_crashing(engine):
    assert engine.retrieve("savings account " * 500)["retrieval_metadata"]["reason"] in ("ok", "low_confidence")
def test_embedding_model_mismatch_is_refused(engine):
    class Other(FakeModel): model_id = "other-model"
    store = engine.store; store.manifest["model_id"] = "fake:hash64"
    with pytest.raises(RagUnavailable): RagEngine(store, Other())

# ---------- PostgreSQL integration (skipped when no database is reachable) ----------
def _pg():
    url = os.getenv("DATABASE_URL")
    if not url: return None
    try:
        import psycopg; c = psycopg.connect(url.replace("postgresql+psycopg://", "postgresql://"), connect_timeout=3)
        c.execute("select 1 from knowledge.document_chunks limit 1"); c.close(); return url
    except Exception: return None
pg = pytest.mark.skipif(_pg() is None, reason="PostgreSQL with ingested data not reachable")

@pg
def test_pg_store_search_fetch_and_status_filter():
    from rag.retrieval.stores import PgStore
    s = PgStore(); ids, texts = s.chunk_texts(); assert len(ids) == len(texts) > 0
    q = np.random.RandomState(0).rand(s.manifest["dimension"]).astype(np.float32); q /= np.linalg.norm(q)
    hits = s.vector_search(q, 5); assert len(hits) == 5 and all(hits[i][1] >= hits[i + 1][1] for i in range(4))
    rec = s.fetch([hits[0][0]])[hits[0][0]]; assert rec["verification_status"] in ("DEMO_LABELLED", "VERIFIED_PUBLIC") and rec["chunk_text"]
    assert s.keyword_fts("savings account minimum balance", 3)
@pg
def test_database_refuses_verified_record_without_url():
    import psycopg
    c = psycopg.connect(_pg().replace("postgresql+psycopg://", "postgresql://"))
    with pytest.raises(psycopg.errors.CheckViolation):
        c.execute("""INSERT INTO knowledge.documents (document_id,title,body,source_record_id,source_name,provenance,verification_status,answer_label)
                     VALUES ('DOC-TEST','t','b','r','s','p','VERIFIED_PUBLIC','x')""")
    c.rollback()
@pg
def test_database_refuses_unknown_status_and_wrong_vector_size():
    import psycopg
    c = psycopg.connect(_pg().replace("postgresql+psycopg://", "postgresql://"))
    with pytest.raises(psycopg.errors.CheckViolation):
        c.execute("""INSERT INTO knowledge.documents (document_id,title,body,source_record_id,source_name,provenance,verification_status,answer_label)
                     VALUES ('DOC-TEST2','t','b','r','s','p','EXCLUDED','x')""")
    c.rollback()
    with pytest.raises(psycopg.Error):
        c.execute("INSERT INTO knowledge.document_chunks (chunk_id,document_id,chunk_index,chunk_text,embedding) VALUES ('c','DOC-00011',9,'t','[1,2,3]'::vector)")
    c.rollback()

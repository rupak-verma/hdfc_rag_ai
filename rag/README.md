# RAG module (Person 3) - knowledge base and retrieval

Answers one question for the rest of the team: **"which approved passages should the model read before it answers?"**

```python
from rag.retrieval.retriever import retrieve_context
retrieve_context("Is tax cut from my savings interest?", top_k=5)
# -> {"context": "...", "sources": [{"document_id","title","chunk_id","verification_status","score"}], "retrieval_metadata": {...}}
```
If the evidence is weak the result is **empty** (`context ""`, `sources []`, `retrieval_metadata.abstained = true`, with a `reason`:
`low_confidence`, `unknown_named_product`, `empty_question`, `no_candidates`). The caller must then say "I don't know", never guess.
If the store or model cannot be used it raises `RagUnavailable` - the backend should turn that into a controlled error reply.

## What happens, in order
`workbook -> audit -> clean + de-duplicate -> canonical documents -> chunks -> embeddings -> PostgreSQL/pgvector -> hybrid search -> retrieve_context`

| Step | Command (run from the repo root) | Needs |
|---|---|---|
| 1 Audit the workbook | `python rag/ingestion/audit_workbook.py` | pandas |
| 2 Build canonical data + chunks | `python -m rag.ingestion.build_canonical` | pandas |
| 2b Fact-check sheet (humans) | `python -m rag.ingestion.make_verification_sample`, then follow `rag/data/VERIFICATION_GUIDE.md` | people |
| 3 Choose the embedding model | `python -m rag.evaluation.compare_embedding_models` | sentence-transformers (Colab) |
| 4 Embed all chunks | `python -m rag.embeddings.generate_embeddings` | sentence-transformers |
| 5 Create tables (size is read from step 4) | `python -m rag.db.migrate --manifest rag/data/embeddings/<tag>/manifest.json` | DATABASE_URL |
| 6 Load the database | `python -m rag.ingestion.ingestion_pipeline --embeddings rag/data/embeddings/<tag>` | DATABASE_URL |
| 7 Set abstention thresholds + measure | `python -m rag.evaluation.retrieval_evaluation --calibrate` | DATABASE_URL |
| 8 Write the report | `python -m rag.evaluation.write_report rag/evaluation/results/retrieval_eval_<tag>.json docs/RAG_EVALUATION_REPORT.md` | - |
| Tests | `python -m pytest -q rag/tests` | (database tests skip if none) |

Changing the embedding model changes the vector size: use `python -m rag.db.migrate --manifest ... --recreate`, re-ingest and re-run step 7.

## Rules the code enforces
- Only `DEMO_LABELLED` or `VERIFIED_PUBLIC` records are ever searched; the database also rejects any other status, and rejects `VERIFIED_PUBLIC` without a source URL.
- Demo-labelled passages arrive in the context marked "Demo knowledge - not confirmed HDFC Bank policy"; the answering model must keep that wording.
- Stand-in (test) vectors are refused by ingestion unless `--allow-standin`; the metadata always names the embedding model.
- Question and stored vectors must come from the same model, or search refuses to run.
- The original workbook is read-only and checksum-tested; every excluded row is listed with a reason in `rag/data/cleaned/excluded_records.csv`.
- Retrieved text is data, not instructions: the prompt builder must wrap it as quoted evidence.

## Contract details to agree with Person 1

## Retrieval backend configuration

- `RAG_STORE=pg` selects the PostgreSQL/Neon pgvector store. Initialization failures raise `RagUnavailable`; there is no silent fallback to file storage.
- `RAG_STORE=file` selects the local file-backed store.
- `KEYWORD_BACKEND=bm25` uses in-process BM25 keyword retrieval.
- `KEYWORD_BACKEND=fts` uses PostgreSQL full-text search with `ts_rank_cd`.
- Both keyword backends can be combined with vector retrieval using Reciprocal Rank Fusion (RRF).
- The default keyword backend is `bm25`.

The guide's response shape is respected. Two **additive** choices need the team's agreement: `sources[].verification_status` / `score`, and
`retrieval_metadata` fields `abstained`, `reason`, `confidence`, `lexical_coverage`, `embedding_model`, `unknown_terms`. The guide's file tree has no
single entry file, so `rag/retrieval/retriever.py` is added as the one door to the module.

## What has and has not been tested

| Tested here | Actual result |
|---|---|
| Automated tests | 28 passed, 3 skipped using `python -m pytest -q rag/tests` |
| Real embedding model | BAAI/bge-small-en-v1.5 tested; actual vector dimension: 384 |
| Embedding dataset | 3,604 chunks embedded using the selected model |
| Embedding comparison | MiniLM, BGE-small-en-v1.5 and MPNet compared; BGE selected based on development hybrid MRR |
| Neon PostgreSQL | Hosted database connection, migration and ingestion tested |
| pgvector | Neon pgvector extension version 0.8.6 |
| Database ingestion | 3,226 documents and 3,604 chunks ingested |
| Retrieval | Neon-backed retrieval tested with BGE embeddings |
| Evaluation | Retrieval evaluation completed with the real BGE embedding model |

### Retrieval evaluation results

Test evaluation sample: 29 questions.

| Metric | Result |
|---|---:|
| Hybrid Recall@5 | 0.793 |
| Hybrid MRR | 0.689 |
| Answerable question pass rate | 0.966 |
| False abstention rate | 0.034 |
| Absent-answer abstention | 0.833 |

The evaluation sample is small and should not be interpreted as production-level validation. Two personal-data questions were not correctly abstained on; personal-data and account-specific questions must be routed through the approved backend/database authorization flow rather than answered using RAG.

Full results: `docs/RAG_EVALUATION_REPORT.md`.

The evaluation questions were Claude-drafted. A separate completed human-review decision file has not been located. These results are therefore automated evaluation results, not independent human evaluation.

## Known limitations

- All 3,226 retained records remain `DEMO_LABELLED`; none has been promoted to `VERIFIED_PUBLIC`. Do not present this dataset as confirmed official HDFC Bank policy.
- 1,240 records are flagged for fact-checking, and 280 are flagged as product-ambiguous.
- No version or effective date exists in the available source records, so outdated-policy detection is not supported.
- The retrieval evaluation uses a small sample of 29 test questions and 42 development questions for threshold calibration.
- The evaluation questions were Claude-drafted and have not been independently validated through a recorded human-review decision file.
- The unknown-product guard has limitations: some lowercase invented product names may not be detected.
- Category labels for 2,216 records were assigned using keyword rules rather than manual classification.
- Personal-data and account-specific questions require the backend's approved, authorized database-query flow; RAG is not an authorization mechanism.
- Production readiness requires broader evaluation, independent content verification and integration testing.

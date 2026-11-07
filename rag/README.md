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
The guide's response shape is respected. Two **additive** choices need the team's agreement: `sources[].verification_status` / `score`, and
`retrieval_metadata` fields `abstained`, `reason`, `confidence`, `lexical_coverage`, `embedding_model`, `unknown_terms`. The guide's file tree has no
single entry file, so `rag/retrieval/retriever.py` is added as the one door to the module.

## What has and has not been tested
| Tested here | NOT tested here |
|---|---|
| 31 automated tests: cleaning, chunking, verification rules, RRF, BM25, guard, contract shape, abstention, determinism, model-mismatch refusal | **The real embedding model** - model downloads are blocked in the build environment; only a 300-dimension stand-in was used |
| PostgreSQL 16 + pgvector 0.6.0: schema, ingestion (idempotent), vector and full-text search, database constraints | Neon (hosted) - it supports pgvector, but the connection was not tried |
| All evaluation scripts run end to end on the stand-in | Any number in `docs/RAG_EVALUATION_REPORT_STANDIN_TEST.md` is a test of the plumbing, **not a result** |

## Known limitations (state these in the report)
- Records are demo-labelled; none is verified until reviewers fill `verification_decisions.csv`.
- No version or effective date exists in the sources, so no "outdated policy" handling is possible yet.
- The test questions are a Claude-drafted draft (83 questions), small, and written from the same documents.
- The unknown-product guard only catches capitalised multi-word names; a lower-case invented product slips past.
- Abstention thresholds are model-specific and set on only 42 dev questions.
- Category labels for 2,216 records come from keyword rules.

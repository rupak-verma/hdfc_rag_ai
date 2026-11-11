# Embedding model comparison

Date: 2026-11-10  |  questions: dev 42 / test 41

| Model | Dim | Embed time (s) | Dev hybrid MRR | Dev hybrid R@5 | Test vector R@5 | Test hybrid MRR | Test hybrid R@5 |
|---|---|---|---|---|---|---|---|
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 314.4 | 0.796 | 0.867 | 0.862 | 0.677 | 0.828 |
| BAAI/bge-small-en-v1.5 | 384 | 608.1 | 0.837 | 0.9 | 0.862 | 0.689 | 0.793 |
| sentence-transformers/all-mpnet-base-v2 | 768 | 3480.2 | 0.783 | 0.9 | 0.897 | 0.652 | 0.793 |

**Recommendation (highest dev hybrid MRR, ties -> smaller vector): `BAAI/bge-small-en-v1.5` (dimension 384).**

Next: set EMBEDDING_MODEL and EMBEDDING_DIMENSION in .env, then run generate_embeddings, db.migrate, ingestion_pipeline and `retrieval_evaluation --calibrate`.
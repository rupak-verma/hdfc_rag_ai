# RAG retrieval evaluation - BAAI/bge-small-en-v1.5

Date: 2026-11-11 | vector size: 384 | store: file | keyword search: BM25 | chunks searched: 2,074

Questions: 42 dev (used only to set abstention thresholds) + 41 test. The evaluation set is a Claude-drafted DRAFT awaiting human review. The answerable retrieval results below use 29 test questions.

## 1. Finding the right document (answerable test questions)

| Search method | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| Keyword (BM25) | 51.7% | 55.2% | 55.2% | 0.535 |
| Vector (meaning) | 37.9% | 55.2% | 62.1% | 0.486 |
| Hybrid (RRF) | 44.8% | 51.7% | 55.2% | 0.503 |

n = 29 answerable test questions. This is a small sample, so interpret the results cautiously.

## 2. Knowing when not to answer (abstention)

Thresholds (set on dev): similarity >= 0.73, word coverage >= 0.0; plus the unknown-product guard.

| Measure | Test result |
|---|---:|
| Answerable questions wrongly refused (false abstain) | 6.9% |
| Questions with no answer correctly refused | 91.7% (95% interval 64.6-98.5%) |
| Same, excluding personal-data questions | 100.0% |
| Correct document in top 5 AND not refused | 51.7% |

Refused correctly, by kind: fake_product 1/1, near_topic 4/4, off_topic 5/5, personal_data 1/2.

Hybrid Recall@5 by question style: keyword-style 4/10, paraphrase-style 12/19.

## 3. Dataset and curation

- 2,003 documents retained and 2,074 chunks generated.
- 1,225 records marked QUARANTINE_UNVERIFIED_CLAIM were excluded.
- 9 greeting or too-short records and 1 record with a missing answer were also excluded.
- 98 retained records are flagged as needing fact-checking.
- 140 retained records are flagged as product-ambiguous.
- Retained documents are DEMO_LABELLED, not verified HDFC policy.

## 4. How to read this honestly

- Questions were drafted from the same project material used to prepare the dataset; real customers may phrase questions differently.
- The evaluation set is a Claude-drafted draft awaiting human review. It is not an independent human evaluation.
- Documents are demo-labelled, not verified HDFC policy, so a correct retrieval is not proof that the retrieved information is current or authoritative.
- The test sample is small. The results are prototype measurements, not production-level validation.
- One of two personal-data questions was not correctly abstained on. Personal-data and account-specific questions must be routed through the approved backend/database authorization flow rather than answered using RAG.
- Thresholds depend on the embedding model; changing the model requires running `--calibrate` again.
- Retrieval scores are lower than the earlier run because quarantined records were excluded from the corrected dataset. Do not reintroduce quarantined records to improve evaluation scores.

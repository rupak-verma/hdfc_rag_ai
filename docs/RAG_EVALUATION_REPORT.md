# RAG retrieval evaluation - BAAI/bge-small-en-v1.5

Date 2026-11-10 | vector size 384 | store: file | keyword search: bm25 | chunks searched: 3,604
Questions: 42 dev (used only to set the abstention thresholds) + 41 test (used for every headline number). The set is a Claude-drafted DRAFT awaiting human review.

## 1. Finding the right document (test questions that have an answer)

| Search method | Recall@1 | Recall@3 | Recall@5 | Recall@5 95% interval | MRR |
|---|---|---|---|---|---|
| Keyword (BM25) | 62.1% | 72.4% | 75.9% | 58-88% | 0.686 |
| Vector (meaning) | 55.2% | 75.9% | 86.2% | 69-94% | 0.693 |
| Hybrid (RRF) | 58.6% | 75.9% | 79.3% | 62-90% | 0.689 |

n = 29 answerable test questions, so one question moves a percentage by about 3 points: read the intervals, not just the point values.

## 2. Knowing when not to answer (abstention)

Thresholds (set on dev): similarity >= 0.73, word coverage >= 0.0; plus the unknown-product guard.

| Measure | Test result |
|---|---|
| Answerable questions wrongly refused (false abstain) | 3.4% |
| Questions with no answer correctly refused | 83.3% (95% interval 55-95%) |
| ...same, excluding personal-data questions (routed to the database before RAG) | 100.0% |
| Correct document in top 5 AND not refused | 75.9% |

Refused correctly, by kind: fake_product 1/1, near_topic 4/4, off_topic 5/5, personal_data 0/2.
Hybrid Recall@5 by question style: kw 9/10, para 14/19 (kw = shares key words, para = paraphrase).

## 3. Where it fails

Absent questions that were wrongly answered:
- Q056 (personal_data): What is my current account balance?
- Q080 (personal_data): When will my last transaction reach the beneficiary?

Answerable questions whose document was not in the top 5 (all splits): 9. First five:
- Q007: Is tax cut from the interest I earn on my savings account? (wanted DOC-00098, got DOC-01653, DOC-01939, DOC-00080)
- Q014: Where can I use the Regalia ForexPlus Card? (wanted DOC-01402, got DOC-01414, DOC-01412, DOC-01417)
- Q017: Can I divide one fixed deposit into several smaller deposits? (wanted DOC-00958, got DOC-01900, DOC-01889, DOC-01896)
- Q019: Can I set up a standing instruction for postpaid bill payments too? (wanted DOC-01506, got DOC-01507, DOC-01507, DOC-01493)
- Q027: Can I take money out of my recurring deposit before maturity? (wanted DOC-01871, got DOC-01863, DOC-01856, DOC-01867)

## 4. How to read this honestly
- Questions were written by the same side that prepared the data, from the same documents; real customers will phrase things differently.
- Documents are demo-labelled, not verified HDFC policy, so a correct retrieval is not a correct fact.
- Thresholds depend on the model: change the model and you must re-run `--calibrate`.

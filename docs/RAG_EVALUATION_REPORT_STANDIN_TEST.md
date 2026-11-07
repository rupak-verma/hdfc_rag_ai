> **STAND-IN MODEL - NOT PROJECT RESULTS.** These numbers come from averaged word vectors (en_core_web_md), used only to test the pipeline
> where real model downloads are blocked. Re-run with the real embedding model before quoting anything below.

# RAG retrieval evaluation - standin:spacy-en_core_web_md

Date 2026-11-04 | vector size 300 | store: pg | keyword search: bm25 | chunks searched: 3,604
Questions: 42 dev (used only to set the abstention thresholds) + 41 test (used for every headline number). The set is a Claude-drafted DRAFT awaiting human review.

## 1. Finding the right document (test questions that have an answer)

| Search method | Recall@1 | Recall@3 | Recall@5 | Recall@5 95% interval | MRR |
|---|---|---|---|---|---|
| Keyword (BM25) | 62.1% | 72.4% | 75.9% | 58-88% | 0.686 |
| Vector (meaning) | 27.6% | 48.3% | 51.7% | 34-69% | 0.382 |
| Hybrid (RRF) | 44.8% | 62.1% | 65.5% | 47-80% | 0.553 |

n = 29 answerable test questions, so one question moves a percentage by about 3 points: read the intervals, not just the point values.

## 2. Knowing when not to answer (abstention)

Thresholds (set on dev): similarity >= 0.88, word coverage >= 0.0; plus the unknown-product guard.

| Measure | Test result |
|---|---|
| Answerable questions wrongly refused (false abstain) | 20.7% |
| Questions with no answer correctly refused | 75.0% (95% interval 47-91%) |
| ...same, excluding personal-data questions (routed to the database before RAG) | 90.0% |
| Correct document in top 5 AND not refused | 51.7% |

Refused correctly, by kind: fake_product 1/1, near_topic 3/4, off_topic 5/5, personal_data 0/2.
Hybrid Recall@5 by question style: kw 8/10, para 11/19 (kw = shares key words, para = paraphrase).

## 3. Where it fails

Absent questions that were wrongly answered:
- Q056 (personal_data): What is my current account balance?
- Q071 (near_topic): What is the phone number of the Chennai branch manager?
- Q080 (personal_data): When will my last transaction reach the beneficiary?

Answerable questions whose document was not in the top 5 (all splits): 19. First five:
- Q003: What do you call a borrower who can pay but deliberately does not? (wanted DOC-00866, got DOC-02427, DOC-00303, DOC-02074)
- Q007: Is tax cut from the interest I earn on my savings account? (wanted DOC-00098, got DOC-01653, DOC-01939, DOC-00080)
- Q008: What is a letter of credit that pays at once when shipping documents are shown? (wanted DOC-00712, got DOC-00702, DOC-01154, DOC-01672)
- Q014: Where can I use the Regalia ForexPlus Card? (wanted DOC-01402, got DOC-01414, DOC-01408, DOC-01417)
- Q015: Do I need an account with the bank to buy a Giftplus card? (wanted DOC-01282, got DOC-01270, DOC-01273, DOC-01279)

## 4. How to read this honestly
- Questions were written by the same side that prepared the data, from the same documents; real customers will phrase things differently.
- Documents are demo-labelled, not verified HDFC policy, so a correct retrieval is not a correct fact.
- Thresholds depend on the model: change the model and you must re-run `--calibrate`.

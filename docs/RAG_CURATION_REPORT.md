# RAG curation report (Person 3) - Step 2

Inputs: `rag/data/raw/HDFC_Banking_LLM_Final_Curated_Dataset_v1.xlsx` (SHA-256 `79c99ac6...0993`, unchanged; a test checks this).
Reproduce: `python -m rag.ingestion.build_canonical` then `python -m pytest -q rag/tests`.

## Decisions (made 2026-11-04; delegated by Preethi to Claude; mentor/Person 1 review pending)
| # | Decision | Why |
|---|---|---|
| 1 | Statuses: `VERIFIED_PUBLIC`, `DEMO_LABELLED`, `EXCLUDED` | The workbook forbids treating anything as approved without review; demo content must be labelled, never passed off as HDFC policy |
| 2 | Start from `FAQ_Canonical` (3,238 rows) | Already merges and de-duplicates the three FAQ sources and keeps source and flag columns |
| 3 | Excluded entirely: 7,000 synthetic "HDFC policy" rows (`RAG_Candidates`, same rows as `Suite_RAG_Quarantine`) | Quarantined, invented text |
| 4 | Excluded from RAG: `Privacy_Security_Clean` (60) | Fine-tuning rows built on an unverified HDFC policy snippet |
| 5 | Held out: `Cybersecurity_Clean` (55) | Workbook requires a security specialist review first |
| 6 | Remove a row only if question AND answer match another row; never remove by question alone | The HDFC FAQ repeats one question per card/plan (answer names the product), so question-only de-duplication would delete real, different answers |
| 7 | Verified sample = 150 records (80 figure/fee/limit, 40 contact/URL, 30 unflagged controls), 75 per reviewer | Small enough to finish in about 3 hours; the control group measures the error rate of unflagged records |
| 8 | Nothing is promoted to `VERIFIED_PUBLIC` without a public URL and a reviewer name | Keeps "verified" meaningful |

## Result
| Measure | Value |
|---|---|
| Rows in / kept / excluded | 3,238 / 3,226 / 12 (11 greetings or under 20 characters, 1 missing answer) |
| Status of kept records | 3,226 `DEMO_LABELLED`, 0 `VERIFIED_PUBLIC` (until reviewers finish) |
| Chunks | 3,604 (200 documents split into several chunks; largest chunk 892 characters, median 214) |
| Flagged for fact-check | 1,240 (figure/fee/limit/rate, contact or URL, or an HDFC-specific claim) |
| Product-ambiguous questions ("this card...") | 280 - the retriever must show the product named in the answer |
| Category source | 1,010 from the workbook; 2,216 assigned by a keyword rule (lower trust, flagged in `category_method`) |

## Data-quality finding: broken rupee signs (repaired)
81 answers (53 records) held "â‚1" instead of the rupee sign: the sign's UTF-8 bytes were misread as Windows-1252 and the superscript one then flattened to a plain "1". So "₹5 lakh" read "â‚15 lakh" - a customer would have been told the deposit-insurance cover is 15 lakh. The pattern is always followed by that extra "1" (checked on all 81), and the repaired values match known figures. `clean_text` now repairs it and a test guards it. The workbook's own claim that encodings were repaired is therefore incomplete; report this to whoever owns the workbook.

## Outputs (all in `rag/data/approved/` unless stated)
`canonical_documents.csv`, `canonical_chunks.csv` / `.jsonl`, `data_dictionary.csv`, `curation_summary.json`, `verification_sample.csv`; `rag/data/cleaned/excluded_records.csv` (every dropped row and why).
IDs are stable: `DOC-<row position in FAQ_Canonical>`, chunk `CH-<same number>-<part>`; re-running never renumbers.

## Known limits (state these in the report)
- All provenance is "synthetic or unverified"; only the 150-row sample can be checked in the time available. Until reviewed, no answer may be presented as confirmed HDFC policy.
- No record has a version or effective date (the sources have none); content may be out of date.
- Category rules are keyword-based and will misfile some records.
- De-duplication catches exact question+answer repeats only, not paraphrases.
- 20 answers exceed 2,000 characters (web-page style text) and are split into many chunks.

# RAG workbook inventory and curation plan (Person 3)

**Status: decisions MADE on 2026-11-04 (delegated by Preethi to Claude; to be shown to the mentor and Person 1). See `RAG_CURATION_REPORT.md`.**
Source: `rag/data/raw/HDFC_Banking_LLM_Final_Curated_Dataset_v1.xlsx` (33 sheets, read-only copy).
SHA-256: `79c99ac6c03936143c32633c26591276a190c1085431f8bc1dc63d1cd50b0993`
Reproduce: `python rag/ingestion/audit_workbook.py` (tables in `rag/data/inventory/`).

## 1. Headline findings
1. **No record is approved for RAG today.** `RAG_Candidates` (8,000 rows) is 7,000 quarantined + 1,000 `REQUIRES_FACT_CHECK`; every row has `rag_eligible` = NO or NO_UNTIL_VERIFIED. The workbook README says the same.
2. **7,000 of the 8,000 are synthetic "HDFC policy" text** (`synthetic_unverified_hdfc_policy_text`) and are word-for-word the same 7,000 rows as `Suite_RAG_Quarantine`. They are one pool, not two.
3. **`BankFAQ_Clean` is 750 rows but only 10 distinct answers** (740 repeats; the questions differ only by "query N"). `FAQ_Canonical` already reduced it to 10.
4. **`FAQ_Canonical` (3,238 rows) is already the merged FAQ pool:** 2,228 from HDFC_Faq + 1,000 knowledge base + 10 bank_faq. All rows are `needs_content_review`.
5. `HDFC_FAQ_Clean` has 115 answers over 1,000 characters (longest 10,541), so chunking matters; 11 answers under 20 characters (greetings and fragments); 80 repeated answers; 1 missing answer.
6. **Time-sensitive content needs fact-checking first:** 198 of 1,000 knowledge-base answers and 527 of 2,240 HDFC FAQ answers mention a rate, fee, limit or figure.
7. Overlap between the three FAQ sheets is tiny (1-6 shared questions), so merging them creates little double counting.

## 2. Sheet-by-sheet decisions (PROPOSED)
| Sheet | Rows | What it is | Proposed decision | Reason |
|---|---|---|---|---|
| FAQ_Canonical | 3,238 | Merged, partly deduped FAQ pool | **Starting table** for the canonical dataset | Already carries source, flags and review status |
| HDFC_FAQ_Clean | 2,240 | Unverified HDFC-style FAQ | Source inside FAQ_Canonical; `DEMO_LABELLED` after cleaning; verify a sample, figures first | Biggest and most on-topic; provenance unverified |
| KnowledgeBase_Clean | 1,000 | General banking knowledge | Source inside FAQ_Canonical; `DEMO_LABELLED`; fact-check the 198 figure answers | Generic text, time-sensitive claims |
| BankFAQ_Clean | 750 | Synthetic FAQ | Keep the 10 distinct answers only | 740 repeats add nothing |
| Cybersecurity_Clean | 55 | Synthetic security Q&A | **HOLD** until a security review | Workbook says specialist review required |
| Privacy_Security_Clean | 60 | Fine-tuning rows with an "unverified HDFC policy" snippet (29 distinct) | **EXCLUDE from RAG** (belongs to fine-tuning; safety behaviour lives in guardrails) | Snippets are unverified HDFC policy |
| RAG_Candidates (7,000 synthetic) | 7,000 | Synthetic HDFC policy text | **EXCLUDED** | Quarantined |
| RAG_Candidates (1,000 KB) | 1,000 | Same source as KnowledgeBase_Clean | Do not double count; handled via KnowledgeBase_Clean | Same source file (to confirm in cleaning) |
| Suite_RAG_Quarantine | 7,000 | Duplicate of the 7,000 above | **EXCLUDED** | Quarantined |
| All other sheets | - | Intent, SFT, chat, eval, governance, Banking77 | Not RAG sources | Belong to Person 2 / testing |

## 3. Proposed status values
- `VERIFIED_PUBLIC` - checked against a named public source; URL and date recorded.
- `DEMO_LABELLED` - usable; any answer built on it must say it is demo knowledge, not confirmed HDFC policy.
- `EXCLUDED` - never loaded.

Expected pool after cleaning: roughly 3,100-3,200 FAQ records, all starting as `DEMO_LABELLED` or `EXCLUDED`, with a small verified subset promoted after human review.

## 4. Decisions taken
1. The three statuses and the table above are adopted.
2. Verified sample size: 150 records, 2 reviewers (see `rag/data/VERIFICATION_GUIDE.md`).
3. The 55 cybersecurity rows stay out until a security review.

## 5. Known limits of this audit
- Duplicate counts use a simple text normalisation; paraphrased duplicates are not detected.
- "Mentions a figure" is a keyword pattern, so it over- and under-counts a little.
- Nothing here checks whether any answer is factually correct.

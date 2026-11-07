# How to fact-check the 150 sample records (about 2 minutes each)

File: `rag/data/approved/verification_sample.csv` - reviewer A and B each have 75 rows.

For each row:
1. Search for the claim on a **public official source**: the bank's own website (hdfcbank.com) for product answers, rbi.org.in for rules and limits. Do not use blogs or forums.
2. Fill the columns:
   - `matches`: **Y** (the source says the same, including the numbers), **PARTIAL** (mostly right, some detail differs or is out of date), **N** (source contradicts it), or leave empty if you could not decide.
   - `source_url`: the exact page. `source_page`: section name or page number. `source_date`: date shown on the page, if any.
   - `checked_by` (your name) and `date_checked`.
   - `notes`: one line on what differed.
3. Never mark **Y** without a URL. The pipeline rejects it.
4. Rows in group `control_unflagged` are the control: their error rate tells us how reliable the *unflagged* records are.

When done, save the file as `rag/data/approved/verification_decisions.csv` (same columns) and re-run `python -m rag.ingestion.build_canonical`. Y rows become `VERIFIED_PUBLIC`, N rows are excluded, PARTIAL rows stay `DEMO_LABELLED`.

"""Step 2 - build the canonical RAG dataset from the workbook's FAQ_Canonical sheet.
Run from repo root:  python rag/ingestion/build_canonical.py
Reads  : rag/data/raw/<workbook>.xlsx            (never modified)
Writes : rag/data/cleaned/excluded_records.csv   (every dropped row + reason)
         rag/data/approved/canonical_documents.csv, canonical_chunks.csv / .jsonl
         rag/data/approved/data_dictionary.csv, curation_summary.json
Deterministic: same input -> same IDs and files."""
import json, hashlib
from pathlib import Path
import pandas as pd
from rag.ingestion.data_cleaning import (clean_text, clean_title, norm_key, fact_check_reasons,
        is_product_ambiguous, assign_category, DEVANAGARI)
from rag.ingestion.chunking import make_chunks
from rag.ingestion.verification import apply_decisions

WB = Path("rag/data/raw/HDFC_Banking_LLM_Final_Curated_Dataset_v1.xlsx")
CLEAN, APPR = Path("rag/data/cleaned"), Path("rag/data/approved")
for p in (CLEAN, APPR): p.mkdir(parents=True, exist_ok=True)
LABEL = "Demo knowledge - not confirmed HDFC Bank policy"

src = pd.read_excel(WB, sheet_name="FAQ_Canonical")
rows_in = len(src)
kept, excluded, seen = [], [], set()

def exclude(r, reason): excluded.append({**r._asdict(), "exclusion_reason": reason})

for pos, r in enumerate(src.itertuples(index=False), start=1):
    q, a = clean_title(r.question), clean_text(r.answer)
    if not a or a.lower() == "nan":                       exclude(r, "missing_answer"); continue
    if r.training_eligibility == "EXCLUDE_GREETING_OR_SHORT" or len(a) < 20:
                                                          exclude(r, "greeting_or_too_short"); continue
    if DEVANAGARI.search(q + a):                          exclude(r, "non_english_out_of_scope"); continue
    key = (norm_key(q), norm_key(a))
    if key in seen:                                       exclude(r, "exact_duplicate_question_and_answer"); continue
    seen.add(key)
    kept.append((pos, r, q, a))

docs, chunks = [], []
for pos, r, q, a in kept:
    doc_id = f"DOC-{pos:05d}"   # = row position in FAQ_Canonical, so IDs never shift
    reasons = fact_check_reasons(q, a)
    docs.append(dict(
        document_id=doc_id, source_record_id=r.record_id, title=q, text=a,
        category=assign_category(q, a) if r.category == "Uncategorized" else r.category,
        category_method="keyword_rule" if r.category == "Uncategorized" else "workbook",
        source_name=r.source_file, source_sheet="FAQ_Canonical", provenance=r.provenance,
        verification_status="DEMO_LABELLED", answer_label=LABEL,
        version="unversioned", effective_date="", source_url="", source_page="",
        needs_fact_check=bool(reasons), fact_check_reasons="|".join(reasons),
        product_ambiguous=is_product_ambiguous(q), language="en", char_count=len(a)))

D, X = pd.DataFrame(docs), pd.DataFrame(excluded)
DEC = Path("rag/data/approved/verification_decisions.csv")     # filled in by humans (optional)
warnings = []
if DEC.exists():
    D, removed, warnings = apply_decisions(D, pd.read_csv(DEC, dtype=str))
    X = pd.concat([X, removed.rename(columns={"title": "question", "text": "answer"})], ignore_index=True)
for w in warnings: print("VERIFICATION WARNING:", w)
for d in D.itertuples(index=False):
    for j, ctext in enumerate(make_chunks(d.title, d.text)):
        chunks.append(dict(chunk_id=f"CH-{d.document_id[4:]}-{j:02d}", document_id=d.document_id, chunk_index=j,
                           chunk_text=ctext, char_count=len(ctext)))
C = pd.DataFrame(chunks)
meta_cols = ["category", "source_name", "verification_status", "answer_label", "needs_fact_check", "product_ambiguous"]
C = C.merge(D[["document_id"] + meta_cols], on="document_id")
C["metadata"] = C[meta_cols].apply(lambda s: json.dumps({k: (bool(v) if isinstance(v, (bool,)) or str(v) in ("True","False") else v) for k, v in s.items()}), axis=1)
C = C.drop(columns=meta_cols)

D.to_csv(APPR/"canonical_documents.csv", index=False)
C.to_csv(APPR/"canonical_chunks.csv", index=False)
C.to_json(APPR/"canonical_chunks.jsonl", orient="records", lines=True, force_ascii=False)
X.to_csv(CLEAN/"excluded_records.csv", index=False)

dictionary = [
 ("document_id","DOC-00001 style stable ID, assigned in source order","text"),
 ("source_record_id","record_id in the workbook's FAQ_Canonical sheet (traceability)","text"),
 ("title","The question (or title) as cleaned","text"),
 ("text","The answer as cleaned","text"),
 ("category","Workbook category; for 'Uncategorized' rows a keyword-rule category","text"),
 ("category_method","'workbook' or 'keyword_rule' (rule-assigned, lower trust)","text"),
 ("source_name","Original supplied file the record came from","text"),
 ("source_sheet","Workbook sheet used","text"),
 ("provenance","Workbook provenance tag (all are synthetic_or_unverified_source)","text"),
 ("verification_status","DEMO_LABELLED (usable, must be labelled) or VERIFIED_PUBLIC (checked against a public URL)","enum"),
 ("answer_label","Wording any answer built on this record must carry","text"),
 ("version","Document version; 'unversioned' because the sources carry none","text"),
 ("effective_date","Date the content applies from; blank because the sources carry none","date"),
 ("source_url","Public URL checked during verification; blank until verified","text"),
 ("source_page","Page/section in the public source; blank until verified","text"),
 ("needs_fact_check","True if the answer mentions a figure/fee/limit, a contact/URL, or an HDFC-specific claim","bool"),
 ("fact_check_reasons","Which of those triggered the flag","text"),
 ("product_ambiguous","True if the question says 'this card/plan...' so the product is only named in the answer","bool"),
 ("language","Assumed English; non-English records are excluded","text"),
 ("char_count","Length of the answer in characters","int"),
 ("chunk_id","CH-<doc number>-<part> stable chunk ID, e.g. CH-00012-00","text"),
 ("chunk_index","Position of the chunk inside its document, from 0","int"),
 ("chunk_text","Question line + answer part; this is what is embedded and searched","text"),
 ("metadata","JSON copy of category, source, status, label and flags (goes to the pgvector table)","json"),
]
pd.DataFrame(dictionary, columns=["field","meaning","type"]).to_csv(APPR/"data_dictionary.csv", index=False)

summary = dict(
    workbook_sha256=hashlib.sha256(WB.read_bytes()).hexdigest(), source_sheet="FAQ_Canonical",
    rows_in=rows_in, documents_kept=len(D), rows_excluded=len(X),
    exclusion_reasons=X.exclusion_reason.value_counts().to_dict(), chunks=len(C),
    docs_split_into_multiple_chunks=int((C.groupby("document_id").size() > 1).sum()),
    max_chunk_chars=int(C.char_count.max()), median_chunk_chars=int(C.char_count.median()),
    needs_fact_check=int(D.needs_fact_check.sum()), product_ambiguous=int(D.product_ambiguous.sum()),
    status_counts=D.verification_status.value_counts().to_dict(),
    category_counts=D.category.value_counts().to_dict(),
    category_method_counts=D.category_method.value_counts().to_dict())
print(json.dumps({k: v for k, v in summary.items() if k != 'category_counts'}, indent=2))
summary["verification_warnings"] = warnings
Path(APPR/"curation_summary.json").write_text(json.dumps(summary, indent=2))
assert len(D) + len(X) == rows_in, "row accounting broken"

"""Step 1 - audit the curated workbook (read-only).
Run from repo root:  python rag/ingestion/audit_workbook.py
Writes CSV tables to rag/data/inventory/. Never modifies the workbook."""
import re, hashlib
from pathlib import Path
import pandas as pd

WB = Path("rag/data/raw/HDFC_Banking_LLM_Final_Curated_Dataset_v1.xlsx")
OUT = Path("rag/data/inventory"); OUT.mkdir(parents=True, exist_ok=True)

# sheet -> (question column, answer/text column) for the RAG-relevant sheets
RAG_SHEETS = {
    "KnowledgeBase_Clean":    ("Question", "Answer"),
    "BankFAQ_Clean":          ("question", "answer"),
    "HDFC_FAQ_Clean":         ("question", "answer"),
    "Cybersecurity_Clean":    ("question", "answer"),
    "Privacy_Security_Clean": ("Instruction / Input Prompt", "Authoritative Context Snippet"),
    "RAG_Candidates":         ("document_title", "content"),
    "FAQ_Canonical":          ("question", "answer"),
}
def norm(s):
    """lower-case, strip a trailing '- query 12' variant tag, drop punctuation"""
    s = re.sub(r"\s*[-\u2014]\s*query\s*\d+\s*$", "", str(s).lower())
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s).split())
FIGURES = re.compile(r"(?:\d+(?:\.\d+)?\s?%|rs\.?\s?\d|\u20b9\s?\d|\blakh|\bcrore|per year|per annum|interest rate|\bfee\b|charges?|limit)", re.I)

xl = pd.ExcelFile(WB)
sheets = {s: xl.parse(s) for s in xl.sheet_names}

pd.DataFrame([dict(sheet=s, rows=len(d), columns=len(d.columns),
    exact_duplicate_rows=int(d.astype(str).duplicated().sum()),
    empty_cells=int(d.isna().sum().sum()),
    column_names="; ".join(map(str, d.columns))) for s, d in sheets.items()]
).to_csv(OUT/"01_sheet_inventory.csv", index=False)

rr = []
for s, (qc, ac) in RAG_SHEETS.items():
    d = sheets[s]; q, a = d[qc].astype(str), d[ac].astype(str)
    rr.append(dict(sheet=s, rows=len(d),
        missing_answer=int((d[ac].isna() | (a.str.strip() == "")).sum()),
        dup_question_exact=int(q.duplicated().sum()),
        dup_question_normalised=int(q.map(norm).duplicated().sum()),
        dup_answer=int(a.duplicated().sum()), unique_answers=int(a.nunique()),
        answers_under_20_chars=int((a.str.len() < 20).sum()),
        median_answer_chars=int(a.str.len().median()), max_answer_chars=int(a.str.len().max()),
        mentions_figures_fees_limits=int(a.str.contains(FIGURES).sum()),
        mentions_hdfc=int((q + " " + a).str.contains("hdfc", case=False).sum())))
pd.DataFrame(rr).to_csv(OUT/"02_rag_sheet_quality.csv", index=False)

rc, sq = sheets["RAG_Candidates"], sheets["Suite_RAG_Quarantine"]
bk = []
for col in ["source_file","provenance","access_class","verification_status","rag_eligible"]:
    for k, v in rc[col].value_counts(dropna=False).items():
        bk.append(dict(table="RAG_Candidates", column=col, value=k, rows=int(v)))
for col in ["verification_status","rag_eligible","access_class"]:
    for k, v in sq[col].value_counts(dropna=False).items():
        bk.append(dict(table="Suite_RAG_Quarantine", column=col, value=k, rows=int(v)))
pd.DataFrame(bk).to_csv(OUT/"03_status_breakdown.csv", index=False)

names = {"KnowledgeBase_Clean":"Question","BankFAQ_Clean":"question","HDFC_FAQ_Clean":"question"}
qs = lambda s: set(sheets[s][names[s]].astype(str).map(norm)); ks = list(names); pairs = []
for i in range(3):
    for j in range(i+1, 3):
        pairs.append(dict(a=ks[i], b=ks[j], shared_normalised_questions=len(qs(ks[i]) & qs(ks[j]))))
pairs.append(dict(a="RAG_Candidates.content", b="Suite_RAG_Quarantine.content",
                  shared_normalised_questions=len(set(rc["content"]) & set(sq["content"]))))
pd.DataFrame(pairs).to_csv(OUT/"04_cross_sheet_overlap.csv", index=False)

sheets["KnowledgeBase_Clean"]["Section"].value_counts().rename_axis("section").reset_index(name="rows").to_csv(OUT/"05_kb_sections.csv", index=False)
sheets["BankFAQ_Clean"]["category"].value_counts().rename_axis("category").reset_index(name="rows").to_csv(OUT/"06_bankfaq_categories.csv", index=False)

print("workbook sha256:", hashlib.sha256(WB.read_bytes()).hexdigest())
for f in sorted(OUT.glob("*.csv")):
    print("\n==", f.name); print(pd.read_csv(f).to_string(index=False)[:3000])

import json, re
from pathlib import Path
import pandas as pd
import pytest
from rag.ingestion.data_cleaning import (clean_text, clean_title, norm_key, fact_check_reasons,
                                         is_product_ambiguous, assign_category)
from rag.ingestion.chunking import split_answer, make_chunks, MAX_ANSWER_CHARS
from rag.ingestion.verification import apply_decisions

APPR = Path("rag/data/approved")

# ---- cleaning ----
def test_clean_text_whitespace_and_invisible():
    assert clean_text("  a\u00a0 b\u200b \n\n c ") == "a b\nc"
def test_clean_title_strips_query_tag():
    assert clean_title("How can I check my loan status \u2014 query 7") == "How can I check my loan status"
def test_norm_key_ignores_case_and_punctuation():
    assert norm_key("Can I use the Card?") == norm_key("can i use the card")
def test_fact_check_flags():
    assert "figure_fee_limit_or_rate" in fact_check_reasons("q", "Interest is 4% per year")
    assert "contact_or_url" in fact_check_reasons("q", "Visit www.example.com")
    assert fact_check_reasons("What is KYC?", "Identity check.") == []
def test_product_ambiguous():
    assert is_product_ambiguous("Can I use this card abroad?")
    assert not is_product_ambiguous("What is a savings account?")
def test_category_rules():
    assert assign_category("How do I reset my net banking password?", "") == "Digital Banking"
    assert assign_category("Who can avail a Loan Against Property?", "") == "Loans"

def test_broken_rupee_sign_repaired():
    assert clean_text("up to \u00e2\u201a15 lakh per bank") == "up to \u20b95 lakh per bank"
    assert clean_text("\u00e2\u201a11.5 lakh") == "\u20b91.5 lakh"
    assert clean_text("Rs. 5 lakh and \u20b95,000") == "Rs. 5 lakh and \u20b95,000"      # healthy text untouched

# ---- chunking ----
def test_short_text_one_chunk():
    assert make_chunks("Q?", "Short answer.") == ["Q?\nShort answer."]
def test_long_text_splits_within_limit_and_loses_nothing():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    parts = split_answer(text)
    assert len(parts) > 1 and all(len(p) <= MAX_ANSWER_CHARS for p in parts)
    joined = " ".join(parts)
    assert all(f"Sentence number {i} is here." in joined for i in range(200))
def test_huge_single_sentence_is_cut():
    parts = split_answer("word " * 600)
    assert all(len(p) <= MAX_ANSWER_CHARS for p in parts)
def test_multi_chunk_titles_marked():
    ch = make_chunks("Q?", "A sentence here. " * 100)
    assert ch[0].startswith("Q? (part 1 of ") and ch[-1].startswith(f"Q? (part {len(ch)} of ")

# ---- verification rules ----
def _docs():
    return pd.DataFrame(dict(source_record_id=["r1", "r2", "r3", "r4"], title="t", text="x",
        verification_status="DEMO_LABELLED", source_url="", source_page="", effective_date="", answer_label="demo"))
def _dec(rows): return pd.DataFrame(rows, columns=["source_record_id","matches","source_url","source_page","source_date","checked_by","date_checked","notes"])
def test_Y_with_url_promotes_N_excludes_partial_stays_blank_ignored():
    d, removed, w = apply_decisions(_docs(), _dec([
        ["r1","Y","https://example.org/p","s2","2026-01-01","P","2026-11-05",""],
        ["r2","N","","","","P","",""], ["r3","PARTIAL","","","","P","",""], ["r4","","","","","","",""]]))
    st = d.set_index("source_record_id").verification_status.to_dict()
    assert st == {"r1": "VERIFIED_PUBLIC", "r3": "DEMO_LABELLED", "r4": "DEMO_LABELLED"}
    assert list(removed.source_record_id) == ["r2"] and w == []
def test_Y_without_url_is_rejected():
    d, _, w = apply_decisions(_docs(), _dec([["r1","Y","","","","P","",""]]))
    assert d.set_index("source_record_id").loc["r1", "verification_status"] == "DEMO_LABELLED" and len(w) == 1
def test_unknown_record_warns():
    _, _, w = apply_decisions(_docs(), _dec([["zzz","Y","https://x.org","","","P","",""]]))
    assert any("not found" in x for x in w)

# ---- built dataset invariants (run after build_canonical) ----
pytestmark_built = pytest.mark.skipif(not (APPR/"canonical_documents.csv").exists(), reason="run build_canonical first")
@pytestmark_built
def test_built_dataset_invariants():
    D = pd.read_csv(APPR/"canonical_documents.csv", dtype=str).fillna("")
    C = pd.read_csv(APPR/"canonical_chunks.csv", dtype=str).fillna("")
    X = pd.read_csv("rag/data/cleaned/excluded_records.csv", dtype=str)
    assert len(D) + len(X) == 3238
    assert D.document_id.is_unique and C.chunk_id.is_unique
    assert set(C.document_id) == set(D.document_id)
    assert set(D.verification_status) <= {"DEMO_LABELLED", "VERIFIED_PUBLIC"}
    assert (D.text.str.strip() != "").all() and (D.title.str.strip() != "").all()
    assert not D.title.str.contains(r"query \d+$").any()
    assert not D.text.str.contains("\u00e2\u201a").any() and not C.chunk_text.str.contains("\u00e2\u201a").any()   # no broken rupee signs
    assert C.chunk_text.str.len().astype(int).max() <= 1000
    for m in C.metadata.head(200): assert json.loads(m)["verification_status"] in ("DEMO_LABELLED", "VERIFIED_PUBLIC")
    # VERIFIED_PUBLIC must always have a URL; demo records must carry the label
    assert (D[D.verification_status == "VERIFIED_PUBLIC"].source_url != "").all()
    assert D[D.verification_status == "DEMO_LABELLED"].answer_label.str.contains("Demo knowledge").all()
    # the synthetic HDFC policy pool never enters
    assert not D.source_name.str.contains("7000rows", case=False).any()
@pytestmark_built
def test_original_workbook_unchanged():
    import hashlib
    wb = Path("rag/data/raw/HDFC_Banking_LLM_Final_Curated_Dataset_v1.xlsx")
    expected = Path("rag/data/raw/SHA256.txt").read_text().split()[0]
    assert hashlib.sha256(wb.read_bytes()).hexdigest() == expected

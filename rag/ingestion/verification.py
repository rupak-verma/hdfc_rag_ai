"""Apply human verification decisions to the canonical documents.
A decision file is a CSV with columns:
  source_record_id, matches (Y / N / PARTIAL), source_url, source_page, source_date, checked_by, date_checked, notes
Rules (nothing is promoted on a guess):
  Y       -> VERIFIED_PUBLIC, needs a http(s) source_url and checked_by, else the row is rejected with a warning
  PARTIAL -> stays DEMO_LABELLED (note kept)
  N       -> EXCLUDED (reason failed_verification)
  blank   -> ignored (not reviewed yet)"""
import pandas as pd

REQUIRED_FOR_Y = ("source_url", "checked_by")

def apply_decisions(docs: pd.DataFrame, decisions: pd.DataFrame):
    docs = docs.copy(); warnings, drop_ids = [], []
    if decisions is None or decisions.empty: return docs, docs.iloc[0:0], warnings
    dec = decisions.fillna("").astype(str)
    for r in dec.itertuples(index=False):
        m = r.matches.strip().upper()
        if m == "": continue
        idx = docs.index[docs.source_record_id == r.source_record_id]
        if len(idx) != 1: warnings.append(f"{r.source_record_id}: not found in documents"); continue
        i = idx[0]
        if m == "Y":
            missing = [c for c in REQUIRED_FOR_Y if not str(getattr(r, c)).strip()]
            if missing or not str(r.source_url).strip().lower().startswith("http"):
                warnings.append(f"{r.source_record_id}: Y rejected - needs a http(s) source_url and checked_by"); continue
            docs.loc[i, ["verification_status", "source_url", "source_page", "effective_date", "answer_label"]] = [
                "VERIFIED_PUBLIC", r.source_url.strip(), r.source_page.strip(), r.source_date.strip(),
                f"Verified against public source: {r.source_url.strip()}"]
        elif m == "N": drop_ids.append(i)
        elif m != "PARTIAL": warnings.append(f"{r.source_record_id}: unknown value '{r.matches}'")
    removed = docs.loc[drop_ids].assign(exclusion_reason="failed_verification")
    return docs.drop(index=drop_ids), removed, warnings

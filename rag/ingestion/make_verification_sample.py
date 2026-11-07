"""Create the human fact-check sheet (150 rows, fixed seed).  Run from repo root:
   python -m rag.ingestion.make_verification_sample
80 records with figures/fees/limits + 40 with contacts/URLs + 30 unflagged controls.
Half go to reviewer A, half to reviewer B (alternating after a seeded shuffle)."""
import pandas as pd
D = pd.read_csv("rag/data/approved/canonical_documents.csv", dtype=str).fillna("")
D["needs_fact_check"] = D.needs_fact_check == "True"
fig = D[D.fact_check_reasons.str.contains("figure_fee_limit_or_rate")]
con = D[D.fact_check_reasons.str.contains("contact_or_url") & ~D.index.isin(fig.index)]
ctl = D[~D.needs_fact_check]
pick = lambda df, n, why: df.sample(min(n, len(df)), random_state=42).assign(sample_group=why)
S = pd.concat([pick(fig, 80, "figure_fee_limit_rate"), pick(con, 40, "contact_or_url"), pick(ctl, 30, "control_unflagged")])
S = S.sample(frac=1, random_state=7).reset_index(drop=True)
S["reviewer"] = ["A" if i % 2 == 0 else "B" for i in range(len(S))]
cols = ["document_id", "source_record_id", "reviewer", "sample_group", "category", "source_name", "title", "text"]
out = S[cols].sort_values(["reviewer", "sample_group"]).copy()
for c in ["matches", "source_url", "source_page", "source_date", "checked_by", "date_checked", "notes"]: out[c] = ""
out.to_csv("rag/data/approved/verification_sample.csv", index=False)
print(len(out), out.reviewer.value_counts().to_dict(), out.sample_group.value_counts().to_dict())

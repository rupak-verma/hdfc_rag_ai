"""Measure retrieval quality and abstention.  Run from repo root:
    python -m rag.evaluation.retrieval_evaluation --calibrate     # sets abstention thresholds on the DEV half, reports on the TEST half
    python -m rag.evaluation.retrieval_evaluation                 # report only (uses saved thresholds)
Metrics (plain language)
  Recall@k : share of answerable questions whose correct document is among the first k results
  MRR      : average of 1/rank of the correct document (1.0 = always first; 0.5 = typically second)
  abstain  : share of 'absent' questions where the system correctly returns nothing; false abstain = answerable questions wrongly refused
Honesty rules: thresholds are chosen on dev only; headline numbers come from test only; Wilson 95% intervals are shown because the set is small."""
import argparse, json, math, os, datetime as dt
from pathlib import Path
import numpy as np
from rag.retrieval.retriever import get_engine, CALIBRATION, MAX_PER_DOC
from rag.retrieval.guards import unknown_phrases

QS = Path("rag/evaluation/test_questions.json"); OUT = Path("rag/evaluation/results"); OUT.mkdir(parents=True, exist_ok=True)
doc_of = lambda cid: "DOC-" + cid.split("-")[1]

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - w) / d, (c + w) / d)

def first_k_chunks(chunk_ids, k):
    """apply the same 'max 2 chunks per document' rule as the live retriever, then take k"""
    out, per = [], {}
    for c in chunk_ids:
        d = doc_of(c)
        if per.get(d, 0) >= MAX_PER_DOC: continue
        per[d] = per.get(d, 0) + 1; out.append(c)
        if len(out) == k: break
    return out

def doc_rank(chunk_ids, relevant, depth=30):
    for r, c in enumerate(first_k_chunks(chunk_ids, depth), 1):
        if doc_of(c) in relevant: return r
    return None

def compute(engine, items):
    rows = []
    for it in items:
        fused, kw, vec = engine.rank(it["question"], n=30)
        head = [c for c, _, _ in fused[:3]]
        cov = max((engine.bm25.coverage_by_id(it["question"], c) for c in head), default=0.0)
        rel = set(it["relevant_document_ids"])
        rows.append(dict(**{k: it[k] for k in ("id", "type", "style", "split", "question")}, relevant=sorted(rel),
            cos_top=vec[0][1] if vec else 0.0, cov_top=cov, unknown_phrase=bool(unknown_phrases(it["question"], engine.corpus_lower)),
            rank_keyword=doc_rank([c for c, _ in kw], rel), rank_vector=doc_rank([c for c, _ in vec], rel),
            rank_hybrid=doc_rank([c for c, _, _ in fused], rel),
            top3_hybrid=[doc_of(c) for c in head]))
    return rows

def retrieval_metrics(rows, key):
    ans = [r for r in rows if r["type"] == "answerable"]; n = len(ans); out = {"n": n}
    for k in (1, 3, 5):
        hit = sum(1 for r in ans if r[key] and r[key] <= k); lo, hi = wilson(hit, n)
        out[f"recall@{k}"] = round(hit / n, 3) if n else None; out[f"recall@{k}_ci95"] = [round(lo, 3), round(hi, 3)]
    out["mrr"] = round(sum(1 / r[key] for r in ans if r[key]) / n, 3) if n else None
    return out

def passes(r, a, b): return (not r["unknown_phrase"]) and r["cos_top"] >= a and r["cov_top"] >= b

def calibrate(dev):
    ans = [r for r in dev if r["type"] == "answerable"]; ab = [r for r in dev if r["type"] == "absent"]
    best = None
    for a in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        for b in np.round(np.arange(0.0, 1.0001, 0.05), 2):
            tpr = sum(passes(r, a, b) for r in ans) / len(ans); tnr = sum(not passes(r, a, b) for r in ab) / len(ab)
            key = (tpr + tnr - 1, tpr, -a - b)                         # Youden's J, then fewer false abstains, then looser thresholds
            if best is None or key > best[0]: best = (key, float(a), float(b), float(tpr), float(tnr))
    return dict(min_cosine=best[1], min_coverage=best[2], dev_answerable_pass_rate=round(best[3], 3), dev_absent_abstain_rate=round(best[4], 3))

def gate_metrics(rows, a, b):
    ans = [r for r in rows if r["type"] == "answerable"]; ab = [r for r in rows if r["type"] == "absent"]
    out = dict(answerable_pass_rate=round(sum(passes(r, a, b) for r in ans) / len(ans), 3),
               false_abstain_rate=round(sum(not passes(r, a, b) for r in ans) / len(ans), 3),
               absent_abstain_rate=round(sum(not passes(r, a, b) for r in ab) / len(ab), 3),
               answerable_correct_and_passed_at5=round(sum(bool(passes(r, a, b) and r["rank_hybrid"] and r["rank_hybrid"] <= 5) for r in ans) / len(ans), 3))
    rag_ab = [r for r in ab if r["style"] != "personal_data"]     # personal-data questions are routed to the database before RAG
    out["absent_abstain_rate_excl_personal_data"] = round(sum(not passes(r, a, b) for r in rag_ab) / len(rag_ab), 3)
    lo, hi = wilson(sum(not passes(r, a, b) for r in ab), len(ab)); out["absent_abstain_ci95"] = [round(lo, 3), round(hi, 3)]
    kinds = {}
    for r in ab: kinds.setdefault(r["style"], []).append(not passes(r, a, b))
    out["absent_abstain_by_kind"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(kinds.items())}
    styles = {}
    for r in ans: styles.setdefault(r["style"], []).append(bool(r["rank_hybrid"] and r["rank_hybrid"] <= 5))
    out["hybrid_recall@5_by_style"] = {k: f"{sum(v)}/{len(v)}" for k, v in sorted(styles.items())}
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--calibrate", action="store_true"); a = ap.parse_args()
    eng = get_engine(); mid = eng.model.model_id; items = json.loads(QS.read_text())
    rows = compute(eng, items); dev = [r for r in rows if r["split"] == "dev"]; test = [r for r in rows if r["split"] == "test"]
    if a.calibrate:
        cal = calibrate(dev); allc = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
        allc[mid] = {**cal, "calibrated_on": "dev split", "date": dt.date.today().isoformat()}; CALIBRATION.write_text(json.dumps(allc, indent=2))
        thr = (cal["min_cosine"], cal["min_coverage"]); print("calibrated on dev:", cal)
    else:
        thr = (eng.thresholds["min_cosine"], eng.thresholds["min_coverage"])
        if not eng.calibrated: print("WARNING: no saved calibration for this model; using defaults")
    res = dict(model_id=mid, standin=eng.model.is_standin, dimension=eng.model.dimension, keyword_backend=eng.keyword_backend, store=eng.store.kind,
               corpus_chunks=len(eng.bm25.ids), thresholds=dict(min_cosine=thr[0], min_coverage=thr[1]), date=dt.date.today().isoformat(),
               n_dev=len(dev), n_test=len(test),
               retrieval_test={m: retrieval_metrics(test, f"rank_{m}") for m in ("keyword", "vector", "hybrid")},
               retrieval_all={m: retrieval_metrics(rows, f"rank_{m}") for m in ("keyword", "vector", "hybrid")},
               gate_test=gate_metrics(test, *thr), gate_dev=gate_metrics(dev, *thr))
    miss = [dict(id=r["id"], question=r["question"], correct=r["relevant"][:2], got_top3=r["top3_hybrid"], rank=r["rank_hybrid"])
            for r in rows if r["type"] == "answerable" and not (r["rank_hybrid"] and r["rank_hybrid"] <= 5)]
    false_pass = [dict(id=r["id"], question=r["question"], kind=r["style"], cos=round(r["cos_top"], 3), cov=round(r["cov_top"], 3))
                  for r in rows if r["type"] == "absent" and passes(r, *thr)]
    res["failures_answerable_missed_at5"] = miss; res["failures_absent_answered"] = false_pass
    tag = mid.replace("/", "-").replace(":", "-"); (OUT / f"retrieval_eval_{tag}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("model_id", "standin", "thresholds", "retrieval_test", "gate_test")}, indent=1))
    print(f"missed@5 (answerable): {len(miss)} | absent questions wrongly answered: {len(false_pass)}")

if __name__ == "__main__": main()

"""Turn a retrieval_eval_<model>.json into a readable Markdown report.
    python -m rag.evaluation.write_report rag/evaluation/results/retrieval_eval_<tag>.json docs/RAG_EVALUATION_REPORT.md"""
import json, sys
from pathlib import Path

def pct(x): return "n/a" if x is None else f"{x*100:.1f}%"
def ci(c): return f"{c[0]*100:.0f}-{c[1]*100:.0f}%"

def main(src, dst):
    r = json.loads(Path(src).read_text()); L = []
    if r["standin"]:
        L += ["> **STAND-IN MODEL - NOT PROJECT RESULTS.** These numbers come from averaged word vectors (en_core_web_md), used only to test the pipeline",
              "> where real model downloads are blocked. Re-run with the real embedding model before quoting anything below.", ""]
    L += [f"# RAG retrieval evaluation - {r['model_id']}", "",
          f"Date {r['date']} | vector size {r['dimension']} | store: {r['store']} | keyword search: {r['keyword_backend']} | chunks searched: {r['corpus_chunks']:,}",
          f"Questions: {r['n_dev']} dev (used only to set the abstention thresholds) + {r['n_test']} test (used for every headline number). "
          "The set is a Claude-drafted DRAFT awaiting human review.", "",
          "## 1. Finding the right document (test questions that have an answer)", "",
          "| Search method | Recall@1 | Recall@3 | Recall@5 | Recall@5 95% interval | MRR |", "|---|---|---|---|---|---|"]
    for m, name in (("keyword", "Keyword (BM25)"), ("vector", "Vector (meaning)"), ("hybrid", "Hybrid (RRF)")):
        v = r["retrieval_test"][m]
        L.append(f"| {name} | {pct(v['recall@1'])} | {pct(v['recall@3'])} | {pct(v['recall@5'])} | {ci(v['recall@5_ci95'])} | {v['mrr']} |")
    n = r["retrieval_test"]["hybrid"]["n"]
    L += ["", f"n = {n} answerable test questions, so one question moves a percentage by about {100/n:.0f} points: read the intervals, not just the point values.", "",
          "## 2. Knowing when not to answer (abstention)", "",
          f"Thresholds (set on dev): similarity >= {r['thresholds']['min_cosine']}, word coverage >= {r['thresholds']['min_coverage']}; plus the unknown-product guard.", "",
          "| Measure | Test result |", "|---|---|"]
    g = r["gate_test"]
    L += [f"| Answerable questions wrongly refused (false abstain) | {pct(g['false_abstain_rate'])} |",
          f"| Questions with no answer correctly refused | {pct(g['absent_abstain_rate'])} (95% interval {ci(g['absent_abstain_ci95'])}) |",
          f"| ...same, excluding personal-data questions (routed to the database before RAG) | {pct(g['absent_abstain_rate_excl_personal_data'])} |",
          f"| Correct document in top 5 AND not refused | {pct(g['answerable_correct_and_passed_at5'])} |", "",
          "Refused correctly, by kind: " + ", ".join(f"{k} {v}" for k, v in g["absent_abstain_by_kind"].items()) + ".",
          "Hybrid Recall@5 by question style: " + ", ".join(f"{k} {v}" for k, v in g["hybrid_recall@5_by_style"].items()) + " (kw = shares key words, para = paraphrase).", "",
          "## 3. Where it fails", ""]
    L += ["Absent questions that were wrongly answered:"] + [f"- {x['id']} ({x['kind']}): {x['question']}" for x in r["failures_absent_answered"]] if r["failures_absent_answered"] else ["No absent question was wrongly answered on this run."]
    L += ["", f"Answerable questions whose document was not in the top 5 (all splits): {len(r['failures_answerable_missed_at5'])}. First five:"]
    L += [f"- {x['id']}: {x['question']} (wanted {x['correct'][0]}, got {', '.join(x['got_top3'])})" for x in r["failures_answerable_missed_at5"][:5]]
    L += ["", "## 4. How to read this honestly",
          "- Questions were written by the same side that prepared the data, from the same documents; real customers will phrase things differently.",
          "- Documents are demo-labelled, not verified HDFC policy, so a correct retrieval is not a correct fact.",
          "- Thresholds depend on the model: change the model and you must re-run `--calibrate`."]
    Path(dst).write_text("\n".join(L) + "\n"); print("wrote", dst)

if __name__ == "__main__": main(sys.argv[1], sys.argv[2])

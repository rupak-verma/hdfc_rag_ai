"""Keyword search with BM25 (Okapi BM25 from the rank_bm25 library) - exact words matter here."""
import re
from rank_bm25 import BM25Okapi

STOP = set("a an the is are was were be been am do does did can could i me my we our you your it its of to in on for with at by from "
           "and or if as that this these those what which who whom how when where why will would should may might shall about into "
           "than then there their they them he she his her not no any".split())

def tokenize(text: str) -> list:
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOP: continue
        if len(t) > 4:                                     # tiny stemmer: card/cards, transfer/transfers/transferred
            for suf in ("ing", "ed", "es", "s"):
                if t.endswith(suf) and len(t) - len(suf) >= 3: t = t[: -len(suf)]; break
        out.append(t)
    return out

class BM25Retriever:
    def __init__(self, ids, texts):
        self.ids = list(ids); self._tok = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tok)
    def search(self, query: str, k: int):
        q = tokenize(query)
        if not q: return []
        s = self._bm25.get_scores(q)
        order = sorted(range(len(s)), key=lambda i: (-s[i], self.ids[i]))[:k]     # deterministic ties
        return [(self.ids[i], float(s[i])) for i in order if s[i] > 0]
    def coverage(self, query: str, chunk_id_index: int) -> float:
        q = set(tokenize(query)); return len(q & set(self._tok[chunk_id_index])) / len(q) if q else 0.0
    def coverage_by_id(self, query: str, chunk_id: str) -> float:
        if not hasattr(self, "_pos"): self._pos = {c: i for i, c in enumerate(self.ids)}
        return self.coverage(query, self._pos[chunk_id])

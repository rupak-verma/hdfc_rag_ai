"""Hybrid ranking with Reciprocal Rank Fusion (RRF).
RRF score = sum over searches of weight / (60 + rank). It uses ranks, not raw scores, so BM25 and cosine (different scales)
combine fairly. Deterministic: ties are broken by chunk_id."""
RRF_K = 60

def rrf_fuse(rankings: dict, weights: dict, top_n: int):
    """rankings: {'keyword': [(id, score), ...], 'vector': [...]} ordered best-first -> [(id, fused_score, {name: rank})]"""
    fused, ranks = {}, {}
    for name, lst in rankings.items():
        for r, (cid, _s) in enumerate(lst, start=1):
            fused[cid] = fused.get(cid, 0.0) + weights.get(name, 1.0) / (RRF_K + r)
            ranks.setdefault(cid, {})[name] = r
    order = sorted(fused, key=lambda c: (-fused[c], c))[:top_n]
    return [(c, fused[c], ranks[c]) for c in order]

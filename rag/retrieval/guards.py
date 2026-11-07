"""Unknown-product guard.
Confidence scores cannot catch a made-up product ("HDFC Platinum Elite Supreme card"): its words overlap real documents,
so retrieval happily returns the nearest real card. This guard looks for a capitalised multi-word phrase in the question
(e.g. 'Platinum Elite Supreme') and abstains if that exact phrase appears nowhere in the knowledge base.
Limits: only catches capitalised phrases of 2+ words; a lower-case question ('platinum elite supreme card') slips past."""
import re

_PHRASE = re.compile(r"\b[A-Z][A-Za-z0-9+&]*(?:\s+[A-Z][A-Za-z0-9+&]*)+\b")
_IGNORE = {"hdfc", "bank", "i", "what", "how", "is", "are", "can", "does", "do", "will", "when", "where", "which", "who", "why",
           "the", "a", "an", "my", "your", "if", "should", "could", "would", "please", "tell", "give", "explain", "write"}

def named_phrases(question: str) -> list:
    out = []
    for m in _PHRASE.finditer(question):
        words = [w for w in m.group().split()]
        while words and words[0].lower() in _IGNORE: words = words[1:]      # 'What Is', 'HDFC ...' -> drop leading filler
        while words and words[-1].lower() in _IGNORE: words = words[:-1]
        if len(words) >= 2: out.append(" ".join(words))
    return out

def unknown_phrases(question: str, corpus_lower: str) -> list:
    """Capitalised phrases in the question that appear nowhere in the corpus."""
    return [p for p in named_phrases(question) if p.lower() not in corpus_lower]

"""Split a document into chunks that fit the embedding model (about 256 tokens ~ 1000 chars)."""
import re
MAX_ANSWER_CHARS = 700   # leaves room for the question line
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")

def split_answer(text: str, max_chars: int = MAX_ANSWER_CHARS) -> list:
    """Sentence-aware split; carries the last sentence into the next chunk as overlap."""
    if len(text) <= max_chars: return [text]
    sents = [s for s in _SENT.split(text) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        while len(s) > max_chars:                      # one huge sentence: hard cut on a space
            cut = s.rfind(" ", 0, max_chars); cut = cut if cut > 0 else max_chars
            if cur: chunks.append(" ".join(cur)); cur = []
            chunks.append(s[:cut].strip()); s = s[cut:].strip()
        if cur and len(" ".join(cur + [s])) > max_chars:
            chunks.append(" ".join(cur)); cur = [cur[-1], s] if len(cur[-1]) + len(s) < max_chars else [s]
        else:
            cur.append(s)
    if cur: chunks.append(" ".join(cur))
    return chunks

def make_chunks(title: str, text: str) -> list:
    """Every chunk repeats the question/title so it can be understood on its own."""
    parts = split_answer(text); n = len(parts)
    if n == 1: return [f"{title}\n{parts[0]}"]
    return [f"{title} (part {i} of {n})\n{part}" for i, part in enumerate(parts, 1)]

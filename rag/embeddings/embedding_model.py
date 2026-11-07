"""One place that knows how to turn text into vectors.

Backends
  sentence-transformers  (default, REAL)  - EMBEDDING_MODEL, e.g. sentence-transformers/all-MiniLM-L6-v2
  spacy-standin          (TEST ONLY)      - averaged word vectors from spaCy en_core_web_md (300 dims).
                                            Exists so the pipeline can be tested where model downloads are blocked.
                                            Its numbers must never be reported as the project's embedding results.
Vectors are L2-normalised, so cosine similarity = dot product and pgvector's cosine distance (<=>) matches.
The vector dimension is MEASURED from the model, never assumed."""
import os
import numpy as np

# Prefix rules differ per model family; wrong prefixes silently hurt retrieval quality.
MODEL_PREFIXES = {
    "BAAI/bge-small-en-v1.5": dict(query="Represent this sentence for searching relevant passages: ", passage=""),
    "BAAI/bge-base-en-v1.5":  dict(query="Represent this sentence for searching relevant passages: ", passage=""),
    "intfloat/e5-small-v2":   dict(query="query: ", passage="passage: "),
    "intfloat/e5-base-v2":    dict(query="query: ", passage="passage: "),
}
STANDIN_ID = "standin:spacy-en_core_web_md"

class EmbeddingModel:
    def __init__(self, backend: str | None = None, name: str | None = None):
        self.backend = backend or os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
        self.name = name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.is_standin = self.backend == "spacy-standin"
        pre = {} if self.is_standin else MODEL_PREFIXES.get(self.name, {})
        self.query_prefix, self.passage_prefix = pre.get("query", ""), pre.get("passage", "")
        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer   # imported lazily: heavy
            self._m = SentenceTransformer(self.name)
            self.model_id = self.name
        elif self.is_standin:
            import spacy
            self._m = spacy.load("en_core_web_md", disable=["parser", "ner", "tagger", "lemmatizer", "attribute_ruler"])
            self.model_id = STANDIN_ID
        else:
            raise ValueError(f"unknown EMBEDDING_BACKEND '{self.backend}'")
        self.dimension = int(self._encode(["dimension probe"]).shape[1])   # measured, not assumed
        declared = os.getenv("EMBEDDING_DIMENSION")
        if declared and int(declared) != self.dimension:
            raise ValueError(f"EMBEDDING_DIMENSION={declared} but {self.model_id} produces {self.dimension}. "
                             "Fix .env (and re-create the pgvector column) - do not guess.")

    def _encode(self, texts, batch_size=64) -> np.ndarray:
        if self.backend == "sentence-transformers":
            v = self._m.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)
        out = []
        for doc in self._m.pipe(list(texts), batch_size=batch_size):
            vecs = [t.vector for t in doc if t.has_vector and not t.is_stop and not t.is_punct]
            v = np.mean(vecs, axis=0) if vecs else np.zeros(self._m.vocab.vectors_length, dtype=np.float32)
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return np.asarray(out, dtype=np.float32)

    def encode_passages(self, texts, batch_size=64) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts], batch_size)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

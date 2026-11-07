"""Meaning search: embed the question, find the nearest chunks by cosine similarity."""
class VectorRetriever:
    def __init__(self, model, store): self.model, self.store = model, store
    def search(self, query: str, k: int):
        return self.store.vector_search(self.model.encode_query(query), k)

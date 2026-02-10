from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class Embedder:
    def __init__(self, n_features: int = 384):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            norm=None,
            alternate_sign=False,
            ngram_range=(1, 2),
        )

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        matrix = self.vectorizer.transform(texts).toarray().astype(np.float32)
        return self._l2_normalize(matrix)

    @lru_cache(maxsize=1024)
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_batch([query])[0]

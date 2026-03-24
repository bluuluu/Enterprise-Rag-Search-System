from __future__ import annotations

import numpy as np

try:
    import vector_search_cpp

    CPP_ENGINE_AVAILABLE = True
except ImportError:
    vector_search_cpp = None
    CPP_ENGINE_AVAILABLE = False


def _normalize_query(query: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(query)
    if norm == 0:
        return query
    return query / norm


def topk_cosine(
    query: np.ndarray,
    matrix: np.ndarray,
    top_k: int,
    num_threads: int = 0,
    assume_normalized: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0 or top_k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    query = np.ascontiguousarray(query, dtype=np.float32)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("query must be 1D")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if query.shape[0] != matrix.shape[1]:
        raise ValueError("query dimension must match matrix dimension")

    k = min(top_k, matrix.shape[0])

    if CPP_ENGINE_AVAILABLE:
        indices, scores = vector_search_cpp.topk_cosine_similarity(
            query,
            matrix,
            k,
            num_threads,
            assume_normalized,
        )
        return np.asarray(indices, dtype=np.int64), np.asarray(scores, dtype=np.float32)

    query_for_scoring = query if assume_normalized else _normalize_query(query)
    scores = matrix @ query_for_scoring
    if not assume_normalized:
        denom = np.linalg.norm(matrix, axis=1)
        denom[denom == 0.0] = 1.0
        scores = scores / denom

    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return top_idx.astype(np.int64), scores[top_idx].astype(np.float32)


def topk_euclidean(
    query: np.ndarray,
    matrix: np.ndarray,
    top_k: int,
    num_threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if matrix.size == 0 or top_k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    query = np.ascontiguousarray(query, dtype=np.float32)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("query must be 1D")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    if query.shape[0] != matrix.shape[1]:
        raise ValueError("query dimension must match matrix dimension")

    k = min(top_k, matrix.shape[0])

    if CPP_ENGINE_AVAILABLE:
        indices, distances = vector_search_cpp.topk_euclidean_distance(query, matrix, k, num_threads)
        return np.asarray(indices, dtype=np.int64), np.asarray(distances, dtype=np.float32)

    distances = np.sum((matrix - query) ** 2, axis=1)
    top_idx = np.argpartition(distances, k - 1)[:k]
    top_idx = top_idx[np.argsort(distances[top_idx])]
    return top_idx.astype(np.int64), distances[top_idx].astype(np.float32)

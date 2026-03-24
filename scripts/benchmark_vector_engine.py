from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vector_engine import CPP_ENGINE_AVAILABLE, topk_cosine


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def run_benchmark(
    num_vectors: int,
    dim: int,
    top_k: int,
    num_queries: int,
    num_threads: int,
    compare_numpy: bool,
) -> None:
    effective_top_k = min(top_k, num_vectors)
    if effective_top_k <= 0:
        raise ValueError("top_k and num_vectors must both be > 0")

    rng = np.random.default_rng(42)

    print(
        f"Preparing dataset: vectors={num_vectors:,} dim={dim} top_k={top_k} queries={num_queries} threads={num_threads}"
    )

    matrix = l2_normalize(rng.standard_normal((num_vectors, dim), dtype=np.float32))
    queries = l2_normalize(rng.standard_normal((num_queries, dim), dtype=np.float32))

    warmup_count = min(3, num_queries)
    for i in range(warmup_count):
        topk_cosine(
            queries[i],
            matrix,
            top_k=effective_top_k,
            num_threads=num_threads,
            assume_normalized=True,
        )

    latencies_ms: list[float] = []
    for i in range(num_queries):
        t0 = time.perf_counter()
        topk_cosine(
            queries[i],
            matrix,
            top_k=effective_top_k,
            num_threads=num_threads,
            assume_normalized=True,
        )
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50 = statistics.median(latencies_ms)
    p95 = np.percentile(latencies_ms, 95)
    p99 = np.percentile(latencies_ms, 99)

    backend = "cpp_simd_mt" if CPP_ENGINE_AVAILABLE else "numpy_fallback"
    print(f"Backend: {backend}")
    print(f"Latency p50: {p50:.3f} ms")
    print(f"Latency p95: {p95:.3f} ms")
    print(f"Latency p99: {p99:.3f} ms")
    print(f"Latency mean: {statistics.mean(latencies_ms):.3f} ms")

    if compare_numpy:
        numpy_latencies_ms: list[float] = []
        for i in range(num_queries):
            t0 = time.perf_counter()
            scores = matrix @ queries[i]
            top_idx = np.argpartition(-scores, effective_top_k - 1)[:effective_top_k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            _ = top_idx
            numpy_latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        numpy_mean = statistics.mean(numpy_latencies_ms)
        print(f"NumPy mean: {numpy_mean:.3f} ms")
        if CPP_ENGINE_AVAILABLE and numpy_mean > 0:
            speedup = numpy_mean / statistics.mean(latencies_ms)
            print(f"C++ speedup vs NumPy: {speedup:.2f}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-vectors", type=int, default=200_000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--compare-numpy", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        num_vectors=args.num_vectors,
        dim=args.dim,
        top_k=args.top_k,
        num_queries=args.num_queries,
        num_threads=args.num_threads,
        compare_numpy=args.compare_numpy,
    )

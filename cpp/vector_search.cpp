#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace py = pybind11;

namespace {

inline float dot_product_simd(const float* a, const float* b, std::size_t dim) {
    std::size_t i = 0;
    float sum = 0.0f;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }

    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (float val : tmp) {
        sum += val;
    }
#elif defined(__SSE2__)
    __m128 acc = _mm_setzero_ps();
    for (; i + 4 <= dim; i += 4) {
        const __m128 va = _mm_loadu_ps(a + i);
        const __m128 vb = _mm_loadu_ps(b + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }

    alignas(16) float tmp[4];
    _mm_store_ps(tmp, acc);
    for (float val : tmp) {
        sum += val;
    }
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);
    }

    alignas(16) float tmp[4];
    vst1q_f32(tmp, acc);
    for (float val : tmp) {
        sum += val;
    }
#endif

    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

inline float l2_norm_simd(const float* v, std::size_t dim) {
    return std::sqrt(std::max(dot_product_simd(v, v, dim), 0.0f));
}

inline float l2_distance_squared_simd(const float* a, const float* b, std::size_t dim) {
    std::size_t i = 0;
    float sum = 0.0f;

#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
    }

    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    for (float val : tmp) {
        sum += val;
    }
#elif defined(__SSE2__)
    __m128 acc = _mm_setzero_ps();
    for (; i + 4 <= dim; i += 4) {
        const __m128 va = _mm_loadu_ps(a + i);
        const __m128 vb = _mm_loadu_ps(b + i);
        const __m128 diff = _mm_sub_ps(va, vb);
        acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
    }

    alignas(16) float tmp[4];
    _mm_store_ps(tmp, acc);
    for (float val : tmp) {
        sum += val;
    }
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= dim; i += 4) {
        const float32x4_t va = vld1q_f32(a + i);
        const float32x4_t vb = vld1q_f32(b + i);
        const float32x4_t diff = vsubq_f32(va, vb);
        acc = vmlaq_f32(acc, diff, diff);
    }

    alignas(16) float tmp[4];
    vst1q_f32(tmp, acc);
    for (float val : tmp) {
        sum += val;
    }
#endif

    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

int choose_thread_count(std::size_t workload, int num_threads) {
    if (workload == 0) {
        return 1;
    }

    int threads = num_threads;
    if (threads <= 0) {
        const unsigned int hw = std::thread::hardware_concurrency();
        threads = hw == 0 ? 4 : static_cast<int>(hw);
    }

    threads = std::max(1, threads);
    threads = std::min<int>(threads, static_cast<int>(workload));
    return threads;
}

py::tuple rank_cosine_impl(
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    py::array_t<float, py::array::c_style | py::array::forcecast> matrix,
    std::int64_t top_k,
    int num_threads,
    bool assume_normalized
) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float32 array");
    }
    if (matrix.ndim() != 2) {
        throw std::invalid_argument("matrix must be a 2D float32 array");
    }

    const auto num_vectors = static_cast<std::size_t>(matrix.shape(0));
    const auto dim = static_cast<std::size_t>(matrix.shape(1));
    if (static_cast<std::size_t>(query.shape(0)) != dim) {
        throw std::invalid_argument("query dimension must match matrix dimension");
    }

    if (num_vectors == 0 || top_k <= 0) {
        return py::make_tuple(py::array_t<std::int64_t>(0), py::array_t<float>(0));
    }

    const std::size_t k = static_cast<std::size_t>(std::min<std::int64_t>(top_k, static_cast<std::int64_t>(num_vectors)));
    const auto q_info = query.request();
    const auto m_info = matrix.request();

    const float* query_ptr = static_cast<const float*>(q_info.ptr);
    const float* matrix_ptr = static_cast<const float*>(m_info.ptr);

    std::vector<float> query_normalized(dim);
    std::copy(query_ptr, query_ptr + dim, query_normalized.begin());

    float query_norm = 1.0f;
    if (!assume_normalized) {
        query_norm = l2_norm_simd(query_ptr, dim);
        if (query_norm <= 0.0f) {
            throw std::invalid_argument("query vector norm must be > 0");
        }
        const float inv = 1.0f / query_norm;
        for (std::size_t i = 0; i < dim; ++i) {
            query_normalized[i] *= inv;
        }
    }

    std::vector<float> scores(num_vectors, -std::numeric_limits<float>::infinity());
    const int threads = choose_thread_count(num_vectors, num_threads);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));

    const std::size_t chunk = (num_vectors + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads);

    auto worker = [&](std::size_t start, std::size_t end) {
        for (std::size_t row = start; row < end; ++row) {
            const float* vec = matrix_ptr + (row * dim);
            float score = dot_product_simd(query_normalized.data(), vec, dim);

            if (!assume_normalized) {
                const float vec_norm = l2_norm_simd(vec, dim);
                if (vec_norm > 0.0f) {
                    score /= vec_norm;
                } else {
                    score = -std::numeric_limits<float>::infinity();
                }
            }

            scores[row] = score;
        }
    };

    for (int t = 0; t < threads; ++t) {
        const std::size_t start = static_cast<std::size_t>(t) * chunk;
        if (start >= num_vectors) {
            break;
        }
        const std::size_t end = std::min(num_vectors, start + chunk);
        workers.emplace_back(worker, start, end);
    }

    for (auto& th : workers) {
        th.join();
    }

    std::vector<std::int64_t> ranking(num_vectors);
    std::iota(ranking.begin(), ranking.end(), static_cast<std::int64_t>(0));

    auto descending_cmp = [&](std::int64_t a, std::int64_t b) {
        return scores[static_cast<std::size_t>(a)] > scores[static_cast<std::size_t>(b)];
    };

    if (k < num_vectors) {
        std::nth_element(ranking.begin(), ranking.begin() + static_cast<std::ptrdiff_t>(k), ranking.end(), descending_cmp);
        ranking.resize(k);
    }
    std::sort(ranking.begin(), ranking.end(), descending_cmp);

    py::array_t<std::int64_t> top_indices(ranking.size());
    py::array_t<float> top_scores(ranking.size());

    auto idx_view = top_indices.mutable_unchecked<1>();
    auto score_view = top_scores.mutable_unchecked<1>();

    for (std::size_t i = 0; i < ranking.size(); ++i) {
        const std::int64_t idx = ranking[i];
        idx_view(static_cast<py::ssize_t>(i)) = idx;
        score_view(static_cast<py::ssize_t>(i)) = scores[static_cast<std::size_t>(idx)];
    }

    return py::make_tuple(top_indices, top_scores);
}

py::tuple rank_euclidean_impl(
    py::array_t<float, py::array::c_style | py::array::forcecast> query,
    py::array_t<float, py::array::c_style | py::array::forcecast> matrix,
    std::int64_t top_k,
    int num_threads
) {
    if (query.ndim() != 1) {
        throw std::invalid_argument("query must be a 1D float32 array");
    }
    if (matrix.ndim() != 2) {
        throw std::invalid_argument("matrix must be a 2D float32 array");
    }

    const auto num_vectors = static_cast<std::size_t>(matrix.shape(0));
    const auto dim = static_cast<std::size_t>(matrix.shape(1));
    if (static_cast<std::size_t>(query.shape(0)) != dim) {
        throw std::invalid_argument("query dimension must match matrix dimension");
    }

    if (num_vectors == 0 || top_k <= 0) {
        return py::make_tuple(py::array_t<std::int64_t>(0), py::array_t<float>(0));
    }

    const std::size_t k = static_cast<std::size_t>(std::min<std::int64_t>(top_k, static_cast<std::int64_t>(num_vectors)));
    const auto q_info = query.request();
    const auto m_info = matrix.request();

    const float* query_ptr = static_cast<const float*>(q_info.ptr);
    const float* matrix_ptr = static_cast<const float*>(m_info.ptr);

    std::vector<float> distances(num_vectors, std::numeric_limits<float>::infinity());
    const int threads = choose_thread_count(num_vectors, num_threads);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));

    const std::size_t chunk = (num_vectors + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads);

    auto worker = [&](std::size_t start, std::size_t end) {
        for (std::size_t row = start; row < end; ++row) {
            const float* vec = matrix_ptr + (row * dim);
            distances[row] = l2_distance_squared_simd(query_ptr, vec, dim);
        }
    };

    for (int t = 0; t < threads; ++t) {
        const std::size_t start = static_cast<std::size_t>(t) * chunk;
        if (start >= num_vectors) {
            break;
        }
        const std::size_t end = std::min(num_vectors, start + chunk);
        workers.emplace_back(worker, start, end);
    }

    for (auto& th : workers) {
        th.join();
    }

    std::vector<std::int64_t> ranking(num_vectors);
    std::iota(ranking.begin(), ranking.end(), static_cast<std::int64_t>(0));

    auto ascending_cmp = [&](std::int64_t a, std::int64_t b) {
        return distances[static_cast<std::size_t>(a)] < distances[static_cast<std::size_t>(b)];
    };

    if (k < num_vectors) {
        std::nth_element(ranking.begin(), ranking.begin() + static_cast<std::ptrdiff_t>(k), ranking.end(), ascending_cmp);
        ranking.resize(k);
    }
    std::sort(ranking.begin(), ranking.end(), ascending_cmp);

    py::array_t<std::int64_t> top_indices(ranking.size());
    py::array_t<float> top_distances(ranking.size());

    auto idx_view = top_indices.mutable_unchecked<1>();
    auto dist_view = top_distances.mutable_unchecked<1>();

    for (std::size_t i = 0; i < ranking.size(); ++i) {
        const std::int64_t idx = ranking[i];
        idx_view(static_cast<py::ssize_t>(i)) = idx;
        dist_view(static_cast<py::ssize_t>(i)) = distances[static_cast<std::size_t>(idx)];
    }

    return py::make_tuple(top_indices, top_distances);
}

}  // namespace

PYBIND11_MODULE(vector_search_cpp, m) {
    m.doc() = "SIMD + multi-threaded vector ranking for enterprise-scale RAG retrieval";

    m.def(
        "topk_cosine_similarity",
        &rank_cosine_impl,
        py::arg("query"),
        py::arg("matrix"),
        py::arg("top_k") = 5,
        py::arg("num_threads") = 0,
        py::arg("assume_normalized") = true,
        "Return top-k vector indices and cosine similarity scores"
    );

    m.def(
        "topk_euclidean_distance",
        &rank_euclidean_impl,
        py::arg("query"),
        py::arg("matrix"),
        py::arg("top_k") = 5,
        py::arg("num_threads") = 0,
        "Return top-k vector indices and L2 distances"
    );
}

#include "rbk_kernels.h"

#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" void rbk_fill_capacity_i32(
    const int64_t *order,
    const int32_t *neigh,
    int32_t *rem,
    int32_t *ids,
    int64_t n,
    int64_t L)
{
    for (int64_t t = 0; t < n; ++t) {
        const int64_t i = order[t];
        const int32_t *row = neigh + i * L;
        for (int64_t j = 0; j < L; ++j) {
            const int32_t c = row[j];
            if (c >= 0 && rem[c] > 0) {
                ids[i] = c;
                rem[c] -= 1;
                break;
            }
        }
    }
}

extern "C" void rbk_accumulate_sums_f32(
    const float *X,
    const int32_t *ids,
    double *sums,
    int64_t n,
    int64_t d,
    int64_t k)
{
    std::memset(sums, 0, static_cast<size_t>(k * d) * sizeof(double));

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif

    if (nthreads <= 1) {
        for (int64_t i = 0; i < n; ++i) {
            const int32_t c = ids[i];
            if (c < 0 || c >= k) continue;
            const float *xi = X + i * d;
            double *sc = sums + static_cast<int64_t>(c) * d;
            for (int64_t j = 0; j < d; ++j) sc[j] += static_cast<double>(xi[j]);
        }
        return;
    }

    std::vector<double> local(static_cast<size_t>(nthreads) * static_cast<size_t>(k * d), 0.0);

#ifdef _OPENMP
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        double *mine = local.data() + static_cast<size_t>(tid) * static_cast<size_t>(k * d);
#pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            const int32_t c = ids[i];
            if (c < 0 || c >= k) continue;
            const float *xi = X + i * d;
            double *sc = mine + static_cast<int64_t>(c) * d;
            for (int64_t j = 0; j < d; ++j) sc[j] += static_cast<double>(xi[j]);
        }
    }
#endif

    for (int t = 0; t < nthreads; ++t) {
        const double *src = local.data() + static_cast<size_t>(t) * static_cast<size_t>(k * d);
        for (int64_t j = 0; j < k * d; ++j) sums[j] += src[j];
    }
}

extern "C" double rbk_sum_sq_f32(
    const float *X,
    const float *C,
    const int32_t *ids,
    int64_t n,
    int64_t d)
{
    double total = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : total) schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        const int32_t c = ids[i];
        const float *xi = X + i * d;
        const float *ci = C + static_cast<int64_t>(c) * d;
        double s = 0.0;
        for (int64_t j = 0; j < d; ++j) {
            const double v = static_cast<double>(xi[j]) - static_cast<double>(ci[j]);
            s += v * v;
        }
        total += s;
    }
    return total;
}

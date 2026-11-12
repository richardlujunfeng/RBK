#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sequential capacity fill (order-dependent; not parallelized). */
void rbk_fill_capacity_i32(
    const int64_t *order,
    const int32_t *neigh,
    int32_t *rem,
    int32_t *ids,
    int64_t n,
    int64_t L);

/* OpenMP: accumulate per-cluster sums. sums is K x D (row-major, float64). */
void rbk_accumulate_sums_f32(
    const float *X,
    const int32_t *ids,
    double *sums,
    int64_t n,
    int64_t d,
    int64_t k);

/* OpenMP: SSE = sum_i ||x_i - c_{ids[i]}||^2 */
double rbk_sum_sq_f32(
    const float *X,
    const float *C,
    const int32_t *ids,
    int64_t n,
    int64_t d);

#ifdef __cplusplus
}
#endif

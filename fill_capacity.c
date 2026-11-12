#include <stdint.h>

void fill_capacity_i32(
    const int64_t *restrict order,
    const int32_t *restrict neigh,
    int32_t *restrict rem,
    int32_t *restrict ids,
    int64_t n,
    int64_t L)
{
    for (int64_t t = 0; t < n; ++t) {
        const int64_t i = order[t];
        const int32_t *restrict row = neigh + i * L;
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

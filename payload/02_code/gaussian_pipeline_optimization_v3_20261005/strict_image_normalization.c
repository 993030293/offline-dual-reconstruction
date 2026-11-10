/* Keep the original float32 division and physical interleaved output. */
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

int strict_normalization_abi(void) { return 1; }

int strict_normalization_avx2(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
}

static void scalar_division(const uint8_t *input, float *output, size_t count) {
    const __m128 divisor = _mm_set_ss(255.0f);
    for (size_t index = 0; index < count; ++index) {
        const __m128 value = _mm_set_ss((float)input[index]);
        _mm_store_ss(output + index, _mm_div_ss(value, divisor));
    }
}

__attribute__((target("avx2")))
static void vector_division(const uint8_t *input, float *output, size_t count) {
    size_t index = 0;
    const __m256 divisor = _mm256_set1_ps(255.0f);
    for (; count - index >= 8; index += 8) {
        const __m128i bytes = _mm_loadl_epi64((const __m128i *)(input + index));
        const __m256 values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
        _mm256_storeu_ps(output + index, _mm256_div_ps(values, divisor));
    }
    scalar_division(input + index, output + index, count - index);
}

int strict_normalization(const uint8_t *input, float *output, uint64_t count,
                         uint32_t vector_enabled) {
    if (vector_enabled > 1) return 1;
    if (count > UINT64_C(67108864)) return 2;
    if (count && (!input || !output)) return 3;
    if (vector_enabled && !strict_normalization_avx2()) return 4;
    if (!count) return 0;
    if (vector_enabled) vector_division(input, output, (size_t)count);
    else scalar_division(input, output, (size_t)count);
    return 0;
}

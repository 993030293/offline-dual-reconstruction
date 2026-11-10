// Independent RGGB quality-input fusion preserving OpenCV 4.13 u8 rounding.
// Algorithm references are recorded in R147_EXACT_FUSED_INPUT_TASK.md.
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <intrin.h>
#include <immintrin.h>

extern "C" __declspec(dllexport) int bayer_gray_avx2_available() {
    int registers[4];
    __cpuid(registers, 1);
    if ((registers[2] & (1 << 27)) == 0 || (registers[2] & (1 << 28)) == 0) return 0;
    if ((_xgetbv(0) & 6) != 6) return 0;
    __cpuid(registers, 0);
    if (registers[0] < 7) return 0;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
}

static __forceinline __m256i samples16(const std::uint8_t* pointer) {
    return _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pointer)));
}

static __forceinline __m256i select16(__m256i mask, __m256i when_true, __m256i when_false) {
    return _mm256_blendv_epi8(when_false, when_true, mask);
}

extern "C" __declspec(dllexport) int bayer_gray_exact_u8(
    const std::uint8_t* source, std::ptrdiff_t row_stride, int height, int width,
    std::uint8_t* destination, std::ptrdiff_t destination_stride) {
    if (!source || !destination || height <= 0 || width <= 0 ||
        row_stride < width || destination_stride < width) return 1;
    if (height <= 2 || width <= 2) {
        for (int row = 0; row < height; ++row) std::memset(destination + row*destination_stride, 0, width);
        return 0;
    }
    const __m256i one = _mm256_set1_epi16(1);
    const __m256i two = _mm256_set1_epi16(2);
    const __m256i even_x_mask = _mm256_setr_epi16(0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1);
    const __m256i bg_weights = _mm256_set1_epi32((19235 << 16) | 3735);
    const __m256i red_and_round_weights = _mm256_set1_epi32((1 << 16) | 9798);
    const __m256i rounding = _mm256_set1_epi16(16384);
    for (int y = 1; y < height - 1; ++y) {
        const auto* current = source + y*row_stride;
        const auto* upper = current - row_stride;
        const auto* lower = current + row_stride;
        auto* output = destination + y*destination_stride;
        int x = 1;
        for (; x + 15 < width - 1; x += 16) {
            __m256i center = samples16(current + x);
            __m256i lr = _mm256_add_epi16(samples16(current + x - 1), samples16(current + x + 1));
            __m256i ud = _mm256_add_epi16(samples16(upper + x), samples16(lower + x));
            __m256i diagonals = _mm256_add_epi16(
                _mm256_add_epi16(samples16(upper + x - 1), samples16(upper + x + 1)),
                _mm256_add_epi16(samples16(lower + x - 1), samples16(lower + x + 1)));
            __m256i crossed = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(lr, ud), two), 2);
            __m256i diagonal = _mm256_srli_epi16(_mm256_add_epi16(diagonals, two), 2);
            lr = _mm256_srli_epi16(_mm256_add_epi16(lr, one), 1);
            ud = _mm256_srli_epi16(_mm256_add_epi16(ud, one), 1);
            __m256i red, green, blue;
            if ((y & 1) == 0) {
                red = select16(even_x_mask, center, lr);
                green = select16(even_x_mask, crossed, center);
                blue = select16(even_x_mask, diagonal, ud);
            } else {
                red = select16(even_x_mask, ud, diagonal);
                green = select16(even_x_mask, center, crossed);
                blue = select16(even_x_mask, lr, center);
            }
            __m256i low = _mm256_add_epi32(
                _mm256_madd_epi16(_mm256_unpacklo_epi16(blue, green), bg_weights),
                _mm256_madd_epi16(_mm256_unpacklo_epi16(red, rounding), red_and_round_weights));
            __m256i high = _mm256_add_epi32(
                _mm256_madd_epi16(_mm256_unpackhi_epi16(blue, green), bg_weights),
                _mm256_madd_epi16(_mm256_unpackhi_epi16(red, rounding), red_and_round_weights));
            __m256i words = _mm256_packus_epi32(_mm256_srli_epi32(low, 15), _mm256_srli_epi32(high, 15));
            __m128i bytes = _mm_packus_epi16(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output + x), bytes);
        }
        for (; x < width - 1; ++x) {
            int red, green, blue;
            int lr = (current[x-1] + current[x+1] + 1) >> 1;
            int ud = (upper[x] + lower[x] + 1) >> 1;
            int diagonal = (upper[x-1] + upper[x+1] + lower[x-1] + lower[x+1] + 2) >> 2;
            int crossed = (current[x-1] + current[x+1] + upper[x] + lower[x] + 2) >> 2;
            if ((y & 1) == 0) {
                red = (x & 1) ? lr : current[x];
                green = (x & 1) ? current[x] : crossed;
                blue = (x & 1) ? ud : diagonal;
            } else {
                red = (x & 1) ? diagonal : ud;
                green = (x & 1) ? crossed : current[x];
                blue = (x & 1) ? current[x] : lr;
            }
            output[x] = static_cast<std::uint8_t>((3735*blue + 19235*green + 9798*red + 16384) >> 15);
        }
        output[0] = output[1];
        output[width-1] = output[width-2];
    }
    std::memcpy(destination, destination + destination_stride, width);
    std::memcpy(destination + (height-1)*destination_stride, destination + (height-2)*destination_stride, width);
    return 0;
}

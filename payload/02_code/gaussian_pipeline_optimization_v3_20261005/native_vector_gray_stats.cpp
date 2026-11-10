// Exact integer cross-Laplacian and gray moments for the bounded u8 consumer.
#include <cstdint>
#include <intrin.h>
#include <immintrin.h>

extern "C" __declspec(dllexport) int vector_gray_stats_abi() noexcept { return 2; }

extern "C" __declspec(dllexport) int vector_gray_stats_avx2_available() noexcept {
    int registers[4];
    __cpuid(registers, 1);
    if ((registers[2] & (1 << 27)) == 0 || (registers[2] & (1 << 28)) == 0 ||
        (registers[2] & (1 << 23)) == 0) return 0;
    if ((_xgetbv(0) & 6) != 6) return 0;
    __cpuid(registers, 0);
    if (registers[0] < 7) return 0;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
}

static __forceinline __m256i samples16(const std::uint8_t* pointer) {
    return _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pointer)));
}

static void flush(__m256i& sums, __m256i& squares, std::int64_t* output) noexcept {
    alignas(32) std::int32_t partial_sum[8], partial_square[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(partial_sum), sums);
    _mm256_store_si256(reinterpret_cast<__m256i*>(partial_square), squares);
    for (int lane = 0; lane < 8; ++lane) {
        output[2] += partial_sum[lane];
        output[3] += partial_square[lane];
    }
    sums = squares = _mm256_setzero_si256();
}

extern "C" __declspec(dllexport) int vector_gray_stats_u8(
    const std::uint8_t* gray, std::uint64_t rows, std::uint64_t cols,
    std::int64_t* output, std::uint64_t output_count) noexcept {
    constexpr std::uint64_t pixel_limit = UINT64_C(1) << 24;
    if (!gray || !output || !rows || !cols || output_count != 4 || rows > pixel_limit / cols) return 1;
    if (!vector_gray_stats_avx2_available()) return 2;
    for (int index = 0; index < 4; ++index) output[index] = 0;
    const __m256i one = _mm256_set1_epi16(1), six = _mm256_set1_epi16(6), high = _mm256_set1_epi16(249);
    __m256i sums = _mm256_setzero_si256(), squares = _mm256_setzero_si256();
    std::uint32_t blocks = 0;
    for (std::uint64_t y = 0; y < rows; ++y) {
        const auto up = y ? y - 1 : (rows > 1 ? 1 : 0);
        const auto down = y + 1 < rows ? y + 1 : (rows > 1 ? rows - 2 : 0);
        const auto* current = gray + y * cols;
        const auto* previous = gray + up * cols;
        const auto* next = gray + down * cols;
        const auto scalar = [&](std::uint64_t x) {
            const auto left = x ? x - 1 : (cols > 1 ? 1 : 0);
            const auto right = x + 1 < cols ? x + 1 : (cols > 1 ? cols - 2 : 0);
            const int center = current[x];
            const int derivative = previous[x] + next[x] + current[left] + current[right] - 4 * center;
            output[0] += center;
            output[1] += center <= 5 || center >= 250;
            output[2] += derivative;
            output[3] += derivative * derivative;
        };
        scalar(0);
        std::uint64_t x = 1;
        for (; x + 16 < cols; x += 16) {
            const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(current + x));
            const __m256i center = _mm256_cvtepu8_epi16(bytes);
            const __m256i neighbors = _mm256_add_epi16(
                _mm256_add_epi16(samples16(previous + x), samples16(next + x)),
                _mm256_add_epi16(samples16(current + x - 1), samples16(current + x + 1)));
            const __m256i derivative = _mm256_sub_epi16(neighbors, _mm256_slli_epi16(center, 2));
            sums = _mm256_add_epi32(sums, _mm256_madd_epi16(derivative, one));
            squares = _mm256_add_epi32(squares, _mm256_madd_epi16(derivative, derivative));
            const __m128i brightness = _mm_sad_epu8(bytes, _mm_setzero_si128());
            output[0] += _mm_cvtsi128_si64(brightness) + _mm_extract_epi64(brightness, 1);
            const __m256i clipped = _mm256_or_si256(_mm256_cmpgt_epi16(six, center), _mm256_cmpgt_epi16(center, high));
            output[1] += __popcnt(static_cast<unsigned int>(_mm256_movemask_epi8(clipped))) / 2;
            if (++blocks == 256) {
                flush(sums, squares, output);
                blocks = 0;
            }
        }
        for (; x < cols; ++x) scalar(x);
    }
    flush(sums, squares, output);
    return 0;
}

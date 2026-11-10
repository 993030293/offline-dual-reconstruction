// Preserve strict RGB interpolation and rounding while vectorizing the channels.
// The geometry and interpolation definition belong to COLMAP; no library code is copied.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <immintrin.h>
#include <intrin.h>

#pragma fp_contract(off)

extern "C" __declspec(dllexport) int native_camera_sampling_simd_capabilities() noexcept {
    int registers[4];
    __cpuid(registers, 0);
    if (registers[0] < 7) return 0;
    __cpuid(registers, 1);
    if ((registers[2] & (1 << 27)) == 0 || (registers[2] & (1 << 28)) == 0) return 0;
    if ((_xgetbv(0) & 6) != 6) return 0;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0 ? 1 : 0;
}

extern "C" __declspec(dllexport) int native_camera_sampling_abi() noexcept {
    return 1;
}

static __m256d rgb(const std::uint8_t* pixel) noexcept {
    // Read exactly three channels, including at the final source pixel.
    return _mm256_cvtepi32_pd(_mm_set_epi32(0, pixel[2], pixel[1], pixel[0]));
}

extern "C" __declspec(dllexport) int native_camera_sample_rgb8(
    const std::uint8_t* input, const std::int32_t* x0, const std::int32_t* y0,
    const double* dx, const double* dy, const std::uint8_t* valid,
    std::uint8_t* output, const std::size_t width, const std::size_t height) noexcept {
    if (!input || !x0 || !y0 || !dx || !dy || !valid || !output || width < 2 || height < 2 ||
        height > std::numeric_limits<std::size_t>::max() / width / 3) {
        return 1;
    }
    if (!native_camera_sampling_simd_capabilities()) return 3;
    const std::size_t count = width * height;
    const auto in_address = reinterpret_cast<std::uintptr_t>(input);
    const auto out_address = reinterpret_cast<std::uintptr_t>(output);
    const bool overlap = (out_address >= in_address ? out_address - in_address : in_address - out_address) < count * 3;
    for (std::size_t index = 0; index < count; ++index) {
        if (!valid[index]) {
            output[index * 3] = output[index * 3 + 1] = output[index * 3 + 2] = 0;
            continue;
        }
        const std::int32_t x = x0[index], y = y0[index];
        const double wx = dx[index], wy = dy[index];
        if (x < 0 || y < 0 || static_cast<std::size_t>(x) >= width - 1 ||
            static_cast<std::size_t>(y) >= height - 1 || !std::isfinite(wx) ||
            !std::isfinite(wy) || wx < 0 || wx >= 1 || wy < 0 || wy >= 1) {
            return 2;
        }
        const std::size_t source = (static_cast<std::size_t>(y) * width + x) * 3;
        if (overlap) {
            // Preserve the old observable channel order for aliased direct ABI calls.
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const double upper = (1.0 - wx) * input[source + channel] + wx * input[source + 3 + channel];
                const double lower = (1.0 - wx) * input[source + width * 3 + channel] + wx * input[source + width * 3 + 3 + channel];
                const float color = static_cast<float>((1.0 - wy) * upper + wy * lower);
                const double rounded = std::floor(static_cast<double>(color) + 0.5);
                output[index * 3 + channel] = static_cast<std::uint8_t>(rounded < 0 ? 0 : rounded > 255 ? 255 : rounded);
            }
            continue;
        }
        const __m256d left = _mm256_set1_pd(1.0 - wx);
        const __m256d right = _mm256_set1_pd(wx);
        const __m256d upper = _mm256_add_pd(_mm256_mul_pd(left, rgb(input + source)),
                                           _mm256_mul_pd(right, rgb(input + source + 3)));
        const __m256d lower = _mm256_add_pd(_mm256_mul_pd(left, rgb(input + source + width * 3)),
                                           _mm256_mul_pd(right, rgb(input + source + width * 3 + 3)));
        const __m256d combined = _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(1.0 - wy), upper),
                                              _mm256_mul_pd(_mm256_set1_pd(wy), lower));
        const __m256d color = _mm256_cvtps_pd(_mm256_cvtpd_ps(combined));
        const __m256d rounded = _mm256_floor_pd(_mm256_add_pd(color, _mm256_set1_pd(0.5)));
        const __m256d clamped = _mm256_min_pd(_mm256_set1_pd(255.0), _mm256_max_pd(_mm256_setzero_pd(), rounded));
        alignas(16) std::int32_t result[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(result), _mm256_cvttpd_epi32(clamped));
        output[index * 3] = static_cast<std::uint8_t>(result[0]);
        output[index * 3 + 1] = static_cast<std::uint8_t>(result[1]);
        output[index * 3 + 2] = static_cast<std::uint8_t>(result[2]);
    }
    return 0;
}

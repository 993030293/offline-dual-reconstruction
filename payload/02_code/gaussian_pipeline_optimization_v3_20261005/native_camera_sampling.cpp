// Implement the existing COLMAP 4.1.1 RGB sampling order for private cached maps.
// The interpolation definition belongs to COLMAP; this file contains no copied library code.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#pragma fp_contract(off)

extern "C" __declspec(dllexport) int native_camera_sampling_abi() noexcept {
    return 1;
}

extern "C" __declspec(dllexport) int native_camera_sample_rgb8(
    const std::uint8_t* input, const std::int32_t* x0, const std::int32_t* y0,
    const double* dx, const double* dy, const std::uint8_t* valid,
    std::uint8_t* output, const std::size_t width, const std::size_t height) noexcept {
    if (!input || !x0 || !y0 || !dx || !dy || !valid || !output || width < 2 || height < 2 ||
        height > std::numeric_limits<std::size_t>::max() / width / 3) {
        return 1;
    }
    const std::size_t count = width * height;
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
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const double upper = (1.0 - wx) * input[source + channel] +
                                 wx * input[source + 3 + channel];
            const double lower = (1.0 - wx) * input[source + width * 3 + channel] +
                                 wx * input[source + width * 3 + 3 + channel];
            const float color = static_cast<float>((1.0 - wy) * upper + wy * lower);
            const double rounded = std::floor(static_cast<double>(color) + 0.5);
            output[index * 3 + channel] = static_cast<std::uint8_t>(
                rounded < 0 ? 0 : rounded > 255 ? 255 : rounded);
        }
    }
    return 0;
}

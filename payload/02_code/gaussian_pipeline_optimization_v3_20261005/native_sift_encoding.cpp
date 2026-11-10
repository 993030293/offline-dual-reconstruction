// Preserve float32 SIFT byte representability while scanning each input once.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

extern "C" __declspec(dllexport) int encode_sift_bytes(
    const void* source, std::int64_t rows, std::int64_t row_stride,
    std::int64_t column_stride, std::uint8_t* target,
    std::int64_t target_row_stride, std::int64_t target_column_stride) {
    if (!source || !target || rows <= 0 || rows > 12000) return 4;
    const auto* base = static_cast<const char*>(source);
    bool nonfinite = false, invalid = false, overflow = false;
    for (std::int64_t row = 0; row < rows; ++row) {
        for (std::int64_t column = 0; column < 128; ++column) {
            float original;
            std::memcpy(&original, base + row * row_stride + column * column_stride, sizeof(float));
            if (!std::isfinite(original)) { nonfinite = true; continue; }
            const float scaled = original * 512.0f;
            if (!std::isfinite(scaled)) { overflow = true; invalid = true; continue; }
            if (scaled < 0.0f || scaled > 255.0f || std::trunc(scaled) != scaled) {
                invalid = true; continue;
            }
            const auto packed = static_cast<std::uint8_t>(scaled);
            const float restored = static_cast<float>(packed) / 512.0f;
            if (restored != original) { invalid = true; continue; }
            target[row * target_row_stride + column * target_column_stride] = packed;
        }
    }
    if (nonfinite) return 1;
    if (overflow) return 3;
    return invalid ? 2 : 0;
}

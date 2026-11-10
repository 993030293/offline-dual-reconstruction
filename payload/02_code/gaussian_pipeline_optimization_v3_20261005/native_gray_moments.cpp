#include <cstdint>
#include <limits>

extern "C" __declspec(dllexport) std::uint64_t gray_moments_abi() noexcept {
    return 1;
}

extern "C" __declspec(dllexport) int gray_moments(
    const std::uint8_t* gray, std::uint64_t rows, std::uint64_t cols,
    std::uint64_t* histogram, std::uint64_t histogram_count,
    std::int64_t* derivative_sum, std::uint64_t* derivative_squared_sum) noexcept {
    constexpr std::uint64_t pixel_limit = UINT64_C(1) << 24;
    if (!gray || !histogram || !derivative_sum || !derivative_squared_sum ||
        !rows || !cols || histogram_count != 256 || rows > pixel_limit / cols) {
        return 1;
    }
    for (std::uint64_t index = 0; index < 256; ++index) {
        histogram[index] = 0;
    }
    std::int64_t sum = 0;
    std::uint64_t squared_sum = 0;
    for (std::uint64_t y = 0; y < rows; ++y) {
        const auto up = y ? y - 1 : (rows > 1 ? 1 : 0);
        const auto down = y + 1 < rows ? y + 1 : (rows > 1 ? rows - 2 : 0);
        const auto* current = gray + y * cols;
        const auto* previous = gray + up * cols;
        const auto* next = gray + down * cols;
        for (std::uint64_t x = 0; x < cols; ++x) {
            const auto left = x ? x - 1 : (cols > 1 ? 1 : 0);
            const auto right = x + 1 < cols ? x + 1 : (cols > 1 ? cols - 2 : 0);
            const int value = static_cast<int>(previous[x]) + next[x] +
                              current[left] + current[right] - 4 * current[x];
            ++histogram[current[x]];
            sum += value;
            squared_sum += static_cast<std::uint64_t>(value * value);
        }
    }
    *derivative_sum = sum;
    *derivative_squared_sum = squared_sum;
    return 0;
}

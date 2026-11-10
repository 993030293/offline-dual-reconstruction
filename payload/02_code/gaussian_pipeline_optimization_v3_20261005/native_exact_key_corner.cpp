// Sort original candidate priorities as exact integer keys; retain the grid predicate.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

static std::uint32_t float_rank(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000U) ? ~bits : bits ^ 0x80000000U;
}

static std::vector<std::uint64_t> collect(const float* response, const float* dilated,
                                        int height, int width) {
    std::vector<std::uint64_t> values;
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            const int index = y * width + x;
            const float value = response[index];
            if (value != 0.0f && value == dilated[index])
                values.push_back((static_cast<std::uint64_t>(float_rank(value)) << 32)
                                 | static_cast<std::uint32_t>(index));
        }
    }
    return values;
}

static void stable_radix(std::vector<std::uint64_t>& values) {
    // Stable response-key passes retain the original increasing index for ties.
    std::vector<std::uint64_t> scratch(values.size());
    for (int shift = 32; shift < 64; shift += 8) {
        std::array<std::size_t, 256> offsets{};
        for (auto value : values) ++offsets[(value >> shift) & 255U];
        int occupied = 0;
        for (auto count : offsets) occupied += count != 0;
        if (occupied <= 1) continue;
        std::size_t position = 0;
        for (auto& count : offsets) {
            const std::size_t prior = count;
            count = position;
            position += prior;
        }
        for (auto value : values) scratch[offsets[(value >> shift) & 255U]++] = value;
        values.swap(scratch);
    }
}

static int select_ordered(const std::vector<std::uint64_t>& values, bool reverse,
                          int height, int width, float* xy_output, int* selected_count) {
    const int grid_width = (width + 4) / 5;
    const int grid_height = (height + 4) / 5;
    std::vector<int> heads(grid_width * grid_height, -1);
    std::vector<int> tails(grid_width * grid_height, -1);
    int next[800];
    int count = 0;
    for (std::size_t position = 0; position < values.size(); ++position) {
        const std::uint64_t key = values[reverse ? values.size() - 1 - position : position];
        const int index = static_cast<int>(static_cast<std::uint32_t>(key));
        const int y = index / width;
        const int x = index % width;
        const int cell_y = y / 5;
        const int cell_x = x / 5;
        bool allowed = true;
        for (int row = std::max(0, cell_y - 1); row <= std::min(grid_height - 1, cell_y + 1) && allowed; ++row) {
            for (int column = std::max(0, cell_x - 1); column <= std::min(grid_width - 1, cell_x + 1) && allowed; ++column) {
                for (int previous = heads[row * grid_width + column]; previous != -1; previous = next[previous]) {
                    const float dx = static_cast<float>(x) - xy_output[2 * previous];
                    const float dy = static_cast<float>(y) - xy_output[2 * previous + 1];
                    if (dx * dx + dy * dy < 25.0f) { allowed = false; break; }
                }
            }
        }
        if (!allowed) continue;
        xy_output[2 * count] = static_cast<float>(x);
        xy_output[2 * count + 1] = static_cast<float>(y);
        next[count] = -1;
        const int cell = cell_y * grid_width + cell_x;
        if (tails[cell] == -1) heads[cell] = count;
        else next[tails[cell]] = count;
        tails[cell] = count;
        if (++count == 800) break;
    }
    *selected_count = count;
    return 0;
}

static int select_impl(const float* response, const float* dilated, int height, int width,
                       float* xy_output, int* selected_count, bool radix) {
    if (!response || !dilated || !xy_output || !selected_count || height <= 0 || width <= 0
        || static_cast<std::uint64_t>(height) * width > (1ULL << 24)) return 1;
    try {
        auto values = collect(response, dilated, height, width);
        if (radix) stable_radix(values);
        else std::sort(values.begin(), values.end(), std::greater<std::uint64_t>());
        return select_ordered(values, radix, height, width, xy_output, selected_count);
    } catch (...) {
        return 2;
    }
}

extern "C" __declspec(dllexport) int exact_key_corner_abi() { return 2; }
extern "C" __declspec(dllexport) int packed_corner_select(
    const float* response, const float* dilated, int height, int width,
    float* xy_output, int* selected_count) {
    return select_impl(response, dilated, height, width, xy_output, selected_count, false);
}
extern "C" __declspec(dllexport) int radix_corner_select(
    const float* response, const float* dilated, int height, int width,
    float* xy_output, int* selected_count) {
    return select_impl(response, dilated, height, width, xy_output, selected_count, true);
}

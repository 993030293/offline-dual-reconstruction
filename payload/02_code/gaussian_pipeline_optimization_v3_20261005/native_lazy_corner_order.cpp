// Preserve the existing candidate priority and grid predicate without sorting unused tails.
#include <algorithm>
#include <cstdint>
#include <vector>

struct LazyCandidate {
    float value;
    int index;
};

extern "C" __declspec(dllexport) int lazy_corner_order_abi() { return 1; }

extern "C" __declspec(dllexport) int lazy_corner_select(
    const float* response, const float* dilated, int height, int width,
    float* xy_output, int* selected_count) {
    if (!response || !dilated || !xy_output || !selected_count || height <= 0 || width <= 0
        || static_cast<std::uint64_t>(height) * width > (1ULL << 24)) return 1;
    try {
        std::vector<LazyCandidate> candidates;
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const int index = y * width + x;
                const float value = response[index];
                if (value != 0.0f && value == dilated[index]) candidates.push_back({value, index});
            }
        }
        const auto lower_priority = [](const LazyCandidate& left, const LazyCandidate& right) {
            if (left.value != right.value) return left.value < right.value;
            return left.index < right.index;
        };
        std::make_heap(candidates.begin(), candidates.end(), lower_priority);
        const int grid_width = (width + 4) / 5;
        const int grid_height = (height + 4) / 5;
        std::vector<int> heads(grid_width * grid_height, -1);
        std::vector<int> tails(grid_width * grid_height, -1);
        int next[800];
        int count = 0;
        while (!candidates.empty()) {
            const LazyCandidate candidate = candidates.front();
            std::pop_heap(candidates.begin(), candidates.end(), lower_priority);
            candidates.pop_back();
            const int y = candidate.index / width;
            const int x = candidate.index % width;
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
    } catch (...) {
        return 2;
    }
}

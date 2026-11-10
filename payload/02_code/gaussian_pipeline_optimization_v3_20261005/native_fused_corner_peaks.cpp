// Omit the materialized dilation image while retaining original peak and grid rules.
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <vector>

struct PeakCandidate { float value; int index; };

extern "C" __declspec(dllexport) int fused_corner_peaks_abi() { return 1; }
extern "C" __declspec(dllexport) int fused_corner_peaks_select(
    float* response, int height, int width, float* xy_output, int* selected_count) {
    if (!response || !xy_output || !selected_count || height <= 0 || width <= 0
        || static_cast<std::uint64_t>(height) * width > (1ULL << 24)) return 1;
    try {
        const int size = height * width;
        const float infinity = std::numeric_limits<float>::infinity();
        const __m256 inf = _mm256_set1_ps(infinity);
        const __m256 signless = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        __m256 maxima = _mm256_set1_ps(-std::numeric_limits<float>::max());
        int position = 0;
        for (; position + 8 <= size; position += 8) {
            const __m256 value = _mm256_loadu_ps(response + position);
            if (_mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(value, signless), inf, _CMP_LT_OQ)) != 255) return 3;
            maxima = _mm256_max_ps(maxima, value);
        }
        float lanes[8];
        _mm256_storeu_ps(lanes, maxima);
        float maximum = -std::numeric_limits<float>::max();
        for (float lane : lanes) maximum = std::max(maximum, lane);
        for (; position < size; ++position) {
            const float value = response[position];
            if (!(value < infinity && value > -infinity)) return 3;
            maximum = std::max(maximum, value);
        }
        const float threshold = static_cast<float>(static_cast<double>(maximum) * .01);
        const __m256 limit = _mm256_set1_ps(threshold);
        position = 0;
        for (; position + 8 <= size; position += 8) {
            const __m256 value = _mm256_loadu_ps(response + position);
            _mm256_storeu_ps(response + position, _mm256_and_ps(value, _mm256_cmp_ps(value, limit, _CMP_GT_OQ)));
        }
        for (; position < size; ++position) if (!(response[position] > threshold)) response[position] = 0.0f;
        std::vector<PeakCandidate> candidates;
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const int index = y * width + x;
                const float value = response[index];
                if (value == 0.0f) continue;
                bool peak = true;
                for (int row = y - 1; row <= y + 1 && peak; ++row) {
                    for (int column = x - 1; column <= x + 1; ++column) {
                        if (value < response[row * width + column]) { peak = false; break; }
                    }
                }
                if (peak) candidates.push_back({value, index});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const PeakCandidate& left, const PeakCandidate& right) {
            if (left.value != right.value) return left.value > right.value;
            return left.index > right.index;
        });
        const int grid_width = (width + 4) / 5;
        const int grid_height = (height + 4) / 5;
        std::vector<int> heads(grid_width * grid_height, -1), tails(grid_width * grid_height, -1);
        int next[800];
        int count = 0;
        for (const auto& candidate : candidates) {
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

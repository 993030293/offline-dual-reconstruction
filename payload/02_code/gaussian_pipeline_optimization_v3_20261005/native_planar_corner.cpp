// Independent planar covariance layout; algorithm provenance is in R149_PLANAR_CORNER_TASK.md.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <intrin.h>
#include <immintrin.h>
#include <vector>

extern "C" __declspec(dllexport) int planar_corner_avx2_available() {
    int registers[4];
    __cpuid(registers, 1);
    if ((registers[2] & (1 << 27)) == 0 || (registers[2] & (1 << 28)) == 0 || (_xgetbv(0) & 6) != 6) return 0;
    __cpuid(registers, 0);
    if (registers[0] < 7) return 0;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
}

extern "C" __declspec(dllexport) int planar_corner_covariance(
    const float* dx, const float* dy, float* xx, float* xy, float* yy, int count) {
    if (!dx || !dy || !xx || !xy || !yy || count <= 0) return 1;
    int index = 0;
    for (; index + 7 < count; index += 8) {
        const __m256 horizontal = _mm256_loadu_ps(dx + index);
        const __m256 vertical = _mm256_loadu_ps(dy + index);
        _mm256_storeu_ps(xx + index, _mm256_mul_ps(horizontal, horizontal));
        _mm256_storeu_ps(xy + index, _mm256_mul_ps(horizontal, vertical));
        _mm256_storeu_ps(yy + index, _mm256_mul_ps(vertical, vertical));
    }
    for (; index < count; ++index) {
        xx[index] = dx[index]*dx[index];
        xy[index] = dx[index]*dy[index];
        yy[index] = dy[index]*dy[index];
    }
    return 0;
}

extern "C" __declspec(dllexport) int planar_corner_eigen(
    const float* xx, const float* xy, const float* yy, float* response, int count) {
    if (!xx || !xy || !yy || !response || count <= 0) return 1;
    const __m256 half = _mm256_set1_ps(0.5f);
    int index = 0;
    for (; index + 7 < count; index += 8) {
        const __m256 left = _mm256_mul_ps(_mm256_loadu_ps(xx + index), half);
        const __m256 right = _mm256_mul_ps(_mm256_loadu_ps(yy + index), half);
        const __m256 cross = _mm256_loadu_ps(xy + index);
        const __m256 difference = _mm256_sub_ps(left, right);
        const __m256 squared = _mm256_add_ps(_mm256_mul_ps(cross, cross), _mm256_mul_ps(difference, difference));
        _mm256_storeu_ps(response + index, _mm256_sub_ps(_mm256_add_ps(left, right), _mm256_sqrt_ps(squared)));
    }
    for (; index < count; ++index) {
        const float left = xx[index]*0.5f;
        const float right = yy[index]*0.5f;
        const float cross = xy[index];
        const float difference = left-right;
        response[index] = left+right-std::sqrt(difference*difference+cross*cross);
    }
    return 0;
}

struct Candidate {
    float value;
    int index;
};

extern "C" __declspec(dllexport) int planar_corner_select(
    const float* response, const float* dilated, int height, int width, float* xy_output, int* selected_count) {
    if (!response || !dilated || !xy_output || !selected_count || height <= 0 || width <= 0) return 1;
    try {
        std::vector<Candidate> candidates;
        for (int y = 1; y < height-1; ++y) {
            for (int x = 1; x < width-1; ++x) {
                const int index = y*width+x;
                const float value = response[index];
                if (value != 0.0f && value == dilated[index]) candidates.push_back({value,index});
            }
        }
        std::sort(candidates.begin(),candidates.end(),[](const Candidate& left,const Candidate& right) {
            if (left.value != right.value) return left.value > right.value;
            return left.index > right.index;
        });
        const int grid_width = (width+4)/5;
        const int grid_height = (height+4)/5;
        std::vector<int> heads(grid_width*grid_height,-1);
        std::vector<int> tails(grid_width*grid_height,-1);
        int next[800];
        int count = 0;
        for (const Candidate& candidate : candidates) {
            const int y = candidate.index/width;
            const int x = candidate.index%width;
            const int cell_y = y/5;
            const int cell_x = x/5;
            bool allowed = true;
            for (int row = std::max(0,cell_y-1); row <= std::min(grid_height-1,cell_y+1) && allowed; ++row) {
                for (int column = std::max(0,cell_x-1); column <= std::min(grid_width-1,cell_x+1) && allowed; ++column) {
                    for (int previous = heads[row*grid_width+column]; previous != -1; previous = next[previous]) {
                        const float dx = static_cast<float>(x)-xy_output[2*previous];
                        const float dy = static_cast<float>(y)-xy_output[2*previous+1];
                        if (dx*dx+dy*dy < 25.0f) { allowed = false; break; }
                    }
                }
            }
            if (!allowed) continue;
            xy_output[2*count] = static_cast<float>(x);
            xy_output[2*count+1] = static_cast<float>(y);
            next[count] = -1;
            const int cell = cell_y*grid_width+cell_x;
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

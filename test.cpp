#include <immintrin.h>
#include <vector>
#include <stdexcept>
#include <iostream>

std::vector<double> linspace(double start, double stop, int num, bool endpoint = true) {
    if (num < 0) 
        throw std::invalid_argument("num must be non-negative");

    std::vector<double> result(num);
    if (num == 0) 
        return result;
    
    if (num == 1) {
        result[0] = start;
        return result;
    }

    const int simd_width = 4;
    double step = endpoint ?
        (stop - start) / (num - 1) :
        (stop - start) / num;

    __m256d step_v = _mm256_set1_pd(step);
    __m256d start_v = _mm256_set1_pd(start);

    alignas(32) double offsets[simd_width];
    for (int i = 0; i < simd_width; ++i) 
        offsets[i] = i;
    __m256d offsets_v = _mm256_loadu_pd(offsets);

    int i = 0;
    for (; i <= num - simd_width; i += simd_width) {
        __m256d base = _mm256_set1_pd(static_cast<double>(i));

        __m256d indices = _mm256_add_pd(base, offsets_v);

        __m256d values = _mm256_add_pd(_mm256_mul_pd(indices, step_v), start_v);

        _mm256_storeu_pd(&result[i], values);
    }

    for (; i < num; ++i) {
        result[i] = start + static_cast<double>(i) * step;
    }

    if (endpoint) {
        result.back() = stop;
    }

    return result;
}

int main() {
    auto res = linspace(0.0, 11.0, 100, false);

    for (const auto& item : res)
        std::cout << item << ' ';
}
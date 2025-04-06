// #include <immintrin.h>
// #include <vector>
// #include <stdexcept>
// #include <iostream>

// std::vector<double> linspace(double start, double stop, int num, bool endpoint = true) {
//     if (num < 0) 
//         throw std::invalid_argument("num must be non-negative");

//     std::vector<double> result(num);
//     if (num == 0) 
//         return result;
    
//     if (num == 1) {
//         result[0] = start;
//         return result;
//     }

//     const int simd_width = 4;
//     double step = endpoint ?
//         (stop - start) / (num - 1) :
//         (stop - start) / num;

//     __m256d step_v = _mm256_set1_pd(step);
//     __m256d start_v = _mm256_set1_pd(start);

//     alignas(32) double offsets[simd_width];
//     for (int i = 0; i < simd_width; ++i) 
//         offsets[i] = i;
//     __m256d offsets_v = _mm256_loadu_pd(offsets);

//     int i = 0;
//     for (; i <= num - simd_width; i += simd_width) {
//         __m256d base = _mm256_set1_pd(static_cast<double>(i));

//         __m256d indices = _mm256_add_pd(base, offsets_v);

//         __m256d values = _mm256_add_pd(_mm256_mul_pd(indices, step_v), start_v);

//         _mm256_storeu_pd(&result[i], values);
//     }

//     for (; i < num; ++i) {
//         result[i] = start + static_cast<double>(i) * step;
//     }

//     if (endpoint) {
//         result.back() = stop;
//     }

//     return result;
// }

// int main() {
//     auto res = linspace(0.0, 11.0, 100, false);

//     for (const auto& item : res)
//         std::cout << item << ' ';
// }

#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>

int main() {
    // 初始化两个包含 8 个单精度浮点数的 __m256 向量
    __m256 vec1 = _mm256_set_ps(1.1f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    __m256 vec2 = _mm256_set_ps(1.1f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f);

    float float_vec1[] = {1.1f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float float_vec2[] = {1.1f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    // 调用 _mm256_and_ps 执行按位与运算
    __m256 result = _mm256_and_ps(vec1, vec2);

    // 将结果存储到一个数组中
    float output[8];
    _mm256_storeu_ps(output, result);

    // 输出 SIMD 结果
    printf("SIMD result:\n");
    for (int i = 7; ~i; i--) {
        printf("The %d element: %f\n", i, output[i]);
    }

    // 输出普通结果
    printf("Plain result:\n");
    float plain_output[8];
    for (int i = 0; i < 8; ++i) {
        uint32_t int_val1 = *(uint32_t*)&float_vec1[i];
        uint32_t int_val2 = *(uint32_t*)&float_vec2[i];
        uint32_t and_result = int_val1 & int_val2;
        plain_output[i] = *(float*)&and_result;
        printf("The %d element: %f\n", i, plain_output[i]);
    }

    return 0;
}
    
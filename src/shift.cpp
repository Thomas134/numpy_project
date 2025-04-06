#include "../include/shift.hpp"
#include "../include/simd_traits.hpp"
#include <immintrin.h>
#include <stdexcept>

namespace internal {
    // slli1_simd
    template <typename T>
    std::vector<T> slli1_simd(const std::vector<T>& A, const int imm8) {
        if (A.empty())
            throw std::invalid_argument("Vector can't be empty");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = slli_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = slli_simd_traits<T>::load(&A[i]);
            auto vec_result = slli_simd_traits<T>::bitwise_slli(vec_a, imm8);
            slli_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i) {
            result[i] = A[i] << imm8;
        }

        return result;
    }
}

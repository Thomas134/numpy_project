#include "../include/shift.hpp"
#include "../include/simd_traits.hpp"
#include "../include/utils/utils.hpp"
#include "../include/utils/simd_operators.hpp"
#include <immintrin.h>
#include <stdexcept>

namespace internal {
    // slli1_simd
    template <typename T>
    std::vector<T> slli1_simd(const std::vector<T>& A, const int imm8) {
        return apply_unary_op_simd<T, slli_simd_traits<T>>(A, std::min);
    }


    // srli1_simd
    template <typename T>
    std::vector<T> srli1_simd(const std::vector<T>& A, const int imm8) {
        return apply_unary_op_simd<T, srli_simd_traits<T>>(A, std::min);
    }
}

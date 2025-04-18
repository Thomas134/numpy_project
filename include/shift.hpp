#ifndef SHIFT_HPP
#define SHIFT_HPP

#include <vector>

#include "simd_traits.hpp"
#include "utils/utils.hpp"
#include "utils/simd_operators.hpp"
#include <immintrin.h>
#include <stdexcept>

namespace internal {
    // ========================== 1D =============================

    // slli1
    template <typename T>
    std::vector<T> slli1_simd(const std::vector<T>& A, const int imm);


    // srli1
    template <typename T>
    std::vector<T> srli1_simd(const std::vector<T>& A, const int imm);


    // ========================== 2D =============================

    // slli2
    template <typename T>
    std::vector<std::vector<T>> slli2_simd(const std::vector<std::vector<T>>& A, const int imm);

    
    // srli2
    template <typename T>
    std::vector<std::vector<T>> srli2_simd(const std::vector<std::vector<T>>& A, const int imm);
}

namespace internal {
    // ========================== 1D =============================
 
    // slli1_simd
    template <typename T>
    std::vector<T> slli1_simd(const std::vector<T>& A, const int imm8) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, [imm8](const T& element) { return element << imm8; }); 

        return apply_unary_op_simd<T, slli_simd_traits<T>>(A, [imm8](const T& element) { return element << imm8; });
    }


    // srli1_simd
    template <typename T>
    std::vector<T> srli1_simd(const std::vector<T>& A, const int imm8) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, [imm8](const T& element) { return element >> imm8; });

        return apply_unary_op_simd<T, srli_simd_traits<T>>(A, [imm8](const T& element) { return element >> imm8; });
    }


    // ========================== 2D =============================

    // slli2_simd
    template <typename T>
    std::vector<std::vector<T>> slli2_simd(const std::vector<std::vector<T>>& A, const int imm) {
        return apply_unary_op(A, slli1_simd);
    }


    // srli2_simd
    template <typename T>
    std::vector<std::vector<T>> srli2_simd(const std::vector<std::vector<T>>& A, const int imm) {
        return apply_unary_op(A, srli1_simd);
    }
}


#endif

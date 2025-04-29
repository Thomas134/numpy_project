#ifndef SHIFT_HPP
#define SHIFT_HPP

#include <vector>

#include "simd_traits.cpp"
#include "utils/utils.cpp"
#include "utils/simd_operators.cpp"
#ifdef __AVX2__
    #include <immintrin.h>
#endif
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

        #ifdef __riscv
            return apply_unary_op_plain(A, [imm8](const T& element) { return element << imm8; }); 
        #endif

        #ifdef __AVX2__
            return apply_unary_op_simd_shift<T, slli_simd_traits<T>>(A, imm8, [imm8](const T& element) { return element << imm8; });
        #endif
    }


    // srli1_simd
    template <typename T>
    std::vector<T> srli1_simd(const std::vector<T>& A, const int imm8) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, [imm8](const T& element) { return element >> imm8; });

        #ifdef __riscv
            return apply_unary_op_plain(A, [imm8](const T& element) { return element >> imm8; });
        #endif

        #ifdef __AVX2__
            return apply_unary_op_simd_shift<T, srli_simd_traits<T>>(A, imm8, [imm8](const T& element) { return element >> imm8; });
        #endif
    }


    // ========================== 2D =============================

    #ifdef __AVX2__
    // slli2_simd
    template <typename T>
    std::vector<std::vector<T>> slli2_simd(const std::vector<std::vector<T>>& A, const int imm8) {
        return apply_unary_op_shift(A, imm8, slli1_simd<T>);
    }


    // srli2_simd
    template <typename T>
    std::vector<std::vector<T>> srli2_simd(const std::vector<std::vector<T>>& A, const int imm8) {
        return apply_unary_op_shift(A, imm8, srli1_simd<T>);
    }

    #endif
}


#endif

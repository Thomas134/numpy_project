#ifndef LOGICAL_HPP
#define LOGICAL_HPP

#include <vector>
#include "simd_traits.cpp"
#include "utils/utils.cpp"
#include "utils/simd_operators.cpp"
#include <immintrin.h>
#include <stdexcept>

namespace internal {
    // =========================== 1D ======================================
    // and1
    template <typename T>
    std::vector<T> and1_simd(const std::vector<T>& A, const std::vector<T>& B);


    // or1
    template <typename T>
    std::vector<T> or1_simd(const std::vector<T>& A, const std::vector<T>& B);


    // xor1
    template <typename T>
    std::vector<T> xor1_simd(const std::vector<T>& A, const std::vector<T>& B);


    // andnot1
    template <typename T>
    std::vector<T> andnot1_simd(const std::vector<T>& A, const std::vector<T>& B);


    // testc1
    template <typename T>
    std::vector<T> testc1_simd(const std::vector<T>& A, const std::vector<T>& B);


    // =========================== 2D ======================================
    
    // and2
    template <typename T>
    std::vector<std::vector<T>> and2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // or2
    template <typename T>
    std::vector<std::vector<T>> or2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // xor2
    template <typename T>
    std::vector<std::vector<T>> xor2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // andnot2
    template <typename T>
    std::vector<std::vector<T>> andnot2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);
}


namespace internal {
    // =========================== 1D ======================================

    // and1_simd
    template <typename T>
    std::vector<T> and1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() < 32)
            return apply_binary_op_plain(A, B, [](const T& element1, const T& element2) { return element1 & element2; });

        return apply_binary_op_simd<T, and_simd_traits<T>>(A, B, [](const T& element1, const T& element2) { return element1 & element2; });
    }


    // or1_simd
    template <typename T>
    std::vector<T> or1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() < 32)
            return apply_binary_op_plain(A, B, [](const T& element1, const T& element2) { return element1 | element2; });

        return apply_binary_op_simd<T, or_simd_traits<T>>(A, B, [](const T& element1, const T& element2) { return element1 | element2; });
    }


    // xor1_simd
    template <typename T>
    std::vector<T> xor1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() < 32)
            return apply_binary_op_plain(A, B, [](const T& element1, const T& element2) { return element1 ^ element2; });

        return apply_binary_op_simd<T, xor_simd_traits<T>>(A, B, [](const T& element1, const T& element2) { return element1 ^ element2; });
    }


    // andnot1_simd
    template <typename T>
    std::vector<T> andnot1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() < 32)
            return apply_binary_op_plain(A, B, [](const T& element1, const T& element2) { return ~element1 & element2; });

        return apply_binary_op_simd<T, andnot_simd_traits<T>>(A, B, [](const T& element1, const T& element2) { return ~element1 & element2; });
    }


    // testc1_simd
    template <typename T>
    int testc1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() != B.size())
            throw std::invalid_argument("Input vectors must have the same size.");

        int result = 1;
        const size_t simd_step = andnot_simd_traits<T>::step;
        size_t i = 0;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = testc_simd_traits<T>::load(&A[i]);
            auto vec_b = testc_simd_traits<T>::load(&B[i]);
            result &= testc_simd_traits<T>::bitwise_testc(vec_a, vec_b);
        }

        for (; i < A.size(); ++i)
            result &= !(~A[i] & B[i]);

        return result;
    }


    // =========================== 2D ======================================

    // and2_simd
    template <typename T>
    std::vector<std::vector<T>> and2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        return apply_binary_op(A, B, and1_simd<T>);
    }


    // or2_simd
    template <typename T>
    std::vector<std::vector<T>> or2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        return apply_binary_op(A, B, or1_simd<T>);
    }


    // xor2_simd
    template <typename T>
    std::vector<std::vector<T>> xor2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        return apply_binary_op(A, B, xor1_simd<T>);
    }


    // andnot2_simd
    template <typename T>
    std::vector<std::vector<T>> andnot2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        return apply_binary_op(A, B, andnot1_simd<T>);
    }
}


#endif

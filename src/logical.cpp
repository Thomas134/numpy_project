#include "../include/logical.hpp"
#include "../include/simd_traits.hpp"
#include <immintrin.h>

namespace internal {
    // and1_simd
    template <typename T>
    std::vector<T> and1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() != B.size())
            throw std::invalid_argument("Input vectors must have the same size.");

        std::vector<T> result(A.size());
        const size_t simd_step = and_simd_traits<T>::step;
        size_t i = 0;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = and_simd_traits<T>::load(&A[i]);
            auto vec_b = and_simd_traits<T>::load(&B[i]);
            auto vec_result = and_simd_traits<T>::bitwise_and(vec_a, vec_b);
            and_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i) {
            result[i] = A[i] & B[i];
        }

        return result;
    }


    // or1_simd
    template <typename T>
    std::vector<T> or1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() != B.size())
            throw std::invalid_argument("Input vectors must have the same size.");

        std::vector<T> result(A.size());
        const size_t simd_step = or_simd_traits<T>::step;
        size_t i = 0;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = or_simd_traits<T>::load(&A[i]);
            auto vec_b = or_simd_traits<T>::load(&B[i]);
            auto vec_result = or_simd_traits<T>::bitwise_or(vec_a, vec_b);
            or_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i) {
            result[i] = A[i] | B[i];
        }

        return result;
    }


    // xor1_simd
    template <typename T>
    std::vector<T> xor1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() != B.size())
            throw std::invalid_argument("Input vectors must have the same size.");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = xor_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = xor_simd_traits<T>::load(&A[i]);
            auto vec_b = xor_simd_traits<T>::load(&B[i]);
            auto vec_result = xor_simd_traits<T>::bitwise_xor(vec_a, vec_b);
            xor_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i) {
            result[i] = A[i] ^ B[i];
        }

        return result;
    }


    // andnot_simd
    template <typename T>
    std::vector<T> andnot1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() != B.size())
            throw std::invalid_argument("Input vectors must have the same size.");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = andnot_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = andnot_simd_traits<T>::load(&A[i]);
            auto vec_b = andnot_simd_traits<T>::load(&B[i]);
            auto vec_result = andnot_simd_traits<T>::bitwise_andnot(vec_a, vec_b);
            andnot_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i) {
            result[i] = ~A[i] & B[i];
        }

        return result;
    }
}

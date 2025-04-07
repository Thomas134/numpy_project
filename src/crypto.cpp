#include "../include/crypto.hpp"
#include "../include/simd_traits.hpp"
#include <immintrin.h>
#include <stdexcept>

namespace internal {
    // sm4rnds4_1_simd
    template <typename T>
    std::vector<T> sm4rnds4_1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.empty() || B.empty())
            throw std::invalid_argument("Input vectors can't be empty");

        if (A.size() != B.size())
            throw std::invalid_argument("Input vectors must be of the same size.");

        if (A.size() % 8 != 0)
            throw std::invalid_argument("The length of the input vectors must be an integer multiple of 8.");


        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = sm4rnds4_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = sm4rnds4_simd_traits<T>::load(&A[i]);
            auto vec_b = sm4rnds4_simd_traits<T>::load(&B[i]);
            auto vec_result = sm4rnds4_simd_traits<T>::sm4rnds4(vec_a, vec_b);
            sm4rnds4_simd_traits<T>::store(&result[i], vec_result);
        }

        return result;
    }


    // sm4key4_1_simd
    template <typename T>
    std::vector<T> sm4key4_1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.empty() || B.empty())
            throw std::invalid_argument("Input vectors can't be empty");

        if (A.size() != B.size())
            throw std::invalid_argument("Input vectors must be of the same size.");

        if (A.size() % 8 != 0)
            throw std::invalid_argument("The length of the input vectors must be an integer multiple of 8.");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = sm4key4_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = sm4key4_simd_traits<T>::load(&A[i]);
            auto vec_b = sm4key4_simd_traits<T>::load(&B[i]);
            auto vec_result = sm4key4_simd_traits<T>::sm4key4(vec_a, vec_b);
            sm4key4_simd_traits<T>::store(&result[i], vec_result);
        }

        return result;
    }
}

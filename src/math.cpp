#include "../include/math.hpp"
#include <immintrin.h>
#include <stdexcept>
#include <algorithm>


namespace internal {
    template <typename T>
    typename inner_product_simd_traits<T>::accum_type inner_product_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() != B.size()) {
            throw std::invalid_argument("Input vectors must have the same size");
        }

        using Traits = inner_product_simd_traits<T>;
        using simd_type = typename Traits::simd_type;
        using accum_type = typename Traits::accum_type;

        accum_type result = static_cast<accum_type>(0);
        simd_type sum = Traits::zero();

        const size_t n = A.size();
        const T *ptr_A = A.data();
        const T *ptr_B = B.data();

        size_t i = 0;
        for (; i <= n - Traits::step; i += Traits::step) {
            simd_type va = Traits::load(ptr_A + i);
            simd_type vb = Traits::load(ptr_B + i);
            sum = Traits::mul_add(va, vb, sum);
        }

        result = Traits::horizontal_sum(sum);

        for (; i < n; ++i) {
            result += ptr_A[i] * ptr_B[i];
        }

        return result;
    }


    // min1_simd
    template <typename T>
    std::vector<T> min1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.empty() || B.empty())
            throw std::invalid_argument("Input vectors can't be empty");

        if (A.size() != B.size())
            throw std::invalid_argument("Input vectors must be of the same size.");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = min_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = min_simd_traits<T>::load(&A[i]);
            auto vec_b = min_simd_traits<T>::load(&B[i]);
            auto vec_result = min_simd_traits<T>::min(vec_a, vec_b);
            min_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i)
            result[i] = std::min(A[i], B[i]);

        return result;
    }


    // max1_simd
    template <typename T>
    std::vector<T> max1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.empty() || B.empty())
            throw std::invalid_argument("Input vectors can't be empty");

        if (A.size() != B.size())
            throw std::invalid_argument("Input vectors must be of the same size.");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = max_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = max_simd_traits<T>::load(&A[i]);
            auto vec_b = max_simd_traits<T>::load(&B[i]);
            auto vec_result = max_simd_traits<T>::max(vec_a, vec_b);
            max_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i)
            result[i] = std::max(A[i], B[i]);

        return result;
    }
}

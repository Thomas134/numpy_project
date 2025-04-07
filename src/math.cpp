#include "../include/math.hpp"
#include "../include/utils.hpp"
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


    // sqrt1_simd
    template <typename T>
    std::vector<T> sqrt1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        if (A.empty())
            throw std::invalid_argument("Input vector can't be empty");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = sqrt_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = sqrt_simd_traits<T>::load(&A[i]);
            auto vec_result = sqrt_simd_traits<T>::sqrt(vec_a);
            sqrt_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i)
            result[i] = std::sqrt(A[i]);

        return result;
    }


    // rsqrt1_simd
    template <typename T>
    std::vector<T> rsqrt1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float>);

        if (A.empty())
            throw std::invalid_argument("Input vector can't be empty");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = rsqrt_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = rsqrt_simd_traits<T>::load(&A[i]);
            auto vec_result = rsqrt_simd_traits<T>::rsqrt(vec_a);
            rsqrt_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i)
            result[i] = 1.0 / std::sqrt(A[i]);

        return result;
    }


    // round1_simd
    template <typename T>
    std::vector<T> round1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        if (A.empty())
            throw std::invalid_argument("Input vector can't be empty");
    
        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = round_simd_traits<T>::step;
    
        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = round_simd_traits<T>::load(&A[i]);
            auto vec_result = round_simd_traits<T>::round(vec_a);
            round_simd_traits<T>::store(&result[i], vec_result);
        }
    
        for (; i < A.size(); ++i)
            result[i] = std::round(A[i]);
    
        return result;
    }


    // ceil1_simd
    template <typename T>
    std::vector<T> ceil1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        if (A.empty())
            throw std::invalid_argument("Input vector can't be empty");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = ceil_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = ceil_simd_traits<T>::load(&A[i]);
            auto vec_result = ceil_simd_traits<T>::ceil(vec_a);
            ceil_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i)
            result[i] = std::ceil(A[i]);

        return result;
    }


    // floor1_simd
    template <typename T>
    std::vector<T> floor1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        if (A.empty())
            throw std::invalid_argument("Input vector can't be empty");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = floor_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = floor_simd_traits<T>::load(&A[i]);
            auto vec_result = floor_simd_traits<T>::floor(vec_a);
            floor_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i)
            result[i] = std::floor(A[i]);

        return result;
    }


    // abs1_simd
    template <typename T>
    std::vector<T> abs1_simd(const std::vector<T>& A) {
        if (A.empty())
            throw std::invalid_argument("Input vector can't be empty");

        std::vector<T> result(A.size());
        size_t i = 0;
        const size_t simd_step = abs_simd_traits<T>::step;

        for (; i <= A.size() - simd_step; i += simd_step) {
            auto vec_a = abs_simd_traits<T>::load(&A[i]);
            auto vec_result = abs_simd_traits<T>::abs(vec_a);
            abs_simd_traits<T>::store(&result[i], vec_result);
        }

        for (; i < A.size(); ++i)
            result[i] = std::abs(A[i]);

        return result;
    }


    // ================================= 2D ====================================

    // min2_simd
    template <typename T>
    std::vector<std::vector<T>> min2_simd(const std::vector<std::vector<T>>& A, 
                                          const std::vector<std::vector<T>>& B) {
        return apply_binary_op(A, B, min1_simd);
    }


    // max2_simd
    template <typename T>
    std::vector<std::vector<T>> max2_simd(const std::vector<std::vector<T>>& A, 
                                          const std::vector<std::vector<T>>& B) {
        return apply_binary_op(A, B, max1_simd);
    }


    // sqrt2_simd
    template <typename T>
    std::vector<std::vector<T>> sqrt2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, sqrt1_simd);
    }


    // rsqrt2_simd
    template <typename T>
    std::vector<std::vector<T>> rsqrt2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, rsqrt1_simd);
    }


    // round2_simd
    template <typename T>
    std::vector<std::vector<T>> round2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, round1_simd);
    }


    // ceil2_simd
    template <typename T>
    std::vector<std::vector<T>> ceil2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, ceil1_simd);
    }


    // floor2_simd
    template <typename T>
    std::vector<std::vector<T>> floor2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, floor1_simd);
    }


    // abs2_simd
    template <typename T>
    std::vector<std::vector<T>> abs2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, abs1_simd);
    }
}

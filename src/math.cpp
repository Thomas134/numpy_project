#include "../include/math.hpp"
#include "../include/utils/utils.hpp"
#include "../include/utils/simd_operators.hpp"
#include <type_traits>
#include <cmath>


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
        return apply_binary_op_simd<T, min_simd_traits<T>>(A, B, std::min);        
    }


    // max1_simd
    template <typename T>
    std::vector<T> max1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        return apply_binary_op_simd<T, min_simd_traits<T>>(A, B, std::max);        
    }


    // sqrt1_simd
    template <typename T>
    std::vector<T> sqrt1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        return apply_unary_op_simd<T, sqrt_simd_traits<T>>(A, std::sqrt);
    }


    // rsqrt1_simd
    template <typename T>
    std::vector<T> rsqrt1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float>);

        return apply_unary_op_simd<T, rsqrt_simd_traits<T>>(A, std::sqrt);
    }


    // round1_simd
    template <typename T>
    std::vector<T> round1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    
        return apply_unary_op_simd<T, round_simd_traits<T>>(A, std::round);
    }


    // ceil1_simd
    template <typename T>
    std::vector<T> ceil1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        return apply_unary_op_simd<T, ceil_simd_traits<T>>(A, std::ceil);
    }


    // floor1_simd
    template <typename T>
    std::vector<T> floor1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        return apply_unary_op_simd<T, floor_simd_traits<T>>(A, std::floor);
    }


    // abs1_simd
    template <typename T>
    std::vector<T> abs1_simd(const std::vector<T>& A) {
        return apply_unary_op_simd<T, abs_simd_traits<T>>(A, std::abs);
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

#ifndef MATH_HPP
#define MATH_HPP

#include <vector>
#include "simd_traits.cpp"
#include "xsimd_traits.cpp"
#include "utils/utils.cpp"
#include "utils/simd_operators.cpp"
#include <type_traits>
#include <cmath>

namespace internal {
    // ================================= 1D ====================================
    // min1
    template <typename T>
    std::vector<T> min1_simd(const std::vector<T>& A, const std::vector<T>& B);


    // max1
    template <typename T>
    std::vector<T> max1_simd(const std::vector<T>& A, const std::vector<T>& B);


    // sqrt1
    template <typename T>
    std::vector<T> sqrt1_simd(const std::vector<T>& A);


    // rsqrt1
    template <typename T>
    std::vector<T> rsqrt1_simd(const std::vector<T>& A);


    // round1
    template <typename T>
    std::vector<T> round1_simd(const std::vector<T>& A);


    // ceil1
    template <typename T>
    std::vector<T> ceil1_simd(const std::vector<T>& A);

    
    // floor1
    template <typename T>
    std::vector<T> floor1_simd(const std::vector<T>& A);


    // abs1
    template <typename T>
    std::vector<T> abs1_simd(const std::vector<T>& A);


    // log_1
    template <typename T>
    std::vector<T> log_1_simd(const std::vector<T>& A);


    // log2_1
    template <typename T>
    std::vector<T> log2_1_simd(const std::vector<T>& A);


    // log10_1
    template <typename T>
    std::vector<T> log10_1_simd(const std::vector<T>& A);


    // sin1
    template <typename T>
    std::vector<T> sin1_simd(const std::vector<T>& A);


    // cos1
    template <typename T>
    std::vector<T> cos1_simd(const std::vector<T>& A);


    // sincos1
    template <typename T>
    std::vector<T> sincos1_simd(const std::vector<T>& A);


    // tan1
    template <typename T>
    std::vector<T> tan1_simd(const std::vector<T>& A);


    // asin1
    template <typename T>
    std::vector<T> asin1_simd(const std::vector<T>& A);


    // acos1
    template <typename T>
    std::vector<T> acos1_simd(const std::vector<T>& A);


    // atan1
    template <typename T>
    std::vector<T> atan1_simd(const std::vector<T>& A);


    // ================================= 2D ====================================

    // min2
    template <typename T>
    std::vector<std::vector<T>> min2_simd(const std::vector<std::vector<T>>& A, 
                                          const std::vector<std::vector<T>>& B);


    // max2
    template <typename T>
    std::vector<std::vector<T>> max2_simd(const std::vector<std::vector<T>>& A, 
                                          const std::vector<std::vector<T>>& B);


    // sqrt2
    template <typename T>
    std::vector<std::vector<T>> sqrt2_simd(const std::vector<std::vector<T>>& A);


    // rsqrt2
    template <typename T>
    std::vector<std::vector<T>> rsqrt2_simd(const std::vector<std::vector<T>>& A);


    // round2
    template <typename T>
    std::vector<std::vector<T>> round2_simd(const std::vector<std::vector<T>>& A);

    
    // ceil2
    template <typename T>
    std::vector<std::vector<T>> ceil2_simd(const std::vector<std::vector<T>>& A);


    // floor2
    template <typename T>
    std::vector<std::vector<T>> floor2_simd(const std::vector<std::vector<T>>& A);


    // abs2
    template <typename T>
    std::vector<std::vector<T>> abs2_simd(const std::vector<std::vector<T>>& A);


    // log_2
    template <typename T>
    std::vector<std::vector<T>> log_2_simd(const std::vector<std::vector<T>>& A);


    // log2_2
    template <typename T>
    std::vector<std::vector<T>> log2_2_simd(const std::vector<std::vector<T>>& A);


    // log10_2
    template <typename T>
    std::vector<std::vector<T>> log10_2_simd(const std::vector<std::vector<T>>& A);


    // sin2
    template <typename T>
    std::vector<std::vector<T>> sin2_simd(const std::vector<std::vector<T>>& A);


    // cos2
    template <typename T>
    std::vector<std::vector<T>> cos2_simd(const std::vector<std::vector<T>>& A);


    // sincos2
    template <typename T>
    std::vector<std::vector<T>> sincos2_simd(const std::vector<std::vector<T>>& A);


    // tan2
    template <typename T>
    std::vector<std::vector<T>> tan2_simd(const std::vector<std::vector<T>>& A);


    // asin2
    template <typename T>
    std::vector<std::vector<T>> asin2_simd(const std::vector<std::vector<T>>& A);


    // acos2
    template <typename T>
    std::vector<std::vector<T>> acos2_simd(const std::vector<std::vector<T>>& A);


    // atan2
    template <typename T>
    std::vector<std::vector<T>> atan2_simd(const std::vector<std::vector<T>>& A);
}



namespace internal {
    // min1_simd
    template <typename T>
    std::vector<T> min1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() < 32)
            return apply_binary_op_plain(A, B, std::min);

        return apply_binary_op_simd<T, min_simd_traits<T>>(A, B, std::min);        
    }


    // max1_simd
    template <typename T>
    std::vector<T> max1_simd(const std::vector<T>& A, const std::vector<T>& B) {
        if (A.size() < 32)
            return apply_binary_op_plain(A, B, std::max);

        return apply_binary_op_simd<T, max_simd_traits<T>>(A, B, std::max);        
    }


    // sqrt1_simd
    template <typename T>
    std::vector<T> sqrt1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        if (A.size() < 32)
            return apply_unary_op_plain(A, std::sqrt);

        return apply_unary_op_simd<T, sqrt_simd_traits<T>>(A, std::sqrt);
    }


    // rsqrt1_simd
    template <typename T>
    std::vector<T> rsqrt1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float>);

        if (A.size() < 32)
            return apply_unary_op_plain(A, [](const T& element) { return 1 / std::sqrt(element); });

        return apply_unary_op_simd<T, rsqrt_simd_traits<T>>(A, std::sqrt);
    }


    // round1_simd
    template <typename T>
    std::vector<T> round1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        if (A.size() < 32)
            return apply_unary_op_plain(A, std::round);
    
        return apply_unary_op_simd<T, round_simd_traits<T>>(A, std::round);
    }


    // ceil1_simd
    template <typename T>
    std::vector<T> ceil1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        if (A.size() < 32)
            return apply_unary_op_plain(A, std::ceil);

        return apply_unary_op_simd<T, ceil_simd_traits<T>>(A, std::ceil);
    }


    // floor1_simd
    template <typename T>
    std::vector<T> floor1_simd(const std::vector<T>& A) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

        if (A.size() < 32)
            return apply_unary_op_plain(A, std::floor);

        return apply_unary_op_simd<T, floor_simd_traits<T>>(A, std::floor);
    }


    // abs1_simd
    template <typename T>
    std::vector<T> abs1_simd(const std::vector<T>& A) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, std::abs);

        return apply_unary_op_simd<T, abs_simd_traits<T>>(A, std::abs);
    }


    // log_1_simd
    template <typename T>
    std::vector<T> log_1_simd(const std::vector<T>& A) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, std::log);

        return apply_unary_op_simd<T, log_simd_traits<T>>(A, std::log);
    }


    // log2_1_simd
    template <typename T>
    std::vector<T> log2_1_simd(const std::vector<T>& A) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, std::log2);

        return apply_unary_op_simd<T, log2_simd_traits<T>>(A, std::log2);
    }


    // log10_1_simd
    template <typename T>
    std::vector<T> log10_1_simd(const std::vector<T>& A) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, std::log10);

        return apply_unary_op_simd<T, log10_simd_traits<T>>(A, std::log10);
    }


    // sin1_simd
    template <typename T>
    std::vector<T> sin1_simd(const std::vector<T>& A) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, std::sin);

        return apply_unary_op_simd<T, sin_simd_traits<T>>(A, std::sin);
    }


    // cos1_simd
    template <typename T>
    std::vector<T> cos1_simd(const std::vector<T>& A) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, std::cos);

        return apply_unary_op_simd<T, cos_simd_traits<T>>(A, std::cos);
    }


    // sincos1_simd
    

    // tan1_simd
    template <typename T>
    std::vector<T> tan1_simd(const std::vector<T>& A) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, std::tan);

        return apply_unary_op_simd<T, tan_simd_traits<T>>(A, std::tan);
    }


    // asin1_simd
    template <typename T>
    std::vector<T> asin1_simd(const std::vector<T>& A) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, std::asin);

        return apply_unary_op_simd<T, asin_simd_traits<T>>(A, std::asin);
    }


    // acos1_simd
    template <typename T>
    std::vector<T> acos1_simd(const std::vector<T>& A) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, std::acos);

        return apply_unary_op_simd<T, acos_simd_traits<T>>(A, std::acos);
    }


    // atan1_simd
    template <typename T>
    std::vector<T> atan1_simd(const std::vector<T>& A) {
        if (A.size() < 32)
            return apply_unary_op_plain(A, std::atan);

        return apply_unary_op_simd<T, atan_simd_traits<T>>(A, std::atan);
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


    // log_2_simd
    template <typename T>
    std::vector<std::vector<T>> log_2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, log_1_simd);
    }


    // log2_2_simd
    template <typename T>
    std::vector<std::vector<T>> log2_2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, log2_1_simd);
    }


    // log10_2_simd
    template <typename T>
    std::vector<std::vector<T>> log10_2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, log10_1_simd);
    }


    // sin2_simd
    template <typename T>
    std::vector<std::vector<T>> sin2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, sin1_simd);
    }


    // cos2_simd
    template <typename T>
    std::vector<std::vector<T>> cos2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, cos1_simd);
    }


    // tan2_simd
    template <typename T>
    std::vector<std::vector<T>> tan2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, tan1_simd);
    }


    // asin2_simd
    template <typename T>
    std::vector<std::vector<T>> asin2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, asin1_simd);
    }


    // acos2_simd
    template <typename T>
    std::vector<std::vector<T>> acos2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, acos1_simd);
    }


    // atan2_simd
    template <typename T>
    std::vector<std::vector<T>> atan2_simd(const std::vector<std::vector<T>>& A) {
        return apply_unary_op(A, atan1_simd);
    }
}


#endif

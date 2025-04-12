#ifndef MATH_HPP
#define MATH_HPP

#include <vector>

namespace internal {
    // ================================= 1D ====================================

    // inner_product_simd
    // template <typename T>
    // typename inner_product_simd_traits<T>::accum_type inner_product_simd(const std::vector<T>& A, const std::vector<T>& B);


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

#endif

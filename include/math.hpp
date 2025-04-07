#ifndef MATH_HPP
#define MATH_HPP

#include "simd_traits.hpp"
#include <vector>

namespace internal {
    // ================================= 1D ====================================

    // inner_product_simd
    template <typename T>
    typename inner_product_simd_traits<T>::accum_type inner_product_simd(const std::vector<T>& A, const std::vector<T>& B);


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
}

#endif

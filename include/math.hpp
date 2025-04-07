#ifndef MATH_HPP
#define MATH_HPP

#include "simd_traits.hpp"
#include <vector>

namespace internal {
    // inner_product_simd
    template <typename T>
    typename inner_product_simd_traits<T>::accum_type inner_product_simd(const std::vector<T>& A, const std::vector<T>& B);


    // min1
    template <typename T>
    std::vector<T> min1_simd(const std::vector<T>& A, const std::vector<T>& B);


    // max1
    template <typename T>
    std::vector<T> max1_simd(const std::vector<T>& A, const std::vector<T>& B);
}

#endif

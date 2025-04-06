#ifndef LOGICAL_HPP
#define LOGICAL_HPP

#include <vector>

namespace internal {
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
}

#endif

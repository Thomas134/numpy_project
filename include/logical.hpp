#ifndef LOGICAL_HPP
#define LOGICAL_HPP

#include <vector>

namespace internal {
    // =========================== 1D ======================================

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


    // =========================== 2D ======================================
    
    // and2
    template <typename T>
    std::vector<std::vector<T>> and2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // or2
    template <typename T>
    std::vector<std::vector<T>> or2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // xor2
    template <typename T>
    std::vector<std::vector<T>> xor2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // andnot2
    template <typename T>
    std::vector<std::vector<T>> andnot2_simd(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);
}

#endif

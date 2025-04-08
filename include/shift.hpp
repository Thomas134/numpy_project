#ifndef SHIFT_HPP
#define SHIFT_HPP

#include <vector>

namespace internal {
    // ========================== 1D =============================

    // slli1
    template <typename T>
    std::vector<T> slli1_simd(const std::vector<T>& A, const int imm);


    // srli1
    template <typename T>
    std::vector<T> srli1_simd(const std::vector<T>& A, const int imm);


    // ========================== 1D =============================

    // slli2
    template <typename T>
    std::vector<T> slli2_simd(const std::vector<std::vector<T>>& A, const int imm);

    
    // srli2
    template <typename T>
    std::vector<T> srli2_simd(const std::vector<std::vector<T>>& A, const int imm);
}

#endif

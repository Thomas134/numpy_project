#ifndef SHIFT_HPP
#define SHIFT_HPP

#include <vector>

namespace internal {
    // slli1
    template <typename T>
    std::vector<T> slli1_simd(const std::vector<T>& A, const int imm);
}

#endif

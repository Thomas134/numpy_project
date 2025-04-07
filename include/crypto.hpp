#ifndef CRYPTO_HPP
#define CRYPTO_HPP

#include <vector>

namespace internal {
    // sm4rnds4_1d
    template <typename T>
    std::vector<T> sm4rnds4_1_simd(const std::vector<T>& A, const std::vector<T>& B);


    // sm4key4_1d
    template <typename T>
    std::vector<T> sm4key4_1_simd(const std::vector<T>& A, const std::vector<T>& B);
}

#endif

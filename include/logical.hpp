#ifndef LOGICAL
#define LOGICAL

#include <vector>

namespace internal {
    // and1
    template <typename T>
    std::vector<T> and1_simd(const std::vector<T>& A, const std::vector<T>& B);
}

#endif

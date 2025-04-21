#ifndef PARALLEL_FOR_HPP
#define PARALLEL_FOR_HPP

#include <vector>
#include <omp.h>

namespace internal {
    // apply1
    template <typename T, typename Func>
    void apply1(std::vector<T>& A, Func func);


    // apply2
    template <typename T, typename Func>
    void apply2(std::vector<std::vector<T>>& A, Func func);
}


namespace internal {
    // apply1
    template <typename T, typename Func>
    void apply1(std::vector<T>& A, Func func) {
        #pragma omp parallel for
        for (size_t i = 0; i < A.size(); ++i) {
            A[i] = func(A[i]);
        }
    }


    // apply2
    template <typename T, typename Func>
    void apply2(std::vector<std::vector<T>>& A, Func func) {
        #pragma omp parallel for
        for (size_t i = 0; i < A.size(); ++i) {
            for (size_t j = 0; j < A[i].size(); ++j) {
                A[i][j] = func(A[i][j]);
            }
        }
    }
}

#endif

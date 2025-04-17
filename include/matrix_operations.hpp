#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include <vector>

namespace internal {
    // dot
    template <typename T>
    std::vector<std::vector<T>> dot(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // transpose
    template <typename T>
    std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& mat);


    // add1
    template <typename T>
    std::vector<T> add1(const std::vector<T>& A, const std::vector<T>& B);


    // add2
    template <typename T>
    std::vector<std::vector<T>> add2(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // subtract1
    template <typename T>
    std::vector<T> subtract1(const std::vector<T>& A, const std::vector<T>& B);


    // subtract2
    template <typename T>
    std::vector<std::vector<T>> subtract2(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);
}

#endif

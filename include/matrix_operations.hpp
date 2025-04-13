#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include <vector>

namespace internal {
    // dot
    template <typename T>
    std::vector<std::vector<T>> dot(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // transpose
    template <typename T>
    std::vector<std::vector<T>> transpose(std::vector<std::vector<T>> mat);

    template <>
    std::vector<std::vector<float>> transpose(std::vector<std::vector<float>> mat);

    template <>
    std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> mat);

    template <>
    std::vector<std::vector<char>> transpose(std::vector<std::vector<char>> mat);


    // add1
    template <typename T>
    std::vector<T> add1(const std::vector<T>& A, const std::vector<T>& B);


    // add2
    template <typename T>
    std::vector<std::vector<T>> add2(std::vector<std::vector<T>> A, std::vector<std::vector<T>> B);

    template <>
    std::vector<std::vector<float>> add2(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B);

    template <>
    std::vector<std::vector<double>> add2(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);


    // subtract1
    template <typename T>
    std::vector<T> subtract1(const std::vector<T>& A, const std::vector<T>& B);


    // subtract2
    template <typename T>
    std::vector<std::vector<T>> subtract2(std::vector<std::vector<T>> A, std::vector<std::vector<T>> B);
}

#endif

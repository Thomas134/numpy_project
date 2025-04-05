#ifndef MATRIX_OPERATIONS
#define MATRIX_OPERATIONS

#include <vector>

namespace internal {
    // dot
    template <typename T>
    std::vector<std::vector<T>> dot(std::vector<std::vector<T>> A, std::vector<std::vector<T>> B);

    template <>
    std::vector<std::vector<float>> dot(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B);

    template <>
    std::vector<std::vector<double>> dot(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);


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
    std::vector<T> add1(std::vector<T> A, std::vector<T> B);

    template <>
    std::vector<float> add1(std::vector<float> A, std::vector<float> B);

    template <>
    std::vector<double> add1(std::vector<double> A, std::vector<double> B);


    // add2
    template <typename T>
    std::vector<std::vector<T>> add2(std::vector<std::vector<T>> A, std::vector<std::vector<T>> B);

    template <>
    std::vector<std::vector<float>> add2(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B);

    template <>
    std::vector<std::vector<double>> add2(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);


    // subtract1
    template <typename T>
    std::vector<T> subtract1(std::vector<T> A, std::vector<T> B);

    template <>
    std::vector<float> subtract1(std::vector<float> A, std::vector<float> B);

    template <>
    std::vector<double> subtract1(std::vector<double> A, std::vector<double> B);


    // subtract2
    template <typename T>
    std::vector<std::vector<T>> subtract2(std::vector<std::vector<T>> A, std::vector<std::vector<T>> B);

    template <>
    std::vector<std::vector<float>> subtract2(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B);

    template <>
    std::vector<std::vector<double>> subtract2(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B);

}

#endif

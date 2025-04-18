// ndarray.hpp
#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <vector>
#include <array>
#include <cstdint>
#include <iostream>

#define NDARRAY_UNARY_FUNC(func_name, simd_func_1d, simd_func_2d) \
template <typename T> \
ndarray<T> ndarray<T>::func_name() { \
    if (__shape.size() == 1) { \
        std::vector<T> result = simd_func_1d(__data); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < result.size(); ++i) \
            result_ndarray.__data[i] = result[i]; \
        return result_ndarray; \
    } else if (__shape.size() == 2) { \
        std::vector<std::vector<T>> A_2d(__shape[0], std::vector<T>(__shape[1])); \
        for (size_t i = 0; i < __shape[0]; ++i) \
            for (size_t j = 0; j < __shape[1]; ++j) \
                A_2d[i][j] = __data[calculate_offset(i, j)]; \
        std::vector<std::vector<T>> result_2d = simd_func_2d(A_2d); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < __shape[0]; ++i) \
            for (size_t j = 0; j < __shape[1]; ++j) \
                result_ndarray.__data[calculate_offset(i, j)] = result_2d[i][j]; \
        return result_ndarray; \
    } \
    throw std::invalid_argument("Unsupported array dimension."); \
}

#define NDARRAY_BINARY_FUNC(func_name, simd_1d_func, simd_2d_func) \
template <typename T> \
ndarray<T> ndarray<T>::func_name(const ndarray<T>& other) { \
    if (__shape != other.__shape) \
        throw std::invalid_argument("Shapes of the two ndarrays do not match."); \
    if (__shape.size() == 1) { \
        std::vector<T> result = simd_1d_func(__data, other.__data); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < result.size(); ++i) \
            result_ndarray.__data[i] = result[i]; \
        return result_ndarray; \
    } else if (__shape.size() == 2) { \
        std::vector<std::vector<T>> A_2d(__shape[0], std::vector<T>(__shape[1])); \
        std::vector<std::vector<T>> B_2d(__shape[0], std::vector<T>(__shape[1])); \
        for (size_t i = 0; i < __shape[0]; ++i) { \
            for (size_t j = 0; j < __shape[1]; ++j) { \
                A_2d[i][j] = __data[calculate_offset(i, j)]; \
                B_2d[i][j] = other.__data[other.calculate_offset(i, j)]; \
            } \
        } \
        std::vector<std::vector<T>> result_2d = simd_2d_func(A_2d, B_2d); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < __shape[0]; ++i) { \
            for (size_t j = 0; j < __shape[1]; ++j) { \
                result_ndarray.__data[calculate_offset(i, j)] = result_2d[i][j]; \
            } \
        } \
        return result_ndarray; \
    } \
    throw std::invalid_argument("Unsupported array dimension."); \
}

#define NDARRAY_APPLY_FUNC(func_name, apply_1d, apply_2d) \
template <typename T> \
template <typename Func> \
ndarray<T> ndarray<T>::func_name(Func func) { \
    if (__shape.size() == 1) { \
        std::vector<T> result = __data; \
        apply_1d(result, func); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < result.size(); ++i) { \
            result_ndarray.__data[i] = result[i]; \
        } \
        return result_ndarray; \
    } else if (__shape.size() == 2) { \
        std::vector<std::vector<T>> A_2d(__shape[0], std::vector<T>(__shape[1])); \
        for (size_t i = 0; i < __shape[0]; ++i) { \
            for (size_t j = 0; j < __shape[1]; ++j) { \
                A_2d[i][j] = __data[calculate_offset(i, j)]; \
            } \
        } \
        apply_2d(A_2d, func); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < __shape[0]; ++i) { \
            for (size_t j = 0; j < __shape[1]; ++j) { \
                result_ndarray.__data[calculate_offset(i, j)] = A_2d[i][j]; \
            } \
        } \
        return result_ndarray; \
    } \
    throw std::invalid_argument("Unsupported array dimension."); \
}

#define NDARRAY_SHIFT_FUNC(func_name, simd_func_1d, simd_func_2d) \
template<typename T> \
ndarray<T> ndarray<T>::func_name(const int imm) { \
    if (__shape.size() == 1) { \
        std::vector<T> result = simd_func_1d(__data, imm); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < result.size(); ++i) { \
            result_ndarray.__data[i] = result[i]; \
        } \
        return result_ndarray; \
    } else if (__shape.size() == 2) { \
        std::vector<std::vector<T>> A_2d(__shape[0], std::vector<T>(__shape[1])); \
        for (size_t i = 0; i < __shape[0]; ++i) { \
            for (size_t j = 0; j < __shape[1]; ++j) { \
                A_2d[i][j] = __data[calculate_offset(i, j)]; \
            } \
        } \
        std::vector<std::vector<T>> result_2d = simd_func_2d(A_2d, imm); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < __shape[0]; ++i) { \
            for (size_t j = 0; j < __shape[1]; ++j) { \
                result_ndarray.__data[calculate_offset(i, j)] = result_2d[i][j]; \
            } \
        } \
        return result_ndarray; \
    } \
    throw std::invalid_argument("Unsupported array dimension."); \
}

#define NDARRAY_SORT_FUNC(func_name, sort_1d_func, sort_2d_func) \
template<typename T> \
template <typename Compare> \
ndarray<T> ndarray<T>::func_name(Compare comp = std::less<T>{}) { \
    if (__shape.size() == 1) { \
        std::vector<T> result = __data; \
        sort_1d_func(result, comp); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < result.size(); ++i) { \
            result_ndarray.__data[i] = result[i]; \
        } \
        return result_ndarray; \
    } else if (__shape.size() == 2) { \
        std::vector<std::vector<T>> A_2d(__shape[0], std::vector<T>(__shape[1])); \
        for (size_t i = 0; i < __shape[0]; ++i) { \
            for (size_t j = 0; j < __shape[1]; ++j) { \
                A_2d[i][j] = __data[calculate_offset(i, j)]; \
            } \
        } \
        sort_2d_func(A_2d, comp); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < __shape[0]; ++i) { \
            for (size_t j = 0; j < __shape[1]; ++j) { \
                result_ndarray.__data[calculate_offset(i, j)] = A_2d[i][j]; \
            } \
        } \
        return result_ndarray; \
    } \
    throw std::invalid_argument("Unsupported array dimension."); \
}

#define NDARRAY_CRYPTO_FUNC(func_name, crypto_func) \
template<typename T> \
ndarray<T> ndarray<T>::func_name(const ndarray<T>& other) { \
    if (__shape != other.__shape) { \
        throw std::invalid_argument("Shapes of the two ndarrays do not match."); \
    } \
    if (__shape.size() == 1) { \
        std::vector<T> result = crypto_func(__data, other.__data); \
        ndarray<T> result_ndarray(__shape); \
        for (size_t i = 0; i < result.size(); ++i) { \
            result_ndarray.__data[i] = result[i]; \
        } \
        return result_ndarray; \
    } \
    throw std::invalid_argument("Unsupported array dimension."); \
}

template<typename T>
class ndarray {
private:
    std::vector<T> __data;
    std::vector<size_t> __shape;
    std::vector<size_t> __strides;
    size_t __size;

    void compute_strides();

    size_t calculate_offset(size_t row, size_t col) const noexcept;

public:
    ndarray(const std::vector<size_t>& shape);

    const char *dtype() const noexcept;

    size_t itemsize() const noexcept;

    size_t ndim() const noexcept;
    
    size_t size() const noexcept;

    std::vector<size_t> shape() const noexcept;


public:
    std::vector<uint8_t> all(int axis) const;

    // float mean() const;

    // double mean() const;

    // float mean(int axis) const;


    // logical function
    ndarray<T> logical_and(const ndarray<T>& other);

    ndarray<T> logical_or(const ndarray<T>& other);

    ndarray<T> logical_xor(const ndarray<T>& other);

    ndarray<T> logical_andnot(const ndarray<T>& other);


    // math function
    ndarray<T> min(const ndarray<T>& other);

    ndarray<T> max(const ndarray<T>& other);

    ndarray<T> sqrt();

    ndarray<T> rsqrt();

    ndarray<T> round();

    ndarray<T> ceil();

    ndarray<T> floor();

    ndarray<T> abs();

    ndarray<T> log();

    ndarray<T> log2();

    ndarray<T> log10();

    ndarray<T> sin();

    ndarray<T> cos();

    ndarray<T> sincos();

    ndarray<T> tan();

    ndarray<T> asin();

    ndarray<T> acos();

    ndarray<T> atan();


    // parallel function
    template <typename Func>
    ndarray<T> apply(Func func);


    // sort function
    template <typename Compare>
    ndarray<T> sort(Compare comp);


    // shift function
    ndarray<T> slli(const int imm);

    ndarray<T> srli(const int imm);


    // crypto function
    ndarray<T> sm4rnds4(const ndarray<T>& other);

    ndarray<T> sm4key4(const ndarray<T>& other);


    // matrix operations
    ndarray<T> add(const ndarray<T>& other);

    ndarray<T> sub(const ndarray<T>& other);

    ndarray<T> dot(const ndarray<T>& other);

    ndarray<T> transpose();
    

    // access element
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;

    friend std::ostream& operator<<(std::ostream& os, const ndarray<T>& arr);
};

#endif // NDARRAY_HPP    
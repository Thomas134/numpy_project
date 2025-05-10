// ndarray.hpp
#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <vector>
#include <array>
#include <cstdint>
#include <iostream>
#include <optional>

#include <stdexcept>
#include <cmath>
#include <numeric>
#include <omp.h>

#include "dtype_trait.cpp"

#include "../logical.cpp"
#include "../math.cpp"
#include "../parallel_for.cpp"
#include "../shift.cpp"
#include "../sort.cpp"
#include "../matrix_operations.cpp"

#define NDARRAY_UNARY_FUNC(func_name, simd_func_1d) \
template <typename T> \
ndarray<T> ndarray<T>::func_name() { \
    if (__shape.size() != 1 && __shape.size() != 2) \
        throw std::invalid_argument("Unsupported array dimension."); \
    std::vector<T> result = simd_func_1d(__data); \
    ndarray<T> result_ndarray(__shape); \
    for (size_t i = 0; i < result.size(); ++i) \
        result_ndarray.__data[i] = result[i]; \
    return result_ndarray; \
}

#define NDARRAY_BINARY_FUNC(func_name, simd_1d_func) \
template <typename T> \
ndarray<T> ndarray<T>::func_name(const ndarray<T>& other) { \
    if (__shape != other.__shape) \
        throw std::invalid_argument("Shapes of the two ndarrays do not match."); \
    if (__shape.size() != 1 && __shape.size() != 2) \
        throw std::invalid_argument("Unsupported array dimension."); \
    std::vector<T> result = simd_1d_func(__data, other.__data); \
    ndarray<T> result_ndarray(__shape); \
    for (size_t i = 0; i < result.size(); ++i) \
        result_ndarray.__data[i] = result[i]; \
    return result_ndarray; \
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

#define NDARRAY_SHIFT_FUNC(func_name, simd_func_1d) \
template <typename T> \
ndarray<T> ndarray<T>::func_name(const int imm) { \
    if (__shape.size() != 1 && __shape.size() != 2) \
        throw std::invalid_argument("Unsupported array dimension."); \
    std::vector<T> result = simd_func_1d(__data, imm); \
    ndarray<T> result_ndarray(__shape); \
    for (size_t i = 0; i < result.size(); ++i) { \
        result_ndarray.__data[i] = result[i]; \
    } \
    return result_ndarray; \
}

#define NDARRAY_SORT_FUNC(func_name, sort_1d_func, sort_2d_func) \
template <typename T> \
template <typename Compare> \
ndarray<T> ndarray<T>::func_name(Compare comp) { \
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

#define NDARRAY_ARITH_FUNC(func_name, op_1d) \
template <typename T> \
ndarray<T> ndarray<T>::func_name(const ndarray<T>& other) { \
    if (__shape != other.__shape) \
        throw std::invalid_argument("Shapes of the two ndarrays do not match."); \
    if (__shape.size() != 1 && __shape.size() != 2) \
        throw std::invalid_argument("Unsupported array dimension."); \
    std::vector<T> result = op_1d(__data, other.__data); \
    ndarray<T> result_ndarray(__shape); \
    for (size_t i = 0; i < result.size(); ++i) \
        result_ndarray.__data[i] = result[i]; \
    return result_ndarray; \
}

template <typename T>
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

    void assign(const std::vector<T>& data);

    void assign(const std::vector<std::vector<T>>& data);

    const char *dtype() const noexcept;

    size_t itemsize() const noexcept;

    size_t ndim() const noexcept;
    
    size_t size() const noexcept;

    std::vector<size_t> shape() const noexcept;

    std::vector<T> data() const noexcept;


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
    ndarray<T> sort(Compare comp = std::less<T>{});


    // shift function
    ndarray<T> slli(const int imm);

    ndarray<T> srli(const int imm);


    // matrix operations
    ndarray<T> add(const ndarray<T>& other);

    ndarray<T> sub(const ndarray<T>& other);

    ndarray<T> dot(const ndarray<T>& other);

    ndarray<T> transpose();
    

    // access element
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;

    friend std::ostream& operator<<(std::ostream& os, const ndarray<T>& arr) {
        os << "[";
        if (arr.__shape.size() == 1) {
            for (size_t i = 0; i < arr.__size; ++i) {
                if (i > 0) 
                    os << ", ";
                    
                os << arr.__data[i];
            }
        } else if (arr.__shape.size() == 2) {
            for (size_t i = 0; i < arr.__shape[0]; ++i) {
                if (i > 0)
                    os << ",\n ";
                
                os << "[";
                for (size_t j = 0; j < arr.__shape[1]; ++j) {
                    if (j > 0)
                        os << ", ";
                    
                    os << arr.__data[arr.calculate_offset(i, j)];
                }
                os << "]";
            }
        }
        os << "]";

        return os;
    }
};

template <typename T>
uint8_t bool_operation_with_row(const std::vector<T>& vec) {
    for (int i = 0; i < static_cast<int>(vec.size()); ++i)
        if (!vec[i])
            return 0;

    return 1;
}

template <typename T>
uint8_t bool_operation_with_col(const std::vector<std::vector<T>>& vec) {
    for (int i = 0; i < static_cast<int>(vec[0].size()); ++i)
        for (int j = 0; j < static_cast<int>(vec.size()); ++j)
            if (!vec[j][i])
                return 0;

    return 1;
}

template <typename T>
size_t ndarray<T>::calculate_offset(size_t row, size_t col) const noexcept {
    size_t offset = row * __strides[0] + col * __strides[1];
        
    return offset;
}


template <typename T>
ndarray<T>::ndarray(const std::vector<size_t>& shape) : __shape(shape) {
    if (shape.empty())
        throw std::invalid_argument("Shape cannot be empty");

    __size = std::accumulate(
        shape.begin(), shape.end(), 
        1, std::multiplies<size_t>()
    );

    __data.resize(__size);
    compute_strides();
}

template <typename T>
void ndarray<T>::assign(const std::vector<T>& data) {
    if (__shape.size() != 1)
        throw std::invalid_argument("The array is not 1D.");

    if (data.size() != __size)
        throw std::invalid_argument("Data size does not match the array size.");

    __data = data;
}

template <typename T>
void ndarray<T>::assign(const std::vector<std::vector<T>>& data) {
    if (__shape.size() != 2)
        throw std::invalid_argument("The array is not 2D.");

    if (data.size() != __shape[0])
        throw std::invalid_argument("Number of rows does not match the array shape.");
    
    for (size_t i = 0; i < data.size(); ++i)
        if (data[i].size() != __shape[1])
            throw std::invalid_argument("Number of columns does not match the array shape.");

    __data.clear();

    for (const auto& row : data)
        __data.insert(__data.end(), row.begin(), row.end());
}

template <typename T>
void ndarray<T>::compute_strides() {
    __strides.resize(__shape.size());

    if (!__shape.empty()) {
        __strides.back() = 1;
        for (int i = static_cast<int>(__shape.size()) - 2; i >= 0; --i) {
            __strides[i] = __strides[i + 1] * __shape[i + 1];
        }
    }
}

template <typename T>
const char *ndarray<T>::dtype() const noexcept {
    return dtype_traits<T>::name;
}

template <typename T>
size_t ndarray<T>::itemsize() const noexcept {
    return dtype_traits<T>::size;
}

template <typename T>
size_t ndarray<T>::ndim() const noexcept {
    return __shape.size();
}

template <typename T>
size_t ndarray<T>::size() const noexcept {
    return __size;
}

template <typename T>
std::vector<size_t> ndarray<T>::shape() const noexcept {
    return __shape;
}

template <typename T>
std::vector<T> ndarray<T>::data() const noexcept {
    return __data;
}

template <typename T>
std::vector<uint8_t> ndarray<T>::all(int axis) const {
    if (axis < 0 || axis > 1)
        throw std::invalid_argument("Axis should be 0 or 1");

    if (axis == 0 && ndim() == 1)
        throw std::invalid_argument("A one-dimensional array cannot be traversed column-wise");

    std::vector<uint8_t> res;

    if (ndim() == 1) {
        res.push_back(bool_operation_with_row(__data));
    } else {
        size_t row = __shape[0];
        size_t col = __shape[1];
        
        if (axis == 0) {
            for (int i = 0; i < static_cast<int>(col); ++i) {
                bool false_flag = false;
                for (int j = 0; j < static_cast<int>(row); ++j) {
                    size_t offset = calculate_offset(j, i);

                    if (!__data[offset]) {
                        false_flag = true;
                        res.push_back(0);
                        break;
                    }
                }

                if (!false_flag)
                    res.push_back(1);
            }
        } else {
            for (int i = 0; i < static_cast<int>(row); ++i) {
                bool false_flag = false;
                for (int j = 0; j < static_cast<int>(col); ++j) {
                    size_t offset = calculate_offset(i, j);

                    if (!__data[offset]) {
                        false_flag = true;
                        res.push_back(0);
                        break;
                    }
                }

                if (!false_flag)
                    res.push_back(1);
            }
        }
    }

    return res;
}


// logical functions
NDARRAY_BINARY_FUNC(logical_and, internal::and1_simd);

NDARRAY_BINARY_FUNC(logical_or, internal::or1_simd);

NDARRAY_BINARY_FUNC(logical_xor, internal::xor1_simd);

NDARRAY_BINARY_FUNC(logical_andnot, internal::andnot1_simd);


// math functions
NDARRAY_BINARY_FUNC(min, internal::min1_simd);

NDARRAY_BINARY_FUNC(max, internal::max1_simd);
    
NDARRAY_UNARY_FUNC(sqrt, internal::sqrt1_simd);

NDARRAY_UNARY_FUNC(rsqrt, internal::rsqrt1_simd);

NDARRAY_UNARY_FUNC(round, internal::round1_simd);

NDARRAY_UNARY_FUNC(ceil, internal::ceil1_simd);

NDARRAY_UNARY_FUNC(floor, internal::floor1_simd);

NDARRAY_UNARY_FUNC(abs, internal::abs1_simd);

NDARRAY_UNARY_FUNC(log, internal::log_1_simd);

NDARRAY_UNARY_FUNC(log2, internal::log2_1_simd);

NDARRAY_UNARY_FUNC(log10, internal::log10_1_simd);

NDARRAY_UNARY_FUNC(sin, internal::sin1_simd);

NDARRAY_UNARY_FUNC(cos, internal::cos1_simd);

NDARRAY_UNARY_FUNC(sincos, internal::sincos1_simd);

NDARRAY_UNARY_FUNC(tan, internal::tan1_simd);

NDARRAY_UNARY_FUNC(asin, internal::asin1_simd);

NDARRAY_UNARY_FUNC(acos, internal::acos1_simd);

NDARRAY_UNARY_FUNC(atan, internal::atan1_simd);


// parallel functions
NDARRAY_APPLY_FUNC(apply, internal::apply1, internal::apply2);


// sort functions
NDARRAY_SORT_FUNC(sort, internal::sort1, internal::sort2);


// shift functions
NDARRAY_SHIFT_FUNC(slli, internal::slli1_simd);

NDARRAY_SHIFT_FUNC(srli, internal::srli1_simd);


// matrix operations
template <typename T>
ndarray<T> ndarray<T>::dot(const ndarray<T>& other) {
    if (__shape.size() != 2 || other.__shape.size() != 2)
        throw std::invalid_argument("Only 2D arrays are supported for dot operation.");
    
    const size_t M = __shape[0];
    const size_t K_A = __shape[1];
    const size_t K_B = other.__shape[0];
    const size_t N = other.__shape[1];

    if (K_A != K_B) {
        throw std::invalid_argument("Matrix dimension mismatch");
    }

    std::vector<std::vector<T>> A_2d(__shape[0], std::vector<T>(__shape[1]));
    std::vector<std::vector<T>> B_2d(other.__shape[0], std::vector<T>(other.__shape[1]));

    for (size_t i = 0; i < __shape[0]; ++i) {
        for (size_t j = 0; j < __shape[1]; ++j) {
            A_2d[i][j] = __data[calculate_offset(i, j)];
        }
    }
    
    for (size_t i = 0; i < other.__shape[0]; ++i) {
        for (size_t j = 0; j < other.__shape[1]; ++j) {
            B_2d[i][j] = other.__data[other.calculate_offset(i, j)];
        }
    }

    std::vector<std::vector<T>> result_2d = internal::dot(A_2d, B_2d);

    std::vector<size_t> result_shape = {M, N};
    ndarray<T> result_ndarray(result_shape);

    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < N; ++j)
            result_ndarray.__data[result_ndarray.calculate_offset(i, j)] = result_2d[i][j];
    
    return result_ndarray;
}

NDARRAY_ARITH_FUNC(add, internal::add1)

NDARRAY_ARITH_FUNC(sub, internal::subtract1)

template <typename T>
ndarray<T> ndarray<T>::transpose() {
    if (__shape.size() != 2)
        throw std::invalid_argument("Only 2D arrays can be transposed.");

    std::vector<std::vector<T>> mat_2d(__shape[0], std::vector<T>(__shape[1]));
    
    for (size_t i = 0; i < __shape[0]; ++i)
        for (size_t j = 0; j < __shape[1]; ++j)
            mat_2d[i][j] = __data[calculate_offset(i, j)];

    std::vector<std::vector<T>> result_2d = internal::transpose(mat_2d);
    std::vector<size_t> result_shape = {__shape[1], __shape[0]};
    ndarray<T> result_ndarray(result_shape);
    
    for (size_t i = 0; i < result_shape[0]; ++i)
        for (size_t j = 0; j < result_shape[1]; ++j)
            result_ndarray.__data[result_ndarray.calculate_offset(i, j)] = result_2d[i][j];

    return result_ndarray;
}

template <typename T>
T& ndarray<T>::operator()(const std::vector<size_t>& indices) {
    if (indices.size() != __shape.size())
        throw std::out_of_range("Index dimensions do not match array dimensions.");

    if (__shape.size() == 1) 
        return __data[indices[0]];
    else if (__shape.size() == 2)
        return __data[calculate_offset(indices[0], indices[1])];
    
    throw std::out_of_range("Invalid index.");
}

template <typename T>
const T& ndarray<T>::operator()(const std::vector<size_t>& indices) const {
    if (indices.size() != __shape.size())
        throw std::out_of_range("Index dimensions do not match array dimensions.");

    if (__shape.size() == 1)
        return __data[indices[0]];
    else if (__shape.size() == 2)
        return __data[calculate_offset(indices[0], indices[1])];

    throw std::out_of_range("Invalid index.");
}


#endif // NDARRAY_HPP    

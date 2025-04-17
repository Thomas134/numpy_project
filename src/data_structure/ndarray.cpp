#include <stdexcept>
#include <cmath>

#include "../../include/data_structure/ndarray.hpp"
#include "../../include/data_structure/dtype_trait.hpp"

#include "../../include/logical.hpp"
#include "../../include/math.hpp"
#include "../../include/parallel_for.hpp"
#include "../../include/shift.hpp"
#include "../../include/sort.hpp"
#include "../../include/crypto.hpp"

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
template<typename T>
ndarray<T> ndarray<T>::logical_and(const ndarray<T>& other) {
    if (__shape != other.__shape)
        throw std::invalid_argument("Shapes of the two ndarrays do not match.");

        if (__shape.size() == 1) {
            std::vector<T> result = and1_simd(__data, other.__data);
            ndarray<T> result_ndarray(__shape);
            for (size_t i = 0; i < result.size(); ++i) {
                result_ndarray.__data[i] = result[i];
        }
            return result_ndarray;
    } else if (__shape.size() == 2) {
            std::vector<std::vector<T>> A_2d(__shape[0], std::vector<T>(__shape[1]));
            std::vector<std::vector<T>> B_2d(__shape[0], std::vector<T>(__shape[1]));
            for (size_t i = 0; i < __shape[0]; ++i) {
                for (size_t j = 0; j < __shape[1]; ++j) {
                    A_2d[i][j] = __data[calculate_offset(i, j)];
                    B_2d[i][j] = other.__data[other.calculate_offset(i, j)];
                }
            }
            std::vector<std::vector<T>> result_2d = and2_simd(A_2d, B_2d);
            ndarray<T> result_ndarray(__shape);
            for (size_t i = 0; i < __shape[0]; ++i) {
                for (size_t j = 0; j < __shape[1]; ++j) {
                    result_ndarray.__data[calculate_offset(i, j)] = result_2d[i][j];
                }
            }
            return result_ndarray;
    }

    throw std::invalid_argument("Unsupported array dimension.");
}

template<typename T>
ndarray<T> ndarray<T>::logical_or(const ndarray<T>& other) {
    if (__shape != other.__shape)
        throw std::invalid_argument("Shapes of the two ndarrays do not match.");

        if (__shape.size() == 1) {
            std::vector<T> result = or1_simd(__data, other.__data);
            ndarray<T> result_ndarray(__shape);
            for (size_t i = 0; i < result.size(); ++i) {
                result_ndarray.__data[i] = result[i];
        }
            return result_ndarray;
    } else if (__shape.size() == 2) {
            std::vector<std::vector<T>> A_2d(__shape[0], std::vector<T>(__shape[1]));
            std::vector<std::vector<T>> B_2d(__shape[0], std::vector<T>(__shape[1]));
            for (size_t i = 0; i < __shape[0]; ++i) {
                for (size_t j = 0; j < __shape[1]; ++j) {
                    A_2d[i][j] = __data[calculate_offset(i, j)];
                    B_2d[i][j] = other.__data[other.calculate_offset(i, j)];
                }
            }
            std::vector<std::vector<T>> result_2d = or2_simd(A_2d, B_2d);
            ndarray<T> result_ndarray(__shape);
            for (size_t i = 0; i < __shape[0]; ++i) {
                for (size_t j = 0; j < __shape[1]; ++j) {
                    result_ndarray.__data[calculate_offset(i, j)] = result_2d[i][j];
                }
            }
            return result_ndarray;
    }
    
    throw std::invalid_argument("Unsupported array dimension.");
}

template<typename T>
ndarray<T> ndarray<T>::logical_xor(const ndarray<T>& other) {
    if (__shape != other.__shape)
        throw std::invalid_argument("Shapes of the two ndarrays do not match.");

        if (__shape.size() == 1) {
            std::vector<T> result = xor1_simd(__data, other.__data);
            ndarray<T> result_ndarray(__shape);
            for (size_t i = 0; i < result.size(); ++i) {
                result_ndarray.__data[i] = result[i];
        }
            return result_ndarray;
    } else if (__shape.size() == 2) {
            std::vector<std::vector<T>> A_2d(__shape[0], std::vector<T>(__shape[1]));
            std::vector<std::vector<T>> B_2d(__shape[0], std::vector<T>(__shape[1]));
            for (size_t i = 0; i < __shape[0]; ++i) {
                for (size_t j = 0; j < __shape[1]; ++j) {
                    A_2d[i][j] = __data[calculate_offset(i, j)];
                    B_2d[i][j] = other.__data[other.calculate_offset(i, j)];
                }
            }
            std::vector<std::vector<T>> result_2d = xor2_simd(A_2d, B_2d);
            ndarray<T> result_ndarray(__shape);
            for (size_t i = 0; i < __shape[0]; ++i) {
                for (size_t j = 0; j < __shape[1]; ++j) {
                    result_ndarray.__data[calculate_offset(i, j)] = result_2d[i][j];
                }
            }
            return result_ndarray;
    }
    
    throw std::invalid_argument("Unsupported array dimension.");
}

template<typename T>
ndarray<T> ndarray<T>::logical_andnot(const ndarray<T>& other) {
    if (__shape != other.__shape)
        throw std::invalid_argument("Shapes of the two ndarrays do not match.");

        if (__shape.size() == 1) {
            std::vector<T> result = andnot1_simd(__data, other.__data);
            ndarray<T> result_ndarray(__shape);
            for (size_t i = 0; i < result.size(); ++i) {
                result_ndarray.__data[i] = result[i];
        }
            return result_ndarray;
    } else if (__shape.size() == 2) {
            std::vector<std::vector<T>> A_2d(__shape[0], std::vector<T>(__shape[1]));
            std::vector<std::vector<T>> B_2d(__shape[0], std::vector<T>(__shape[1]));
            for (size_t i = 0; i < __shape[0]; ++i) {
                for (size_t j = 0; j < __shape[1]; ++j) {
                    A_2d[i][j] = __data[calculate_offset(i, j)];
                    B_2d[i][j] = other.__data[other.calculate_offset(i, j)];
                }
            }
            std::vector<std::vector<T>> result_2d = andnot2_simd(A_2d, B_2d);
            ndarray<T> result_ndarray(__shape);
            for (size_t i = 0; i < __shape[0]; ++i) {
                for (size_t j = 0; j < __shape[1]; ++j) {
                    result_ndarray.__data[calculate_offset(i, j)] = result_2d[i][j];
                }
            }
            return result_ndarray;
    }
    
    throw std::invalid_argument("Unsupported array dimension.");
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


template <typename T>
std::ostream& operator<<(std::ostream& os, const ndarray<T>& arr) {
    if (arr.__shape.size() == 1) {
        os << "[";
        for (size_t i = 0; i < arr.__size; ++i) {
            if (i > 0) 
                os << ", ";
                
            os << arr.__data[i];
        }
        os << "]";
    } else if (arr.__shape.size() == 2) {
        os << "[";
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
        os << "]";
    }
    return os;
}
// ndarray.hpp
#ifndef NDARRAY_HPP
#define NDARRAY_HPP

#include <vector>
#include <array>
#include <cstdint>


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
    // 构造函数
    ndarray(const std::vector<size_t>& shape);

    const char *dtype() const noexcept;

    size_t itemsize() const noexcept;

    size_t ndim() const noexcept;
    
    size_t size() const noexcept;

    std::vector<size_t> shape() const noexcept;


public:
    std::vector<uint8_t> all(int axis) const;

    std::vector<T> round(int axis) const;

    float mean() const;

    double mean() const;

    float mean(int axis) const;

    // 访问元素
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;
};

#endif // NDARRAY_HPP    
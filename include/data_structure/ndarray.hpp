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
    // ¹¹Ôìº¯Êý
    ndarray(const std::vector<size_t>& shape);

    const char *dtype() const noexcept;

    size_t itemsize() const noexcept;

    size_t ndim() const noexcept;
    
    size_t size() const noexcept;

    std::vector<size_t> shape() const noexcept;


public:
    std::vector<uint8_t> all(int axis) const;

    float mean() const;

    double mean() const;

    float mean(int axis) const;


    // logical function
    ndarray<T> logical_and(const ndarray<T>& other);

    ndarray<T> logical_or(const ndarray<T>& other);

    ndarray<T> logical_xor(const ndarray<T>& other);

    ndarray<T> logical_andnot(const ndarray<T>& other);


    // math function
    ndarray<T> min();

    ndarray<T> max();

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
    

    // access element
    T& operator()(const std::vector<size_t>& indices);
    const T& operator()(const std::vector<size_t>& indices) const;

    friend std::ostream& operator<<(std::ostream& os, const ndarray<T>& arr);
};

#endif // NDARRAY_HPP    
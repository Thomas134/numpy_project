#include "../include/simd_traits.hpp"
#include <immintrin.h>
#include <cstdint>


template<typename T>
struct simd_traits;

template<>
struct simd_traits<int8_t> {
    using type = __m256i;
    static constexpr int width = 32;
    static inline type set1(int8_t val) { return _mm256_set1_epi8(val); }
    static inline type add(type a, type b) { return _mm256_add_epi8(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi8(a, b); }
    static inline void storeu(int8_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr)); }
};

template<>
struct simd_traits<uint8_t> {
    using type = __m256i;
    static constexpr int width = 32;
    static inline type set1(uint8_t val) { return _mm256_set1_epi8(static_cast<int8_t>(val)); }
    static inline type add(type a, type b) { return _mm256_add_epi8(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi8(a, b); }
    static inline void storeu(uint8_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr)); }
};

template<>
struct simd_traits<int16_t> {
    using type = __m256i;
    static constexpr int width = 16;
    static inline type set1(int16_t val) { return _mm256_set1_epi16(val); }
    static inline type add(type a, type b) { return _mm256_add_epi16(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi16(a, b); }
    static inline void storeu(int16_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr)); }
};

template<>
struct simd_traits<uint16_t> {
    using type = __m256i;
    static constexpr int width = 16;
    static inline type set1(uint16_t val) { return _mm256_set1_epi16(static_cast<int16_t>(val)); }
    static inline type add(type a, type b) { return _mm256_add_epi16(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi16(a, b); }
    static inline void storeu(uint16_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr)); }
};

template<>
struct simd_traits<int32_t> {
    using type = __m256i;
    static constexpr int width = 8;
    static inline type set1(int32_t val) { return _mm256_set1_epi32(val); }
    static inline type add(type a, type b) { return _mm256_add_epi32(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi32(a, b); }
    static inline void storeu(int32_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr)); }
};

template<>
struct simd_traits<uint32_t> {
    using type = __m256i;
    static constexpr int width = 8;
    static inline type set1(uint32_t val) { return _mm256_set1_epi32(static_cast<int32_t>(val)); }
    static inline type add(type a, type b) { return _mm256_add_epi32(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi32(a, b); }
    static inline void storeu(uint32_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr)); }
};

template<>
struct simd_traits<int64_t> {
    using type = __m256i;
    static constexpr int width = 4;
    static inline type set1(int64_t val) { return _mm256_set1_epi64x(val); }
    static inline type add(type a, type b) { return _mm256_add_epi64(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi64(a, b); }
    static inline void storeu(int64_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr)); }
};

template<>
struct simd_traits<uint64_t> {
    using type = __m256i;
    static constexpr int width = 4;
    static inline type set1(uint64_t val) { return _mm256_set1_epi64x(static_cast<int64_t>(val)); }
    static inline type add(type a, type b) { return _mm256_add_epi64(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi64(a, b); }
    static inline void storeu(uint64_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr)); }
};

template<>
struct simd_traits<float> {
    using type = __m256;
    static constexpr int width = 8;
    static inline type set1(float val) { return _mm256_set1_ps(val); }
    static inline type add(type a, type b) { return _mm256_add_ps(a, b); }
    static inline type mul(type a, type b) { return _mm256_mul_ps(a, b); }
    static inline void storeu(float* ptr, type val) { _mm256_storeu_ps(ptr, val); }
};

template<>
struct simd_traits<double> {
    using type = __m256d;
    static constexpr int width = 4;
    static inline type set1(double val) { return _mm256_set1_pd(val); }
    static inline type add(type a, type b) { return _mm256_add_pd(a, b); }
    static inline type mul(type a, type b) { return _mm256_mul_pd(a, b); }
    static inline void storeu(double* ptr, type val) { _mm256_storeu_pd(ptr, val); }
};

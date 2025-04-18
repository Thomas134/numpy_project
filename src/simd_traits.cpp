#include "../include/simd_traits.hpp"
#include <immintrin.h>
#include <cstdint>
#include <type_traits>


// linspace_simd
template<typename T>
struct linspace_simd_traits;

template<>
struct linspace_simd_traits<int16_t> {
    using type = __m256i;
    static constexpr int width = 16;
    static inline type set1(int16_t val) { return _mm256_set1_epi16(val); }
    static inline type add(type a, type b) { return _mm256_add_epi16(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi16(a, b); }
    static inline void storeu(int16_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val); }
};

template<>
struct linspace_simd_traits<uint16_t> {
    using type = __m256i;
    static constexpr int width = 16;
    static inline type set1(uint16_t val) { return _mm256_set1_epi16(static_cast<int16_t>(val)); }
    static inline type add(type a, type b) { return _mm256_add_epi16(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi16(a, b); }
    static inline void storeu(uint16_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val); }
};

template<>
struct linspace_simd_traits<int32_t> {
    using type = __m256i;
    static constexpr int width = 8;
    static inline type set1(int32_t val) { return _mm256_set1_epi32(val); }
    static inline type add(type a, type b) { return _mm256_add_epi32(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi32(a, b); }
    static inline void storeu(int32_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val); }
};

template<>
struct linspace_simd_traits<uint32_t> {
    using type = __m256i;
    static constexpr int width = 8;
    static inline type set1(uint32_t val) { return _mm256_set1_epi32(static_cast<int32_t>(val)); }
    static inline type add(type a, type b) { return _mm256_add_epi32(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi32(a, b); }
    static inline void storeu(uint32_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val); }
};

template<>
struct linspace_simd_traits<int64_t> {
    using type = __m256i;
    static constexpr int width = 4;
    static inline type set1(int64_t val) { return _mm256_set1_epi64x(val); }
    static inline type add(type a, type b) { return _mm256_add_epi64(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi64(a, b); }
    static inline void storeu(int64_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val); }
};

template<>
struct linspace_simd_traits<uint64_t> {
    using type = __m256i;
    static constexpr int width = 4;
    static inline type set1(uint64_t val) { return _mm256_set1_epi64x(static_cast<int64_t>(val)); }
    static inline type add(type a, type b) { return _mm256_add_epi64(a, b); }
    static inline type mul(type a, type b) { return _mm256_mullo_epi64(a, b); }
    static inline void storeu(uint64_t* ptr, type val) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val); }
};

template<>
struct linspace_simd_traits<float> {
    using type = __m256;
    static constexpr int width = 8;
    static inline type set1(float val) { return _mm256_set1_ps(val); }
    static inline type add(type a, type b) { return _mm256_add_ps(a, b); }
    static inline type mul(type a, type b) { return _mm256_mul_ps(a, b); }
    static inline void storeu(float* ptr, type val) { _mm256_storeu_ps(ptr, val); }
};

template<>
struct linspace_simd_traits<double> {
    using type = __m256d;
    static constexpr int width = 4;
    static inline type set1(double val) { return _mm256_set1_pd(val); }
    static inline type add(type a, type b) { return _mm256_add_pd(a, b); }
    static inline type mul(type a, type b) { return _mm256_mul_pd(a, b); }
    static inline void storeu(double* ptr, type val) { _mm256_storeu_pd(ptr, val); }
};


// inner_product_simd
template <typename T>
struct inner_product_simd_traits;

template <>
struct inner_product_simd_traits<float> {
    using scalar_type = float;
    using accum_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type zero() noexcept { return _mm256_setzero_ps(); }

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static simd_type mul_add(simd_type a, simd_type b, simd_type c) noexcept {
        return _mm256_fmadd_ps(a, b, c);
    }

    static accum_type horizontal_sum(simd_type sum) noexcept {
        __m128 low = _mm256_extractf128_ps(sum, 0);
        __m128 high = _mm256_extractf128_ps(sum, 1);
        low = _mm_add_ps(low, high);
        low = _mm_hadd_ps(low, low);
        low = _mm_hadd_ps(low, low);
        return _mm_cvtss_f32(low);
    }
};

template <>
struct inner_product_simd_traits<double> {
    using scalar_type = double;
    using accum_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type zero() noexcept { return _mm256_setzero_pd(); }

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static simd_type mul_add(simd_type a, simd_type b, simd_type c) noexcept {
        return _mm256_fmadd_pd(a, b, c);
    }

    static accum_type horizontal_sum(simd_type sum) noexcept {
        __m128d low = _mm256_extractf128_pd(sum, 0);
        __m128d high = _mm256_extractf128_pd(sum, 1);
        low = _mm_add_pd(low, high);
        low = _mm_hadd_pd(low, low);
        return _mm_cvtsd_f64(low);
    }
};

template <>
struct inner_product_simd_traits<int8_t> {
    using scalar_type = int8_t;
    using accum_type = int32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 32;

    static simd_type zero() noexcept { return _mm256_setzero_si256(); }

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static simd_type mul_add(simd_type a, simd_type b, simd_type acc) noexcept {
        // 符号扩展处理
        __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 0));
        __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
        __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 0));
        __m256i b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));

        __m256i prod_lo = _mm256_mullo_epi16(a_lo, b_lo);
        __m256i prod_hi = _mm256_mullo_epi16(a_hi, b_hi);

        // 扩展为32位并累加
        __m256i acc_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 0));
        __m256i acc_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1));
        acc = _mm256_add_epi32(acc, acc_lo);
        acc = _mm256_add_epi32(acc, acc_hi);

        acc_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 0));
        acc_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1));
        return _mm256_add_epi32(acc, _mm256_add_epi32(acc_lo, acc_hi));
    }

    static accum_type horizontal_sum(simd_type sum) noexcept {
        __m128i low = _mm256_extracti128_si256(sum, 0);
        __m128i high = _mm256_extracti128_si256(sum, 1);
        low = _mm_add_epi32(low, high);
        low = _mm_hadd_epi32(low, low);
        low = _mm_hadd_epi32(low, low);
        return _mm_extract_epi32(low, 0) + _mm_extract_epi32(low, 1);
    }
};

template <>
struct inner_product_simd_traits<uint8_t> {
    using scalar_type = uint8_t;
    using accum_type = uint32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 32;

    static simd_type zero() noexcept { return _mm256_setzero_si256(); }

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static simd_type mul_add(simd_type a, simd_type b, simd_type acc) noexcept {
        // 零扩展处理
        __m256i a_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a, 0));
        __m256i a_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(a, 1));
        __m256i b_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b, 0));
        __m256i b_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(b, 1));

        __m256i prod_lo = _mm256_mullo_epi16(a_lo, b_lo);
        __m256i prod_hi = _mm256_mullo_epi16(a_hi, b_hi);

        // 扩展为32位并累加
        __m256i acc_lo = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(prod_lo, 0));
        __m256i acc_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(prod_lo, 1));
        acc = _mm256_add_epi32(acc, acc_lo);
        acc = _mm256_add_epi32(acc, acc_hi);

        acc_lo = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(prod_hi, 0));
        acc_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(prod_hi, 1));
        return _mm256_add_epi32(acc, _mm256_add_epi32(acc_lo, acc_hi));
    }

    static accum_type horizontal_sum(simd_type sum) noexcept {
        __m128i low = _mm256_extracti128_si256(sum, 0);
        __m128i high = _mm256_extracti128_si256(sum, 1);
        low = _mm_add_epi32(low, high);
        low = _mm_hadd_epi32(low, low);
        low = _mm_hadd_epi32(low, low);
        return _mm_extract_epi32(low, 0) + _mm_extract_epi32(low, 1);
    }
};

template <>
struct inner_product_simd_traits<int16_t> {
    using scalar_type = int16_t;
    using accum_type = int32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 16;

    static simd_type zero() noexcept { return _mm256_setzero_si256(); }

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static simd_type mul_add(simd_type a, simd_type b, simd_type acc) noexcept {
        __m256i prod = _mm256_mullo_epi16(a, b);
        __m256i prod_lo = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod, 0));
        __m256i prod_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod, 1));
        return _mm256_add_epi32(acc, _mm256_add_epi32(prod_lo, prod_hi));
    }

    static accum_type horizontal_sum(simd_type sum) noexcept {
        __m128i low = _mm256_extracti128_si256(sum, 0);
        __m128i high = _mm256_extracti128_si256(sum, 1);
        low = _mm_add_epi32(low, high);
        low = _mm_hadd_epi32(low, low);
        low = _mm_hadd_epi32(low, low);
        return _mm_extract_epi32(low, 0) + _mm_extract_epi32(low, 1);
    }
};

template <>
struct inner_product_simd_traits<uint16_t> {
    using scalar_type = uint16_t;
    using accum_type = uint32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 16;

    static simd_type zero() noexcept { return _mm256_setzero_si256(); }

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static simd_type mul_add(simd_type a, simd_type b, simd_type acc) noexcept {
        __m256i prod = _mm256_mullo_epi16(a, b);
        __m256i prod_lo = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(prod, 0));
        __m256i prod_hi = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(prod, 1));
        return _mm256_add_epi32(acc, _mm256_add_epi32(prod_lo, prod_hi));
    }

    static accum_type horizontal_sum(simd_type sum) noexcept {
        __m128i low = _mm256_extracti128_si256(sum, 0);
        __m128i high = _mm256_extracti128_si256(sum, 1);
        low = _mm_add_epi32(low, high);
        low = _mm_hadd_epi32(low, low);
        low = _mm_hadd_epi32(low, low);
        return _mm_extract_epi32(low, 0) + _mm_extract_epi32(low, 1);
    }
};

template <>
struct inner_product_simd_traits<int32_t> {
    using scalar_type = int32_t;
    using accum_type = int32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 8;

    static simd_type zero() noexcept { return _mm256_setzero_si256(); }

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static simd_type mul_add(simd_type a, simd_type b, simd_type acc) noexcept {
        return _mm256_add_epi32(acc, _mm256_mullo_epi32(a, b));
    }

    static accum_type horizontal_sum(simd_type sum) noexcept {
        __m128i low = _mm256_extracti128_si256(sum, 0);
        __m128i high = _mm256_extracti128_si256(sum, 1);
        low = _mm_add_epi32(low, high);
        low = _mm_hadd_epi32(low, low);
        low = _mm_hadd_epi32(low, low);
        return _mm_extract_epi32(low, 0) + _mm_extract_epi32(low, 1);
    }
};

template <>
struct inner_product_simd_traits<uint32_t> {
    using scalar_type = uint32_t;
    using accum_type = uint32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 8;

    static simd_type zero() noexcept { return _mm256_setzero_si256(); }

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static simd_type mul_add(simd_type a, simd_type b, simd_type acc) noexcept {
        return _mm256_add_epi32(acc, _mm256_mullo_epi32(a, b));
    }

    static accum_type horizontal_sum(simd_type sum) noexcept {
        __m128i low = _mm256_extracti128_si256(sum, 0);
        __m128i high = _mm256_extracti128_si256(sum, 1);
        low = _mm_add_epi32(low, high);
        low = _mm_hadd_epi32(low, low);
        low = _mm_hadd_epi32(low, low);
        return _mm_extract_epi32(low, 0) + _mm_extract_epi32(low, 1);
    }
};

template <>
struct inner_product_simd_traits<int64_t> {
    using scalar_type = int64_t;
    using accum_type = int64_t;
    using simd_type = __m256i;
    static constexpr size_t step = 4;

    static simd_type zero() noexcept { return _mm256_setzero_si256(); }

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static simd_type mul_add(simd_type a, simd_type b, simd_type acc) noexcept {
        __m128i a_lo = _mm256_extracti128_si256(a, 0);
        __m128i a_hi = _mm256_extracti128_si256(a, 1);
        __m128i b_lo = _mm256_extracti128_si256(b, 0);
        __m128i b_hi = _mm256_extracti128_si256(b, 1);

        __m128i prod_lo = _mm_mul_epi32(a_lo, b_lo);
        __m128i prod_hi = _mm_mul_epi32(a_hi, b_hi);
        return _mm256_add_epi64(acc, _mm256_setr_m128i(prod_lo, prod_hi));
    }

    static accum_type horizontal_sum(simd_type sum) noexcept {
        __m128i low = _mm256_extracti128_si256(sum, 0);
        __m128i high = _mm256_extracti128_si256(sum, 1);
        low = _mm_add_epi64(low, high);
        return _mm_extract_epi64(low, 0) + _mm_extract_epi64(low, 1);
    }
};

template <>
struct inner_product_simd_traits<uint64_t> {
    using scalar_type = uint64_t;
    using accum_type = uint64_t;
    using simd_type = __m256i;
    static constexpr size_t step = 4;

    static simd_type zero() noexcept { 
        return _mm256_setzero_si256(); 
    }

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static simd_type mul_add(simd_type a, simd_type b, simd_type acc) noexcept {
        // 拆分为两个 128-bit 通道处理
        __m128i a_lo = _mm256_extracti128_si256(a, 0);
        __m128i a_hi = _mm256_extracti128_si256(a, 1);
        __m128i b_lo = _mm256_extracti128_si256(b, 0);
        __m128i b_hi = _mm256_extracti128_si256(b, 1);

        // 无符号 64-bit 乘法 (需要特殊处理)
        __m128i prod_lo = _mm_mul_epu32(a_lo, b_lo);  // 使用无符号乘法
        __m128i prod_hi = _mm_mul_epu32(a_hi, b_hi);

        // 合并结果并累加
        return _mm256_add_epi64(acc, _mm256_setr_m128i(prod_lo, prod_hi));
    }

    static accum_type horizontal_sum(simd_type sum) noexcept {
        // 128-bit 水平相加
        __m128i low = _mm256_extracti128_si256(sum, 0);
        __m128i high = _mm256_extracti128_si256(sum, 1);
        low = _mm_add_epi64(low, high);

        // 提取并返回最终结果
        return static_cast<uint64_t>(_mm_extract_epi64(low, 0)) + static_cast<uint64_t>(_mm_extract_epi64(low, 1));
    }
};


// and_simd
template <typename T>
struct and_simd_traits;

template <typename T>
struct and_simd_traits {
    using scalar_type = T;
    using simd_type = __m256i;
    static constexpr size_t step = sizeof(__m256i) / sizeof(T);

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_and_si256(a, b);
    }
};

template <>
struct and_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_and_ps(a, b);
    }
};

template <>
struct and_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_pd(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_and_pd(a, b);
    }
};


// or_simd
template <typename T>
struct or_simd_traits;

template <typename T>
struct or_simd_traits {
    using scalar_type = T;
    using simd_type = __m256i;
    static constexpr size_t step = sizeof(__m256i) / sizeof(T);

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_or_si256(a, b);
    }
};

template <>
struct or_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_or_ps(a, b);
    }
};

template <>
struct or_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_pd(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_or_pd(a, b);
    }
};


// xor_simd
template <typename T>
struct xor_simd_traits;

template <typename T>
struct xor_simd_traits {
    using scalar_type = T;
    using simd_type = __m256i;
    static constexpr size_t step = sizeof(__m256i) / sizeof(T);

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_xor_si256(a, b);
    }
};

template <>
struct xor_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_xor_ps(a, b);
    }
};

template <>
struct xor_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_pd(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_xor_pd(a, b);
    }
};


// andnot_simd
template <typename T>
struct andnot_simd_traits;

template <typename T>
struct andnot_simd_traits {
    using scalar_type = T;
    using simd_type = __m256i;
    static constexpr size_t step = sizeof(__m256i) / sizeof(T);

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_andnot_si256(a, b);
    }
};

template <>
struct andnot_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_andnot_ps(a, b);
    }
};

template <>
struct andnot_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_pd(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_andnot_pd(a, b);
    }
};


// testc_simd
template <typename T>
struct testc_simd_traits;

template <typename T>
struct testc_simd_traits {
    using scalar_type = T;
    using simd_type = __m256i;
    static constexpr size_t step = sizeof(__m256i) / sizeof(T);

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static int op(simd_type a, simd_type b) noexcept {
        return _mm256_testc_si256(a, b);
    }
};

template <>
struct testc_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static int op(simd_type a, simd_type b) noexcept {
        return _mm256_testc_ps(a, b);
    }
};

template <>
struct testc_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static int op(simd_type a, simd_type b) noexcept {
        return _mm256_testc_pd(a, b);
    }
};


// slli_simd
template <typename T>
struct slli_simd_traits;

template <typename T>
struct slli_simd_traits {
    using scalar_type = T;
    using simd_type = __m256i;
    static constexpr size_t step = sizeof(__m256i) / sizeof(T);

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val);
    }

    static simd_type op(simd_type a, int imm8) noexcept {
        return _mm256_slli_si256(a, imm8);
    }
};


// srli_simd
template <typename T>
struct srli_simd_traits;

template <typename T>
struct srli_simd_traits {
    using scalar_type = T;
    using simd_type = __m256i;
    static constexpr size_t step = sizeof(__m256i) / sizeof(T);

    static simd_type load(const scalar_type* ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), val);
    }

    static simd_type op(simd_type a, int imm8) noexcept {
        return _mm256_srli_si256(a, imm8);
    }
};


// min_simd
template <typename T>
struct min_simd_traits;

template <>
struct min_simd_traits<int8_t> {
    using scalar_type = int8_t;
    using simd_type = __m256i;
    static constexpr size_t step = 32;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_min_epi8(a, b);
    }
};

template <>
struct min_simd_traits<uint8_t> {
    using scalar_type = uint8_t;
    using simd_type = __m256i;
    static constexpr size_t step = 32;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_min_epu8(a, b);
    }
};

template <>
struct min_simd_traits<int16_t> {
    using scalar_type = int16_t;
    using simd_type = __m256i;
    static constexpr size_t step = 16;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_min_epi16(a, b);
    }
};

template <>
struct min_simd_traits<uint16_t> {
    using scalar_type = uint16_t;
    using simd_type = __m256i;
    static constexpr size_t step = 16;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_min_epu16(a, b);
    }
};

template <>
struct min_simd_traits<int32_t> {
    using scalar_type = int32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_min_epi32(a, b);
    }
};

template <>
struct min_simd_traits<uint32_t> {
    using scalar_type = uint32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_min_epu32(a, b);
    }
};

template <>
struct min_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_min_ps(a, b);
    }
};

template <>
struct min_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_pd(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_min_pd(a, b);
    }
};


// max_simd
template <typename T>
struct max_simd_traits;

template <>
struct max_simd_traits<int8_t> {
    using scalar_type = int8_t;
    using simd_type = __m256i;
    static constexpr size_t step = 32;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_max_epi8(a, b);
    }
};

template <>
struct max_simd_traits<uint8_t> {
    using scalar_type = uint8_t;
    using simd_type = __m256i;
    static constexpr size_t step = 32;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_max_epu8(a, b);
    }
};

template <>
struct max_simd_traits<int16_t> {
    using scalar_type = int16_t;
    using simd_type = __m256i;
    static constexpr size_t step = 16;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_max_epi16(a, b);
    }
};

template <>
struct max_simd_traits<uint16_t> {
    using scalar_type = uint16_t;
    using simd_type = __m256i;
    static constexpr size_t step = 16;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_max_epu16(a, b);
    }
};

template <>
struct max_simd_traits<int32_t> {
    using scalar_type = int32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_max_epi32(a, b);
    }
};

template <>
struct max_simd_traits<uint32_t> {
    using scalar_type = uint32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_max_epu32(a, b);
    }
};

template <>
struct max_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_max_ps(a, b);
    }
};

template <>
struct max_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_pd(ptr, val);
    }

    static simd_type op(simd_type a, simd_type b) noexcept {
        return _mm256_max_pd(a, b);
    }
};


// sqrt_simd
template <typename T>
struct sqrt_simd_traits;

template <>
struct sqrt_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_sqrt_ps(a);
    }
};

template <>
struct sqrt_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_pd(ptr, val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_sqrt_pd(a);
    }
};


// rsqrt_simd
template <typename T>
struct rsqrt_simd_traits;

template <>
struct rsqrt_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_rsqrt_ps(a);
    }
};


// round_simd
template <typename T>
struct round_simd_traits;

template <>
struct round_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT);
    }
};

template <>
struct round_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_pd(ptr, val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT);
    }
};


// ceil_simd
template <typename T>
struct ceil_simd_traits;

template <>
struct ceil_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_ceil_ps(a);
    }
};

template <>
struct ceil_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_pd(ptr, val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_ceil_pd(a);
    }
};


// floor_simd
template <typename T>
struct floor_simd_traits;

template <>
struct floor_simd_traits<float> {
    using scalar_type = float;
    using simd_type = __m256;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_ps(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_ps(ptr, val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_floor_ps(a);
    }
};

template <>
struct floor_simd_traits<double> {
    using scalar_type = double;
    using simd_type = __m256d;
    static constexpr size_t step = 4;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_pd(ptr);
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_pd(ptr, val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_floor_pd(a);
    }
};


// abs_simd
template <typename T>
struct abs_simd_traits;

template <>
struct abs_simd_traits<int8_t> {
    using scalar_type = int8_t;
    using simd_type = __m256i;
    static constexpr size_t step = 32;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_abs_epi8(a);
    }
};

template <>
struct abs_simd_traits<int16_t> {
    using scalar_type = int16_t;
    using simd_type = __m256i;
    static constexpr size_t step = 16;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_abs_epi16(a);
    }
};

template <>
struct abs_simd_traits<int32_t> {
    using scalar_type = int32_t;
    using simd_type = __m256i;
    static constexpr size_t step = 8;

    static simd_type load(const scalar_type *ptr) noexcept {
        return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
    }

    static void store(scalar_type *ptr, simd_type val) noexcept {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), val);
    }

    static simd_type op(simd_type a) noexcept {
        return _mm256_abs_epi32(a);
    }
};

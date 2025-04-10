#ifndef SIMD_TRAITS_HPP
#define SIMD_TRAITS_HPP

template<typename T>
struct linspace_simd_traits;

template <typename T>
struct inner_product_simd_traits;

template <typename T>
struct and_simd_traits;

template <typename T>
struct or_simd_traits;

template <typename T>
struct xor_simd_traits;

template <typename T>
struct andnot_simd_traits;

template <typename T>
struct testc_simd_traits;

template <typename T>
struct slli_simd_traits;

template <typename T>
struct srli_simd_traits;

template <typename T>
struct sm4rnds4_simd_traits;

template <typename T>
struct sm4key4_simd_traits;

template <typename T>
struct min_simd_traits;

template <typename T>
struct max_simd_traits;

template <typename T>
struct sqrt_simd_traits;

template <typename T>
struct rsqrt_simd_traits;

template <typename T>
struct round_simd_traits;

template <typename T>
struct ceil_simd_traits;

template <typename T>
struct floor_simd_traits;

template <typename T>
struct abs_simd_traits;

#endif

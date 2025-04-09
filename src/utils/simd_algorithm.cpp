#include "../../include/utils/simd_algorithm.hpp"


inline __m256 mm256_fmaf(__m256 a, __m256 b, __m256 c) {
    return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}

inline __m256 fast_log_sse(__m256 a) {
    __m256i aInt = *(__m256i*)(&a);
    __m256i e = _mm256_sub_epi32(aInt, _mm256_set1_epi32(0x3f2aaaab));
    e = _mm256_and_si256(e, _mm256_set1_epi32(0xff800000));
    __m256i subtr = _mm256_sub_epi32(aInt, e);
    __m256 m = *(__m256*)&subtr;
    __m256 i = _mm256_mul_ps(_mm256_cvtepi32_ps(e), _mm256_set1_ps(1.19209290e-7f));
    __m256 f = _mm256_sub_ps(m, _mm256_set1_ps(1.0f));
    __m256 s = _mm256_mul_ps(f, f);
    __m256 r = mm256_fmaf(_mm256_set1_ps(0.230836749f), f, _mm256_set1_ps(-0.279208571f));
    __m256 t = mm256_fmaf(_mm256_set1_ps(0.331826031f), f, _mm256_set1_ps(-0.498910338f));
    r = mm256_fmaf(r, s, t);
    r = mm256_fmaf(r, s, f);
    r = mm256_fmaf(i, _mm256_set1_ps(0.693147182f), r);
    return r;
}

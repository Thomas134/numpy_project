#include "../include/xsimd_traits.hpp"
#include <xsimd/xsimd.hpp>
#include <type_traits>

// log_simd
template <typename T>
struct log_simd_traits {
    using scalar_type = T;
    using simd_type = typename std::conditional<
        std::is_same<T, float>::value,
        xsimd::simd_type<float>,
        typename std::conditional<
            std::is_same<T, double>::value,
            xsimd::simd_type<double>,
            void
        >::type
    >::type;

    static_assert(!std::is_same<simd_type, void>::value, "Unsupported scalar type. Only float and double are supported.");


    static constexpr size_t step = simd_type::size;

    static simd_type load(const scalar_type* ptr) noexcept {
        return simd_type::load_unaligned(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        val.store_unaligned(ptr);
    }

    static simd_type op(simd_type a) noexcept {
        return xsimd::log(a);
    }
};


// log2_simd
template <typename T>
struct log2_simd_traits {
    using scalar_type = T;
    using simd_type = typename std::conditional<
        std::is_same<T, float>::value,
        xsimd::simd_type<float>,
        typename std::conditional<
            std::is_same<T, double>::value,
            xsimd::simd_type<double>,
            void
        >::type
    >::type;

    static_assert(!std::is_same<simd_type, void>::value, "Unsupported scalar type. Only float and double are supported.");

    static constexpr size_t step = simd_type::size;

    static simd_type load(const scalar_type* ptr) noexcept {
        return simd_type::load_unaligned(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        val.store_unaligned(ptr);
    }

    static simd_type op(simd_type a) noexcept {
        return xsimd::log2(a);
    }
};


// log10_simd
template <typename T>
struct log10_simd_traits {
    using scalar_type = T;
    using simd_type = typename std::conditional<
        std::is_same<T, float>::value,
        xsimd::simd_type<float>,
        typename std::conditional<
            std::is_same<T, double>::value,
            xsimd::simd_type<double>,
            void
        >::type
    >::type;

    static_assert(!std::is_same<simd_type, void>::value, "Unsupported scalar type. Only float and double are supported.");

    static constexpr size_t step = simd_type::size;

    static simd_type load(const scalar_type* ptr) noexcept {
        return simd_type::load_unaligned(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        val.store_unaligned(ptr);
    }

    static simd_type op(simd_type a) noexcept {
        return xsimd::log10(a);
    }
};


// sin_simd
template <typename T>
struct sin_simd_traits {
    using scalar_type = T;
    using simd_type = typename std::conditional<
        std::is_same<T, float>::value,
        xsimd::simd_type<float>,
        typename std::conditional<
            std::is_same<T, double>::value,
            xsimd::simd_type<double>,
            void
        >::type
    >::type;

    static_assert(!std::is_same<simd_type, void>::value, "Unsupported scalar type. Only float and double are supported.");

    static constexpr size_t step = simd_type::size;

    static simd_type load(const scalar_type* ptr) noexcept {
        return simd_type::load_unaligned(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        val.store_unaligned(ptr);
    }

    static simd_type op(simd_type a) noexcept {
        return xsimd::sin(a);
    }
};


// cos_simd;
template <typename T>
struct cos_simd_traits {
    using scalar_type = T;
    using simd_type = typename std::conditional<
        std::is_same<T, float>::value,
        xsimd::simd_type<float>,
        typename std::conditional<
            std::is_same<T, double>::value,
            xsimd::simd_type<double>,
            void
        >::type
    >::type;

    static_assert(!std::is_same<simd_type, void>::value, "Unsupported scalar type. Only float and double are supported.");

    static constexpr size_t step = simd_type::size;

    static simd_type load(const scalar_type* ptr) noexcept {
        return simd_type::load_unaligned(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        val.store_unaligned(ptr);
    }

    static simd_type op(simd_type a) noexcept {
        return xsimd::cos(a);
    }
};


// sincos_simd
template <typename T>
struct sincos_simd_traits {
    using scalar_type = T;
    using simd_type = typename std::conditional<
        std::is_same<T, float>::value,
        xsimd::simd_type<float>,
        typename std::conditional<
            std::is_same<T, double>::value,
            xsimd::simd_type<double>,
            void
        >::type
    >::type;

    static_assert(!std::is_same<simd_type, void>::value, "Unsupported scalar type. Only float and double are supported.");

    static constexpr size_t step = simd_type::size;

    static simd_type load(const scalar_type* ptr) noexcept {
        return simd_type::load_unaligned(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        val.store_unaligned(ptr);
    }

    static std::pair<simd_type, simd_type> op(simd_type a) noexcept {
        return xsimd::sincos(a);
    }
};


// tan_simd;
template <typename T>
struct tan_simd_traits {
    using scalar_type = T;
    using simd_type = typename std::conditional<
        std::is_same<T, float>::value,
        xsimd::simd_type<float>,
        typename std::conditional<
            std::is_same<T, double>::value,
            xsimd::simd_type<double>,
            void
        >::type
    >::type;

    static_assert(!std::is_same<simd_type, void>::value, "Unsupported scalar type. Only float and double are supported.");

    static constexpr size_t step = simd_type::size;

    static simd_type load(const scalar_type* ptr) noexcept {
        return simd_type::load_unaligned(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        val.store_unaligned(ptr);
    }

    static simd_type op(simd_type a) noexcept {
        return xsimd::tan(a);
    }
};


// asin_simd
template <typename T>
struct asin_simd_traits {
    using scalar_type = T;
    using simd_type = typename std::conditional<
        std::is_same<T, float>::value,
        xsimd::simd_type<float>,
        typename std::conditional<
            std::is_same<T, double>::value,
            xsimd::simd_type<double>,
            void
        >::type
    >::type;

    static_assert(!std::is_same<simd_type, void>::value, "Unsupported scalar type. Only float and double are supported.");

    static constexpr size_t step = simd_type::size;

    static simd_type load(const scalar_type* ptr) noexcept {
        return simd_type::load_unaligned(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        val.store_unaligned(ptr);
    }

    static simd_type op(simd_type a) noexcept {
        return xsimd::asin(a);
    }
};


// acos_simd
template <typename T>
struct acos_simd_traits {
    using scalar_type = T;
    using simd_type = typename std::conditional<
        std::is_same<T, float>::value,
        xsimd::simd_type<float>,
        typename std::conditional<
            std::is_same<T, double>::value,
            xsimd::simd_type<double>,
            void
        >::type
    >::type;

    static_assert(!std::is_same<simd_type, void>::value, "Unsupported scalar type. Only float and double are supported.");

    static constexpr size_t step = simd_type::size;

    static simd_type load(const scalar_type* ptr) noexcept {
        return simd_type::load_unaligned(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        val.store_unaligned(ptr);
    }

    static simd_type op(simd_type a) noexcept {
        return xsimd::acos(a);
    }
};


// atan_simd
template <typename T>
struct atan_simd_traits {
    using scalar_type = T;
    using simd_type = typename std::conditional<
        std::is_same<T, float>::value,
        xsimd::simd_type<float>,
        typename std::conditional<
            std::is_same<T, double>::value,
            xsimd::simd_type<double>,
            void
        >::type
    >::type;

    static_assert(!std::is_same<simd_type, void>::value, "Unsupported scalar type. Only float and double are supported.");

    static constexpr size_t step = simd_type::size;

    static simd_type load(const scalar_type* ptr) noexcept {
        return simd_type::load_unaligned(ptr);
    }

    static void store(scalar_type* ptr, simd_type val) noexcept {
        val.store_unaligned(ptr);
    }

    static simd_type op(simd_type a) noexcept {
        return xsimd::atan(a);
    }
};

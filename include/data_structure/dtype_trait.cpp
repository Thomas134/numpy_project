#ifndef DTYPE_TRAIT
#define DTYPE_TRAIT

#include <cstdint>
#include <cstddef>
#include <string>

template <typename T> 
struct dtype_traits;

template<>
struct dtype_traits<int8_t> {
    static constexpr const char* name = "int8";
    static constexpr size_t size = sizeof(int8_t);
};

template<> 
struct dtype_traits<int16_t> {
    static constexpr const char* name = "int16";
    static constexpr size_t size = sizeof(int16_t);
};

template<> 
struct dtype_traits<int32_t> {
    static constexpr const char* name = "int32";
    static constexpr size_t size = sizeof(int32_t);
};

template<> 
struct dtype_traits<int64_t> {
    static constexpr const char* name = "int64";
    static constexpr size_t size = sizeof(int64_t);
};

template<> 
struct dtype_traits<uint8_t> {
    static constexpr const char* name = "uint8";
    static constexpr size_t size = sizeof(uint8_t);
};

template<> 
struct dtype_traits<uint16_t> {
    static constexpr const char* name = "uint16";
    static constexpr size_t size = sizeof(uint16_t);
};

template<> 
struct dtype_traits<uint32_t> {
    static constexpr const char* name = "uint32";
    static constexpr size_t size = sizeof(uint32_t);
};

template<> 
struct dtype_traits<uint64_t> {
    static constexpr const char* name = "uint64";
    static constexpr size_t size = sizeof(uint64_t);
};

template<> 
struct dtype_traits<float> {
    static constexpr const char* name = "float32";
    static constexpr size_t size = sizeof(float);
};

template<> 
struct dtype_traits<double> {
    static constexpr const char* name = "float64";
    static constexpr size_t size = sizeof(double);
};

template<> 
struct dtype_traits<long double> {
    static constexpr const char* name = "long double";
    static constexpr size_t size = sizeof(long double);
};

template<> 
struct dtype_traits<char> {
    static constexpr const char* name = "char";
    static constexpr size_t size = sizeof(char);
};

template<> 
struct dtype_traits<std::string> {
    static constexpr const char* name = "string";
    static constexpr size_t size = sizeof(std::string);
};

#endif

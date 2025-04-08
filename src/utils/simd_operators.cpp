#include "../../include/utils/simd_operators.hpp"
#include "../../include/simd_traits.hpp"
#include <stdexcept>

template <typename T, typename Traits, typename UnaryOp>
std::vector<T> apply_unary_op_simd(const std::vector<T>& A, UnaryOp unary_op) {
    if (A.empty())
        throw std::invalid_argument("Input vector can't be empty");

    std::vector<T> result(A.size());
    size_t i = 0;
    const size_t simd_step = Traits::step;

    for (; i <= A.size() - simd_step; i += simd_step) {
        auto vec_a = Traits::load(&A[i]);
        auto vec_result = Traits::op(vec_a);
        Traits::store(&result[i], vec_result);
    }

    for (; i < A.size(); ++i)
        result[i] = unary_op(A[i]);

    return result;
}


template <typename T, typename Traits, typename BinaryOp>
std::vector<T> apply_binary_op_simd(const std::vector<T>& A,
                                    const std::vector<T>& B,
                                    BinaryOp binary_op) {
    if (A.empty() || B.empty())
        throw std::invalid_argument("Input vectors can't be empty");

    if (A.size() != B.size())
        throw std::invalid_argument("Input vectors must be of the same size.");

    std::vector<T> result(A.size());
    size_t i = 0;
    const size_t simd_step = Traits::step;

    for (; i <= A.size() - simd_step; i += simd_step) {
        auto vec_a = Traits::load(&A[i]);
        auto vec_b = Traits::load(&B[i]);
        auto vec_result = Traits::op(vec_a, vec_b);
        Traits::store(&result[i], vec_result);
    }

    for (; i < A.size(); ++i)
        result[i] = binary_op(A[i], B[i]);

    return result;
}

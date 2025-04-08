#ifndef SIMD_OPERATORS_HPP
#define SIMD_OPERATORS_HPP

#include <vector>

template <typename T, typename Traits, typename UnaryOp>
std::vector<T> apply_unary_op_simd(const std::vector<T>& A, UnaryOp unary_op);

template <typename T, typename Traits, typename BinaryOp>
std::vector<std::vector<T>> apply_binary_op_simd(const std::vector<std::vector<T>>& A,
                                                 const std::vector<std::vector<T>>& B,
                                                 BinaryOp binary_op);


#endif

#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

template <typename T, typename UnaryOp>
std::vector<std::vector<T>> apply_unary_op(const std::vector<std::vector<T>>& A, UnaryOp op);

template <typename T, typename BinaryOp>
std::vector<std::vector<T>> apply_binary_op(const std::vector<std::vector<T>>& A, 
                                            const std::vector<std::vector<T>>& B, 
                                            BinaryOp op);

#endif

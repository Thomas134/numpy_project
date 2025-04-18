#include "../include/utils/utils.hpp"
#include <stdexcept>

template <typename T, typename UnaryOp>
std::vector<std::vector<T>> apply_unary_op(const std::vector<std::vector<T>>& A, UnaryOp op) {
    if (A.empty())
        throw std::invalid_argument("Input 2D vector can't be empty");

    std::vector<std::vector<T>> result;
    for (const auto& row : A)
        result.push_back(op(row));

    return result;
}

template <typename T, typename BinaryOp>
std::vector<std::vector<T>> apply_binary_op(const std::vector<std::vector<T>>& A, 
                                            const std::vector<std::vector<T>>& B, 
                                            BinaryOp op) {
    if (A.empty() || B.empty())
        throw std::invalid_argument("Input 2D vectors can't be empty");

    if (A.size() != B.size())
        throw std::invalid_argument("Input 2D vectors must have the same number of rows.");

    std::vector<std::vector<T>> result;
    for (size_t i = 0; i < A.size(); ++i) {
        result.push_back(op(A[i], B[i]));
    }
    return result;
}

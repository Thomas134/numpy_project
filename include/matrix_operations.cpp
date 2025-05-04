#ifndef MATRIX_OPERATIONS_HPP
#define MATRIX_OPERATIONS_HPP

#include <vector>
#include <string>
#include <algorithm>
#include <type_traits>
#include <stdexcept>
#if defined(__AVX2__) && (defined(__UBUNTU__) || defined(__DEBIAN__) || defined(__KALI__))
    #include <cblas.h>
#elif defined(__riscv) || defined(__FEDORA__) || defined(__ARCHLINUX__)
    #include <openblas/cblas.h>
#endif

namespace internal {
    // dot
    template <typename T>
    std::vector<std::vector<T>> dot(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // transpose
    template <typename T>
    std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& mat);


    // add1
    template <typename T>
    std::vector<T> add1(const std::vector<T>& A, const std::vector<T>& B);


    // add2
    template <typename T>
    std::vector<std::vector<T>> add2(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);


    // subtract1
    template <typename T>
    std::vector<T> subtract1(const std::vector<T>& A, const std::vector<T>& B);


    // subtract2
    template <typename T>
    std::vector<std::vector<T>> subtract2(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B);
}


namespace internal {
    // dot
    template <typename T>
    std::vector<std::vector<T>> dot(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");
        static_assert(!std::is_same_v<T, char>);

        const size_t M = A.size();
        const size_t K_A = A[0].size();
        const size_t K_B = B.size();
        const size_t N = B[0].size();

        if (K_A != K_B) {
            throw std::invalid_argument("Matrix dimension mismatch");
        }

        std::vector<std::vector<T>> C(M, std::vector<T>(N));

        auto performDotProduct = [M, K_A, N](const auto& matA, const auto& matB, auto& matC) {
            using ElementType = std::decay_t<decltype(matA[0][0])>;
            std::vector<ElementType> flat_A, flat_B;
            for (const auto& row : matA) {
                flat_A.insert(flat_A.end(), row.begin(), row.end());
            }
            for (const auto& row : matB) {
                flat_B.insert(flat_B.end(), row.begin(), row.end());
            }

            std::vector<ElementType> flat_C(M * N);

            if constexpr (std::is_same_v<ElementType, float>) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K_A, 1.0f, flat_A.data(), K_A, flat_B.data(), N, 0.0f, flat_C.data(), N);
            } else if constexpr (std::is_same_v<ElementType, double>) {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K_A, 1.0, flat_A.data(), K_A, flat_B.data(), N, 0.0, flat_C.data(), N);
            } else {
                std::vector<float> float_flat_A(flat_A.begin(), flat_A.end());
                std::vector<float> float_flat_B(flat_B.begin(), flat_B.end());
                std::vector<float> float_flat_C(M * N);

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K_A, 1.0f, float_flat_A.data(), K_A, float_flat_B.data(), N, 0.0f, float_flat_C.data(), N);

                for (size_t i = 0; i < M * N; ++i) {
                    flat_C[i] = static_cast<ElementType>(float_flat_C[i]);
                }
            }

            matC.reserve(M);
            for (size_t i = 0; i < M; ++i) {
                matC[i].assign(flat_C.begin() + i * N, flat_C.begin() + (i + 1) * N);
            }
        };

        performDotProduct(A, B, C);
        return C;
    }    


    template <typename T>
    std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& mat) {
        const size_t rows = mat.size();
        const size_t cols = mat[0].size();

        if constexpr (std::is_same_v<T, float>) {
            std::vector<float> flat_mat;
            flat_mat.reserve(rows * cols);

            for (const auto& row : mat)
                flat_mat.insert(flat_mat.end(), row.begin(), row.end());

            std::vector<float> flat_result(cols * rows);

            cblas_somatcopy(CblasRowMajor,
                            CblasTrans, 
                            rows,      
                            cols,          
                            1.0f,         
                            flat_mat.data(), 
                            cols,        
                            flat_result.data(),
                            rows);        

            std::vector<std::vector<float>> result;
            result.reserve(cols);

            for (size_t i = 0; i < cols; ++i) {
                result.emplace_back(
                    flat_result.begin() + i * rows,
                    flat_result.begin() + (i + 1) * rows
                );
            }

            return result;
        } else if constexpr (std::is_same_v<T, double>) {
            std::vector<double> flat_mat;
            flat_mat.reserve(rows * cols);

            for (const auto& row : mat)
                flat_mat.insert(flat_mat.end(), row.begin(), row.end());

            std::vector<double> flat_result(cols * rows);

            cblas_domatcopy(CblasRowMajor,
                            CblasTrans, 
                            rows,      
                            cols,          
                            1.0f,         
                            flat_mat.data(), 
                            cols,        
                            flat_result.data(),
                            rows);        

            std::vector<std::vector<double>> result;
            result.reserve(cols);

            for (size_t i = 0; i < cols; ++i) {
                result.emplace_back(
                    flat_result.begin() + i * rows,
                    flat_result.begin() + (i + 1) * rows
                );
            }

            return result;
        } else {
            std::vector<std::vector<T>> result(cols, std::vector<T>(rows));
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < cols; ++j)
                    result[j][i] = mat[i][j];

            return result;
        }
    }


    // add1
    template <typename T>
    std::vector<T> add1(const std::vector<T>& A, const std::vector<T>& B) {
        static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");
        static_assert(!std::is_same_v<T, char>);

        if (A.size() != B.size()) {
            throw std::invalid_argument("Vector dimension mismatch");
        }

        size_t N = A.size();
        std::vector<T> C(N);

        if constexpr (std::is_same_v<T, float>) {
            cblas_scopy(N, B.data(), 1, C.data(), 1);
            cblas_saxpy(N, 1.0, A.data(), 1, C.data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_dcopy(N, B.data(), 1, C.data(), 1);
            cblas_daxpy(N, 1.0, A.data(), 1, C.data(), 1);
        } else {
            std::vector<float> float_A(N), float_B(N), float_C(N);
            for (size_t i = 0; i < N; ++i) {
                float_A[i] = static_cast<float>(A[i]);
                float_B[i] = static_cast<float>(B[i]);
            }
            cblas_scopy(N, float_B.data(), 1, float_C.data(), 1);
            cblas_saxpy(N, 1.0, float_A.data(), 1, float_C.data(), 1);
            for (size_t i = 0; i < N; ++i) {
                C[i] = static_cast<T>(float_C[i]);
            }
        }

        return C;
    }


    // add2
    template <typename T>
    std::vector<std::vector<T>> add2(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");

        if (A.size() != B.size())
            throw std::invalid_argument("Matrix row dimension mismatch");

        size_t rows = A.size();
        std::vector<std::vector<T>> C(rows);

        for (size_t i = 0; i < rows; ++i)
            C[i] = add1(A[i], B[i]);

        return C;
    } 

    // subtract1
    template <typename T>
    std::vector<T> subtract1(const std::vector<T>& A, const std::vector<T>& B) {
        static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");
        static_assert(!std::is_same_v<T, char>);

        if (A.size() != B.size()) {
            throw std::invalid_argument("Vector dimension mismatch");
        }

        size_t N = A.size();
        std::vector<T> C(N);

        if constexpr (std::is_same_v<T, float>) {
            cblas_scopy(N, A.data(), 1, C.data(), 1);
            cblas_saxpy(N, -1.0, B.data(), 1, C.data(), 1);
        } else if constexpr (std::is_same_v<T, double>) {
            cblas_dcopy(N, A.data(), 1, C.data(), 1);
            cblas_daxpy(N, -1.0, B.data(), 1, C.data(), 1);
        } else {
            std::vector<float> float_A(N), float_B(N), float_C(N);
            for (size_t i = 0; i < N; ++i) {
                float_A[i] = static_cast<float>(A[i]);
                float_B[i] = static_cast<float>(B[i]);
            }
            cblas_scopy(N, float_A.data(), 1, float_C.data(), 1);
            cblas_saxpy(N, -1.0, float_B.data(), 1, float_C.data(), 1);
            for (size_t i = 0; i < N; ++i) {
                C[i] = static_cast<T>(float_C[i]);
            }
        }

        return C;
    }


    // subtract2
    template <typename T>
    std::vector<std::vector<T>> subtract2(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
        static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic");

        if (A.size() != B.size())
            throw std::invalid_argument("Matrix row dimension mismatch");

        size_t rows = A.size();
        std::vector<std::vector<T>> C(rows);

        for (size_t i = 0; i < rows; ++i)
            C[i] = subtract1(A[i], B[i]);

        return C;
    }

}


#endif

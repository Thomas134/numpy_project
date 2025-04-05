#include "../include/matrix_operations.hpp"
#include <string>
#include <algorithm>
#include <type_traits>
#include <stdexcept>
#include <cblas.h>

namespace internal {
    // dot
    template <typename T>
    std::vector<std::vector<T>> dot(std::vector<std::vector<T>> A, std::vector<std::vector<T>> B) {
        static_assert(!std::is_same_v<T, char>);
        static_assert(!std::is_same_v<T, std::string>);

        const size_t M = A.size();
        const size_t K_A = A[0].size();
        const size_t K_B = B.size();
        const size_t N = B[0].size();

        if (K_A != K_B) {
            throw std::invalid_argument("Matrix dimension mismatch");
        }

        std::vector<std::vector<float>> float_A(M, std::vector<float>(K_A));
        std::vector<std::vector<float>> float_B(K_B, std::vector<float>(N));

        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < K_A; ++j)
                float_A[i][j] = static_cast<float>(A[i][j]);

        for (size_t i = 0; i < K_B; ++i)
            for (size_t j = 0; j < N; ++j)
                float_B[i][j] = static_cast<float>(B[i][j]);

        std::vector<float> flat_A, flat_B;
        for (const auto& row : float_A) {
            flat_A.insert(flat_A.end(), row.begin(), row.end());
        }
        for (const auto& row : float_B) {
            flat_B.insert(flat_B.end(), row.begin(), row.end());
        }

        std::vector<float> flat_C(M * N);

        cblas_sgemm(CblasRowMajor,  
                    CblasNoTrans,  
                    CblasNoTrans, 
                    M, 
                    N, 
                    K_A, 
                    1.0f,         
                    flat_A.data(), 
                    K_A,
                    flat_B.data(), 
                    N,
                    0.0f,   
                    flat_C.data(),
                    N); 

        std::vector<std::vector<float>> float_C;
        float_C.reserve(M);
        for (size_t i = 0; i < M; ++i) {
            float_C.emplace_back(
                flat_C.begin() + i * N,
                flat_C.begin() + (i + 1) * N
            );
        }

        std::vector<std::vector<T>> C(M, std::vector<T>(N));
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                C[i][j] = static_cast<T>(float_C[i][j]);
            }
        }

        return C;
    }

    template <>
    std::vector<std::vector<float>> dot(std::vector<std::vector<float>> A, 
                                        std::vector<std::vector<float>> B) {
        const size_t M = A.size();   
        const size_t K_A = A[0].size(); 
        const size_t K_B = B.size();    
        const size_t N = B[0].size(); 
        
        if (K_A != K_B) {
            throw std::invalid_argument("Matrix dimension mismatch");
        }

        std::vector<float> flat_A, flat_B;
        for (const auto& row : A) {
            flat_A.insert(flat_A.end(), row.begin(), row.end());
        }
        for (const auto& row : B) {
            flat_B.insert(flat_B.end(), row.begin(), row.end());
        }

        std::vector<float> flat_C(M * N);

        cblas_sgemm(CblasRowMajor,  
                    CblasNoTrans,  
                    CblasNoTrans,  
                    M,           
                    N,             
                    K_A,           
                    1.0f,          
                    flat_A.data(),  
                    K_A,            
                    flat_B.data(),  
                    N,             
                    0.0f,         
                    flat_C.data(), 
                    N);           

        std::vector<std::vector<float>> C;
        C.reserve(M);
        for (size_t i = 0; i < M; ++i) {
            C.emplace_back(
                flat_C.begin() + i * N,
                flat_C.begin() + (i + 1) * N
            );
        }
        return C;
    }

    template <>
    std::vector<std::vector<double>> dot(std::vector<std::vector<double>> A, 
                                        std::vector<std::vector<double>> B) {
        const size_t M = A.size();   
        const size_t K_A = A[0].size(); 
        const size_t K_B = B.size();    
        const size_t N = B[0].size(); 
        
        if (K_A != K_B) {
            throw std::invalid_argument("Matrix dimension mismatch");
        }

        std::vector<double> flat_A, flat_B;
        for (const auto& row : A) {
            flat_A.insert(flat_A.end(), row.begin(), row.end());
        }
        for (const auto& row : B) {
            flat_B.insert(flat_B.end(), row.begin(), row.end());
        }

        std::vector<double> flat_C(M * N);

        cblas_dgemm(CblasRowMajor, 
                    CblasNoTrans,  
                    CblasNoTrans, 
                    M,            
                    N,           
                    K_A,          
                    1.0,           
                    flat_A.data(),  
                    K_A,            
                    flat_B.data(),  
                    N,              
                    0.0,          
                    flat_C.data(), 
                    N);         

        std::vector<std::vector<double>> C;
        C.reserve(M);
        for (size_t i = 0; i < M; ++i) {
            C.emplace_back(
                flat_C.begin() + i * N,
                flat_C.begin() + (i + 1) * N
            );
        }
        return C;
    }


    // transpose
    template <typename T>
    std::vector<std::vector<T>> transpose(std::vector<std::vector<T>> mat) {
        const size_t rows = mat.size();
        const size_t cols = mat[0].size();

        std::vector<float> float_mat(rows, std::vector<float>(cols));

        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                float_mat[i][j] = static_cast<float>(mat[i][j]);

        std::vector<float> flat_mat;
        flat_mat.reserve(rows * cols);

        for (const auto& row : float_mat) {
            flat_mat.insert(flat_mat.end(), row.begin(), row.end());
        }

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

        std::vector<std::vector<float>> float_result;
        result.reserve(cols);
        for (size_t i = 0; i < cols; ++i) {
            result.emplace_back(
                flat_result.begin() + i * rows,
                flat_result.begin() + (i + 1) * rows
            );
        }

        std::vector<std::vector<T>> result(cols, std::vector<T>(rows));
        
        for (size_t i = 0; i < cols; ++i)
            for (size_t j = 0; j < rows; ++j)
                result[i][j] = static_cast<T>(float_result[i][j]);

        return result;
    }

    template <>
    std::vector<std::vector<float>> transpose(std::vector<std::vector<float>> mat) {
        const size_t rows = mat.size();
        const size_t cols = mat[0].size();

        std::vector<float> flat_mat;
        flat_mat.reserve(rows * cols);

        for (const auto& row : mat) {
            flat_mat.insert(flat_mat.end(), row.begin(), row.end());
        }

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
    }

    template <>
    std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> mat) {
        const size_t rows = mat.size();
        const size_t cols = mat[0].size();

        std::vector<double> flat_mat;
        flat_mat.reserve(rows * cols);

        for (const auto& row : mat) {
            flat_mat.insert(flat_mat.end(), row.begin(), row.end());
        }

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
    }

    template <>
    std::vector<std::vector<char>> transpose(std::vector<std::vector<char>> mat) {
        size_t rows = mat.size();
        size_t cols = mat[0].size();

        std::vector<std::vector<int>> int_mat(rows, std::vector<int>(cols));

        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                int_mat[i][j] = static_cast<int>(mat[i][j]);

        auto int_result = transpose<int>(int_mat);

        std::vector<std::vector<char>> result(cols, std::vector<char>(rows));

        for (size_t i = 0; i < cols; ++i)
            for (size_t j = 0; j < rows; ++j)
                result[i][j] = static_cast<char>(int_result[i][j]);

        return result;
    }


    // add1
    template <typename T>
    std::vector<T> add1(std::vector<T> A, std::vector<T> B) {
        static_assert(!std::is_same_v<T, std::string>);
        static_assert(!std::is_same_v<T, char>);

        size_t N = A.size();

        if (N != B.size()) 
            throw std::invalid_argument("Vector dimension mismatch");

        std::vector<float> float_C(N);
        std::vector<float> float_A(N), float_B(N);

        for (size_t i = 0; i < N; ++i)
            float_A[i] = static_cast<float>(A[i]);

        for (size_t i = 0; i < N; ++i)
            float_B[i] = static_cast<float>(B[i]);

        cblas_scopy(N, float_B.data(), 1, float_C.data(), 1);
        cblas_saxpy(N, 1.0, float_A.data(), 1, float_C.data(), 1);

        std::vector<T> C(N);

        for (size_t i = 0; i < N; ++i)
            C[i] = static_cast<T>(float_C[i]);

        return C;
    }

    template <>
    std::vector<float> add1(std::vector<float> A, std::vector<float> B) {
        size_t N = A.size();

        if (N != B.size()) 
            throw std::invalid_argument("Vector dimension mismatch");

        std::vector<float> C(N);

        cblas_scopy(N, B.data(), 1, C.data(), 1);
        cblas_saxpy(N, 1.0, A.data(), 1, C.data(), 1);

        return C;
    }

    template <>
    std::vector<double> add1(std::vector<double> A, std::vector<double> B) {
        size_t N = A.size();

        if (N != B.size()) 
            throw std::invalid_argument("Vector dimension mismatch");

        std::vector<double> C(N);

        cblas_dcopy(N, B.data(), 1, C.data(), 1);
        cblas_daxpy(N, 1.0, A.data(), 1, C.data(), 1);

        return C;
    }


    // add2
    template <typename T>
    std::vector<std::vector<T>> add2(std::vector<std::vector<T>> A, std::vector<std::vector<T>> B) {
        static_assert(!std::is_same_v<T, char>);
        static_assert(!std::is_same_v<T, std::string>);

        size_t row_A = A.size();
        size_t col_A = A[0].size();
        size_t row_B = B.size();
        size_t col_B = B[0].size();

        if ((row_A != row_B) || (col_A != col_B))
            throw std::invalid_argument("Matrix dimension mismatch");

        std::vector<std::vector<float>> float_A(row_A, std::vector<float>(col_A));
        std::vector<std::vector<float>> float_B(row_B, std::vector<float>(col_B));

        for (size_t i = 0; i < row_A; ++i)
            for (size_t j = 0; j < col_A; ++j)
                float_A[i][j] = static_cast<float>(A[i][j]);

        for (size_t i = 0; i < row_B; ++i)
            for (size_t j = 0; j < col_B; ++j)
                float_B[i][j] = static_cast<float>(B[i][j]);

        std::vector<float> flat_A, flat_B;
        
        for (const auto& row : float_A)
            flat_A.insert(flat_A.end(), row.begin(), row.end());
        
        for (const auto& row : float_B)
            flat_B.insert(flat_B.end(), row.begin(), row.end());

        size_t N = row_A * col_A;
        std::vector<float> flat_C(N);

        cblas_scopy(N, flat_B.data(), 1, flat_C.data(), 1);
        cblas_saxpy(N, 1.0, flat_A.data(), 1, flat_C.data(), 1);

        std::vector<std::vector<float>> float_C;
        float_C.reserve(row_A);
        for (size_t i = 0; i < row_A; ++i) {
            float_C.emplace_back(
                flat_C.begin() + i * col_A,
                flat_C.begin() + (i + 1) * col_A
            );
        }

        std::vector<std::vector<T>> C(row_A, std::vector<T>(col_A));

        for (size_t i = 0; i < row_A; ++i)
            for (size_t j = 0; j < col_A; ++j)
                C[i][j] = static_cast<T>(float_C[i][j]);

        return C;
    }

    template <>
    std::vector<std::vector<float>> add2(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B) {
        size_t row_A = A.size();
        size_t col_A = A[0].size();
        size_t row_B = B.size();
        size_t col_B = B[0].size();

        if ((row_A != row_B) || (col_A != col_B))
            throw std::invalid_argument("Matrix dimension mismatch");

        std::vector<float> flat_A, flat_B;
        
        for (const auto& row : A)
            flat_A.insert(flat_A.end(), row.begin(), row.end());
        
        for (const auto& row : B)
            flat_B.insert(flat_B.end(), row.begin(), row.end());

        size_t N = row_A * col_A;
        std::vector<float> flat_C(N);

        cblas_scopy(N, flat_B.data(), 1, flat_C.data(), 1);
        cblas_saxpy(N, 1.0, flat_A.data(), 1, flat_C.data(), 1);

        std::vector<std::vector<float>> C;
        C.reserve(row_A);
        for (size_t i = 0; i < row_A; ++i) {
            C.emplace_back(
                flat_C.begin() + i * col_A,
                flat_C.begin() + (i + 1) * col_A
            );
        }

        return C;
    }

    template <>
    std::vector<std::vector<double>> add2(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B) {
        size_t row_A = A.size();
        size_t col_A = A[0].size();
        size_t row_B = B.size();
        size_t col_B = B[0].size();

        if ((row_A != row_B) || (col_A != col_B))
            throw std::invalid_argument("Matrix dimension mismatch");

        std::vector<double> flat_A, flat_B;
        
        for (const auto& row : A)
            flat_A.insert(flat_A.end(), row.begin(), row.end());
        
        for (const auto& row : B)
            flat_B.insert(flat_B.end(), row.begin(), row.end());

        size_t N = row_A * col_A;
        std::vector<double> flat_C(N);

        cblas_dcopy(N, flat_B.data(), 1, flat_C.data(), 1);
        cblas_daxpy(N, 1.0, flat_A.data(), 1, flat_C.data(), 1);

        std::vector<std::vector<double>> C;
        C.reserve(row_A);
        for (size_t i = 0; i < row_A; ++i) {
            C.emplace_back(
                flat_C.begin() + i * col_A,
                flat_C.begin() + (i + 1) * col_A
            );
        }

        return C;
    }


    // subtract1
    template <typename T>
    std::vector<T> subtract1(std::vector<T> A, std::vector<T> B) {
        static_assert(!std::is_same_v<T, char>);
        static_assert(!std::is_same_v<T, std::string>);

        size_t N = A.size();
        
        if (N != B.size()) 
            throw std::invalid_argument("Vector dimension mismatch");

        std::vector<float> float_A(N), float_B(N);

        for (size_t i = 0; i < N; ++i)
            float_A[i] = static_cast<float>(A[i]);

        for (size_t i = 0; i < N; ++i)
            float_B[i] = static_cast<float>(B[i]);

        std::vector<float> float_C(N);

        cblas_scopy(N, float_A.data(), 1, float_C.data(), 1);
        cblas_saxpy(N, -1.0, float_B.data(), 1, float_C.data(), 1);

        std::vector<T> C(N);

        for (size_t i = 0; i < N; ++i)
            C[i] = static_cast<T>(float_C[i]);

        return C;
    }

    template <>
    std::vector<float> subtract1(std::vector<float> A, std::vector<float> B) {
        size_t N = A.size();

        if (N != B.size()) 
            throw std::invalid_argument("Vector dimension mismatch");

        std::vector<float> C(N);

        cblas_scopy(N, A.data(), 1, C.data(), 1);
        cblas_saxpy(N, -1.0, B.data(), 1, C.data(), 1);

        return C;
    }

    template <>
    std::vector<double> subtract1(std::vector<double> A, std::vector<double> B) {
        size_t N = A.size();

        if (N != B.size()) 
            throw std::invalid_argument("Vector dimension mismatch");

        std::vector<double> C(N);

        cblas_dcopy(N, A.data(), 1, C.data(), 1);
        cblas_daxpy(N, -1.0, B.data(), 1, C.data(), 1);

        return C;
    }
}

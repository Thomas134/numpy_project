#include <gtest/gtest.h>
#include "../include/data_structure/ndarray.cpp"
#include "../include/matrix_operations.cpp"

template <typename T>
std::vector<std::vector<T>> manual_dot(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
    const size_t M = A.size();
    const size_t K = A[0].size();
    const size_t N = B[0].size();

    std::vector<std::vector<T>> C(M, std::vector<T>(N, 0));
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < K; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

TEST(NDArrayDotTest, TwoDimensionalDotTest) {
    std::vector<size_t> shapeA = {2, 2};
    std::vector<size_t> shapeB = {2, 2};
    ndarray<int> arrA(shapeA);
    ndarray<int> arrB(shapeB);

    std::vector<std::vector<int>> dataA = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> dataB = {{5, 6}, {7, 8}};
    arrA.assign(dataA);
    arrB.assign(dataB);

    ndarray<int> result = arrA.dot(arrB);

    std::vector<std::vector<int>> expected = manual_dot(dataA, dataB);
    std::vector<int> resultData = result.data();
    size_t index = 0;
    for (const auto& row : expected) {
        for (int value : row) {
            EXPECT_EQ(resultData[index++], value);
        }
    }
}

TEST(NDArrayDotTest, MismatchDimensionTest) {
    std::vector<size_t> shapeA = {2, 3};
    std::vector<size_t> shapeB = {2, 2};
    ndarray<int> arrA(shapeA);
    ndarray<int> arrB(shapeB);

    std::vector<std::vector<int>> dataA = {{1, 2, 3}, {4, 5, 6}};
    std::vector<std::vector<int>> dataB = {{5, 6}, {7, 8}};
    arrA.assign(dataA);
    arrB.assign(dataB);

    EXPECT_THROW(arrA.dot(arrB), std::invalid_argument);
}


template <typename T>
std::vector<T> manual_add_1d(const std::vector<T>& A, const std::vector<T>& B) {
    std::vector<T> result(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        result[i] = A[i] + B[i];
    }
    return result;
}

template <typename T>
std::vector<T> manual_sub_1d(const std::vector<T>& A, const std::vector<T>& B) {
    std::vector<T> result(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        result[i] = A[i] - B[i];
    }
    return result;
}

template <typename T>
std::vector<std::vector<T>> manual_add_2d(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
    std::vector<std::vector<T>> result(A.size(), std::vector<T>(A[0].size()));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

template <typename T>
std::vector<std::vector<T>> manual_sub_2d(const std::vector<std::vector<T>>& A, const std::vector<std::vector<T>>& B) {
    std::vector<std::vector<T>> result(A.size(), std::vector<T>(A[0].size()));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

TEST(NDArrayAddSubTest, OneDimensionalAddTest) {
    std::vector<size_t> shape = {3};
    ndarray<int> arrA(shape);
    ndarray<int> arrB(shape);
    std::vector<int> dataA = {1, 2, 3};
    std::vector<int> dataB = {4, 5, 6};
    arrA.assign(dataA);
    arrB.assign(dataB);

    ndarray<int> result = arrA.add(arrB);
    std::vector<int> expected = manual_add_1d(dataA, dataB);

    std::vector<int> resultData = result.data();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(resultData[i], expected[i]);
    }
}

TEST(NDArrayAddSubTest, OneDimensionalSubTest) {
    std::vector<size_t> shape = {3};
    ndarray<int> arrA(shape);
    ndarray<int> arrB(shape);
    std::vector<int> dataA = {4, 5, 6};
    std::vector<int> dataB = {1, 2, 3};
    arrA.assign(dataA);
    arrB.assign(dataB);

    ndarray<int> result = arrA.sub(arrB);
    std::vector<int> expected = manual_sub_1d(dataA, dataB);

    std::vector<int> resultData = result.data();
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(resultData[i], expected[i]);
    }
}

TEST(NDArrayAddSubTest, TwoDimensionalAddTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<int> arrA(shape);
    ndarray<int> arrB(shape);
    std::vector<std::vector<int>> dataA = {{1, 2}, {3, 4}};
    std::vector<std::vector<int>> dataB = {{5, 6}, {7, 8}};
    arrA.assign(dataA);
    arrB.assign(dataB);

    ndarray<int> result = arrA.add(arrB);
    std::vector<std::vector<int>> expected = manual_add_2d(dataA, dataB);

    std::vector<int> resultData = result.data();
    size_t index = 0;
    for (const auto& row : expected) {
        for (int value : row) {
            EXPECT_EQ(resultData[index++], value);
        }
    }
}

TEST(NDArrayAddSubTest, TwoDimensionalSubTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<int> arrA(shape);
    ndarray<int> arrB(shape);
    std::vector<std::vector<int>> dataA = {{5, 6}, {7, 8}};
    std::vector<std::vector<int>> dataB = {{1, 2}, {3, 4}};
    arrA.assign(dataA);
    arrB.assign(dataB);

    ndarray<int> result = arrA.sub(arrB);
    std::vector<std::vector<int>> expected = manual_sub_2d(dataA, dataB);

    std::vector<int> resultData = result.data();
    size_t index = 0;
    for (const auto& row : expected) {
        for (int value : row) {
            EXPECT_EQ(resultData[index++], value);
        }
    }
}

TEST(NDArrayAddSubTest, MismatchShapeAddTest) {
    std::vector<size_t> shapeA = {2, 2};
    std::vector<size_t> shapeB = {2, 3};
    ndarray<int> arrA(shapeA);
    ndarray<int> arrB(shapeB);

    EXPECT_THROW(arrA.add(arrB), std::invalid_argument);
}

TEST(NDArrayAddSubTest, MismatchShapeSubTest) {
    std::vector<size_t> shapeA = {2, 2};
    std::vector<size_t> shapeB = {2, 3};
    ndarray<int> arrA(shapeA);
    ndarray<int> arrB(shapeB);

    EXPECT_THROW(arrA.sub(arrB), std::invalid_argument);
}


template <typename T>
std::vector<std::vector<T>> manual_transpose(const std::vector<std::vector<T>>& mat) {
    const size_t rows = mat.size();
    const size_t cols = mat[0].size();
    std::vector<std::vector<T>> result(cols, std::vector<T>(rows));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

TEST(NDArrayTest, TransposeTwoDimensionalTest) {
    std::vector<size_t> shape = {2, 3};
    ndarray<int> arr(shape);
    std::vector<std::vector<int>> data = {{1, 2, 3}, {4, 5, 6}};
    arr.assign(data);

    try {
        ndarray<int> result = arr.transpose();

        std::vector<std::vector<int>> expected = manual_transpose(data);
        std::vector<size_t> expected_shape = {3, 2};

        EXPECT_EQ(result.shape(), expected_shape);

        for (size_t i = 0; i < expected_shape[0]; ++i) {
            for (size_t j = 0; j < expected_shape[1]; ++j) {
                std::vector<size_t> indices = {i, j};
                EXPECT_EQ(result(indices), expected[i][j]);
            }
        }
    } catch (const std::exception& e) {
        FAIL() << "Exception thrown: " << e.what();
    }
}

TEST(NDArrayTest, TransposeNonTwoDimensionalTest) {
    std::vector<size_t> shape = {3};
    ndarray<int> arr(shape);
    std::vector<int> data = {1, 2, 3};
    arr.assign(data);

    EXPECT_THROW(arr.transpose(), std::invalid_argument);
}
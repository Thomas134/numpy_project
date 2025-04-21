#include <gtest/gtest.h>
#include "../include/data_structure/ndarray.cpp"

TEST(NDArrayLogicalTest, LogicalAndTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<int> arr1(shape);
    ndarray<int> arr2(shape);

    std::vector<std::vector<int>> data1 = {{1, 0}, {1, 1}};
    std::vector<std::vector<int>> data2 = {{1, 1}, {0, 1}};

    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<int> result = arr1.logical_and(arr2);

    std::vector<std::vector<int>> expected = {{1, 0}, {0, 1}};
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(result({i, j}), expected[i][j]);
        }
    }
}

TEST(NDArrayLogicalTest, LogicalOrTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<int> arr1(shape);
    ndarray<int> arr2(shape);

    std::vector<std::vector<int>> data1 = {{1, 0}, {1, 1}};
    std::vector<std::vector<int>> data2 = {{1, 1}, {0, 1}};

    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<int> result = arr1.logical_or(arr2);

    std::vector<std::vector<int>> expected = {{1, 1}, {1, 1}};
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(result({i, j}), expected[i][j]);
        }
    }
}

TEST(NDArrayLogicalTest, LogicalXorTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<int> arr1(shape);
    ndarray<int> arr2(shape);

    std::vector<std::vector<int>> data1 = {{1, 0}, {1, 1}};
    std::vector<std::vector<int>> data2 = {{1, 1}, {0, 1}};

    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<int> result = arr1.logical_xor(arr2);

    std::vector<std::vector<int>> expected = {{0, 1}, {1, 0}};
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(result({i, j}), expected[i][j]);
        }
    }
}

TEST(NDArrayLogicalTest, LogicalAndNotTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<int> arr1(shape);
    ndarray<int> arr2(shape);

    std::vector<std::vector<int>> data1 = {{1, 0}, {1, 1}};
    std::vector<std::vector<int>> data2 = {{1, 1}, {0, 1}};

    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<int> result = arr1.logical_andnot(arr2);

    std::vector<std::vector<int>> expected = {{0, 1}, {0, 0}};
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(result({i, j}), expected[i][j]);
        }
    }
}

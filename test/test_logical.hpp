#include <gtest/gtest.h>
#include <random>
#include "../include/data_structure/ndarray.cpp"

TEST(NDArrayLogicalTest, LogicalAndTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<int> arr1(shape);
    ndarray<int> arr2(shape);

    std::vector<std::vector<int>> data1(1000, std::vector<int>(1000));
    std::vector<std::vector<int>> data2(1000, std::vector<int>(1000));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data1[i][j] = dis(gen);
            data2[i][j] = dis(gen);
        }
    }

    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<int> result = arr1.logical_and(arr2);

    std::vector<std::vector<int>> expected(1000, std::vector<int>(1000));
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            expected[i][j] = data1[i][j] && data2[i][j];
            EXPECT_EQ(result({i, j}), expected[i][j]);
        }
    }
}

TEST(NDArrayLogicalTest, LogicalOrTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<int> arr1(shape);
    ndarray<int> arr2(shape);

    std::vector<std::vector<int>> data1(1000, std::vector<int>(1000));
    std::vector<std::vector<int>> data2(1000, std::vector<int>(1000));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data1[i][j] = dis(gen);
            data2[i][j] = dis(gen);
        }
    }

    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<int> result = arr1.logical_or(arr2);

    std::vector<std::vector<int>> expected(1000, std::vector<int>(1000));
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            expected[i][j] = data1[i][j] || data2[i][j];
            EXPECT_EQ(result({i, j}), expected[i][j]);
        }
    }
}

TEST(NDArrayLogicalTest, LogicalXorTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<int> arr1(shape);
    ndarray<int> arr2(shape);

    std::vector<std::vector<int>> data1(1000, std::vector<int>(1000));
    std::vector<std::vector<int>> data2(1000, std::vector<int>(1000));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data1[i][j] = dis(gen);
            data2[i][j] = dis(gen);
        }
    }

    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<int> result = arr1.logical_xor(arr2);

    std::vector<std::vector<int>> expected(1000, std::vector<int>(1000));
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            expected[i][j] = (data1[i][j] && !data2[i][j]) || (!data1[i][j] && data2[i][j]);
            EXPECT_EQ(result({i, j}), expected[i][j]);
        }
    }
}

TEST(NDArrayLogicalTest, LogicalAndNotTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<int> arr1(shape);
    ndarray<int> arr2(shape);

    std::vector<std::vector<int>> data1(1000, std::vector<int>(1000));
    std::vector<std::vector<int>> data2(1000, std::vector<int>(1000));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data1[i][j] = dis(gen);
            data2[i][j] = dis(gen);
        }
    }

    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<int> result = arr1.logical_andnot(arr2);

    std::vector<std::vector<int>> expected(1000, std::vector<int>(1000));
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            expected[i][j] = !data1[i][j] && data2[i][j];
            EXPECT_EQ(result({i, j}), expected[i][j]);
        }
    }
}
    
#include <gtest/gtest.h>
#include "../include/data_structure/ndarray.cpp"

TEST(NDArrayShiftTest, Slli1DTest) {
    std::vector<size_t> shape = {4};
    ndarray<int> arr(shape);
    std::vector<int> data = {1, 2, 3, 4};
    arr.assign(data);

    int imm = 2;
    ndarray<int> result = arr.slli(imm);
    std::vector<int> resultData = result.data();

    for (size_t i = 0; i < data.size(); ++i) {
        int expected = data[i] << imm;
        EXPECT_EQ(resultData[i], expected);
    }
}

TEST(NDArrayShiftTest, Srli1DTest) {
    std::vector<size_t> shape = {4};
    ndarray<int> arr(shape);
    std::vector<int> data = {4, 8, 12, 16};
    arr.assign(data);

    int imm = 2;
    ndarray<int> result = arr.srli(imm);
    std::vector<int> resultData = result.data();

    for (size_t i = 0; i < data.size(); ++i) {
        int expected = data[i] >> imm;
        EXPECT_EQ(resultData[i], expected);
    }
}

TEST(NDArrayShiftTest, Slli2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<int> arr(shape);
    std::vector<std::vector<int>> data = {{1, 2}, {3, 4}};
    arr.assign(data);

    int imm = 2;
    ndarray<int> result = arr.slli(imm);
    std::vector<int> resultData = result.data();

    size_t index = 0;
    for (const auto& row : data) {
        for (int value : row) {
            int expected = value << imm;
            EXPECT_EQ(resultData[index++], expected);
        }
    }
}

TEST(NDArrayShiftTest, Srli2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<int> arr(shape);
    std::vector<std::vector<int>> data = {{4, 8}, {12, 16}};
    arr.assign(data);

    int imm = 2;
    ndarray<int> result = arr.srli(imm);
    std::vector<int> resultData = result.data();

    size_t index = 0;
    for (const auto& row : data) {
        for (int value : row) {
            int expected = value >> imm;
            EXPECT_EQ(resultData[index++], expected);
        }
    }
}
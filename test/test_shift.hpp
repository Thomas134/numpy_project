#include <gtest/gtest.h>
#include "../include/data_structure/ndarray.cpp"
#include <random>

int generateRandomInt(int min, int max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

TEST(NDArrayShiftTest, Slli1DTest) {
    std::vector<size_t> shape = {1000};
    ndarray<int> arr(shape);
    std::vector<int> data;

    for (size_t i = 0; i < shape[0]; ++i) {
        data.push_back(generateRandomInt(1, 10));
    }
    arr.assign(data);
    
    constexpr int8_t imm = 2;
    ndarray<int> result = arr.slli(2);
    std::vector<int> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < data.size(); ++i) {
        int expected = data[i] << imm;
        EXPECT_EQ(resultData[i], expected);
    }
}

TEST(NDArrayShiftTest, Srli1DTest) {
    std::vector<size_t> shape = {1000};
    ndarray<int> arr(shape);
    std::vector<int> data;

    for (size_t i = 0; i < shape[0]; ++i) {
        data.push_back(generateRandomInt(4, 20));
    }
    arr.assign(data);
    
    constexpr int8_t imm = 2;
    ndarray<int> result = arr.srli(2);
    std::vector<int> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < data.size(); ++i) {
        int expected = data[i] >> imm;
        EXPECT_EQ(resultData[i], expected);
    }
}

TEST(NDArrayShiftTest, Slli2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<int> arr(shape);
    std::vector<std::vector<int>> data(shape[0], std::vector<int>(shape[1]));

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = generateRandomInt(1, 10);
        }
    }
    arr.assign(data);

    constexpr int8_t imm = 2;
    ndarray<int> result = arr.slli(2);
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
    std::vector<size_t> shape = {1000, 1000};
    ndarray<int> arr(shape);
    std::vector<std::vector<int>> data(shape[0], std::vector<int>(shape[1]));

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = generateRandomInt(4, 20);
        }
    }
    arr.assign(data);

    constexpr int8_t imm = 2;
    ndarray<int> result = arr.srli(2);
    std::vector<int> resultData = result.data();

    size_t index = 0;
    for (const auto& row : data) {
        for (int value : row) {
            int expected = value >> imm;
            EXPECT_EQ(resultData[index++], expected);
        }
    }
}
#include <gtest/gtest.h>
#include "../include/data_structure/ndarray.cpp"

template <typename T>
T multiply_by_two(T x) {
    return x * 2;
}

template <typename T>
T add_one(T x) {
    return x + 1;
}

TEST(NDArrayApplyTest, Apply1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<int> arr(shape);
    std::vector<int> data = {1, 2, 3};
    arr.assign(data);

    auto result1 = arr.apply(multiply_by_two<int>);
    std::vector<int> resultData1 = result1.data();
    for (size_t i = 0; i < data.size(); ++i) {
        int expected = multiply_by_two(data[i]);
        EXPECT_EQ(resultData1[i], expected);
    }

    auto result2 = arr.apply(add_one<int>);
    std::vector<int> resultData2 = result2.data();
    for (size_t i = 0; i < data.size(); ++i) {
        int expected = add_one(data[i]);
        EXPECT_EQ(resultData2[i], expected);
    }
}

TEST(NDArrayApplyTest, Apply2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<int> arr(shape);
    std::vector<std::vector<int>> data = {{1, 2}, {3, 4}};
    arr.assign(data);

    auto result1 = arr.apply(multiply_by_two<int>);
    std::vector<int> resultData1 = result1.data();
    size_t index = 0;
    for (const auto& row : data) {
        for (int value : row) {
            int expected = multiply_by_two(value);
            EXPECT_EQ(resultData1[index++], expected);
        }
    }

    auto result2 = arr.apply(add_one<int>);
    std::vector<int> resultData2 = result2.data();
    index = 0;
    for (const auto& row : data) {
        for (int value : row) {
            int expected = add_one(value);
            EXPECT_EQ(resultData2[index++], expected);
        }
    }
}
#include <gtest/gtest.h>

#include "../include/data_structure/ndarray.cpp"

TEST(NDArrayTest, ConstructorTest) {
    std::vector<size_t> shape = {2, 3};
    ndarray<int> arr(shape);

    EXPECT_EQ(arr.ndim(), 2);
    EXPECT_EQ(arr.size(), 6);
    EXPECT_EQ(arr.shape().size(), 2);
    EXPECT_EQ(arr.shape()[0], 2);
    EXPECT_EQ(arr.shape()[1], 3);
}

TEST(NDArrayTest, AssignOneDimensionalDataTest) {
    std::vector<size_t> shape = {5};
    ndarray<int> arr(shape);
    std::vector<int> data = {1, 2, 3, 4, 5};

    arr.assign(data);

    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_EQ(arr.data()[i], data[i]);
}

TEST(NDArrayTest, AssignTwoDimensionalDataTest) {
    std::vector<size_t> shape = {2, 3};
    ndarray<int> arr(shape);
    std::vector<std::vector<int>> data = {{1, 2, 3}, {4, 5, 6}};

    arr.assign(data);

    size_t index = 0;
    for (const auto& row : data)
        for (int value : row)
            EXPECT_EQ(arr.data()[index++], value);
}

TEST(NDArrayTest, OperatorOutputOneDimensionalTest) {
    std::vector<size_t> shape = {3};
    ndarray<int> arr(shape);
    std::vector<int> data = {1, 2, 3};
    arr.assign(data);

    std::ostringstream oss;
    oss << arr;
    std::string expected = "[1, 2, 3]";
    EXPECT_EQ(oss.str(), expected);
}

TEST(NDArrayTest, OperatorOutputTwoDimensionalTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<int> arr(shape);
    std::vector<std::vector<int>> data = {{1, 2}, {3, 4}};
    arr.assign(data);

    std::ostringstream oss;
    oss << arr;
    std::string expected = "[[1, 2],\n [3, 4]]";
    EXPECT_EQ(oss.str(), expected);
}

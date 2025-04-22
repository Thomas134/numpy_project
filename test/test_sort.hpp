#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include "../include/data_structure/ndarray.cpp"

TEST(NDArraySortTest, SortWithDefaultComparator) {
    std::vector<size_t> shape = {10000};
    ndarray<int> arr(shape);
    std::vector<int> data(10000);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dis(gen);
    }
    arr.assign(data);

    ndarray<int> sortedArr = arr.sort(std::less<int>{});

    std::vector<int> sortedData = sortedArr.data();
    std::sort(data.begin(), data.end());

    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_EQ(sortedData[i], data[i]);
}

TEST(NDArraySortTest, SortWithCustomComparator) {
    std::vector<size_t> shape = {10000};
    ndarray<int> arr(shape);
    std::vector<int> data(10000);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dis(gen);
    }
    arr.assign(data);

    ndarray<int> sortedArr = arr.sort(std::greater<int>{});
    std::sort(data.begin(), data.end(), std::greater<int>{});
    std::vector<int> sortedData = sortedArr.data();

    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_EQ(sortedData[i], data[i]);
}
    
#include <gtest/gtest.h>
#include <algorithm>
#include "../include/data_structure/ndarray.cpp"

TEST(NDArraySortTest, SortWithDefaultComparator) {
    std::vector<size_t> shape = {5};
    ndarray<int> arr(shape);
    std::vector<int> data = {5, 3, 1, 4, 2};
    arr.assign(data);

    ndarray<int> sortedArr = arr.sort(std::less<int>{});

    std::vector<int> sortedData = sortedArr.data();
    std::sort(data.begin(), data.end());

    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_EQ(sortedData[i], data[i]);
}

TEST(NDArraySortTest, SortWithCustomComparator) {
    std::vector<size_t> shape = {5};
    ndarray<int> arr(shape);
    std::vector<int> data = {1, 2, 3, 4, 5};
    arr.assign(data);

    ndarray<int> sortedArr = arr.sort(std::greater<int>{});
    std::sort(data.begin(), data.end(), std::greater<int>{});
    std::vector<int> sortedData = sortedArr.data();

    for (size_t i = 0; i < data.size(); ++i)
        EXPECT_EQ(sortedData[i], data[i]);
}

#ifndef SORT_HPP
#define SORT_HPP

#include <vector>

#include <boost/sort/pdqsort/pdqsort.hpp>
#include <algorithm>

template <typename T>
struct CompareRows {
    bool operator()(const std::vector<T>& row1, const std::vector<T>& row2) const {
        return row1[0] < row2[0];
    }
};

namespace internal {
    // sort1
    template <typename T, typename Compare>
    void sort1(std::vector<T>& A, Compare comp = std::less<T>{});


    // sort2
    template <typename T, typename Compare>
    void sort2(std::vector<std::vector<T>>& A, Compare comp = CompareRows<T>{});
}

namespace internal {
    template <typename T, typename Compare>
    void sort1(std::vector<T>& A, Compare comp) {
        if (A.size() < 8192) {
            std::sort(A.begin(), A.end(), comp);
        } else {
            boost::sort::pdqsort(A.begin(), A.end(), comp);
        }
    }


    template <typename T, typename Compare>
    void sort2(std::vector<std::vector<T>>& A, Compare comp) {
        for (auto& innerVector : A) {
            if (innerVector.size() < 8192) {
                std::sort(innerVector.begin(), innerVector.end(), comp);
            } else {
                boost::sort::pdqsort(innerVector.begin(), innerVector.end(), comp);
            }
        }
    }
}

#endif

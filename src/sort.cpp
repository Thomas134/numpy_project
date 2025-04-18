#include "../include/sort.hpp"
#include <boost/sort/pdqsort/pdqsort.hpp>
#include <algorithm>


namespace internal {
    template <typename T, typename Compare>
    void sort1(std::vector<T>& A, Compare comp = Compare()) {
        if (A.size() < 8192) {
            std::sort(A.begin(), A.end(), comp);
        } else {
            boost::sort::pdqsort(A.begin(), A.end(), comp);
        }
    }


    template <typename T, typename Compare>
    void sort2(std::vector<std::vector<T>>& A, Compare comp = Compare()) {
        for (auto& innerVector : A) {
            if (innerVector.size() < 8192) {
                std::sort(innerVector.begin(), innerVector.end(), comp);
            } else {
                boost::sort::pdqsort(innerVector.begin(), innerVector.end(), comp);
            }
        }
    }
}
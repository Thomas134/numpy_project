#ifndef SORT_HPP
#define SORT_HPP

#include <vector>

namespace internal {
    // sort1
    template <typename T, typename Compare = std::less<T>>
    void sort1(std::vector<T>& A, Compare comp = Compare());


    // sort2
    template <typename T, typename Compare = std::less<T>>
    void sort2(std::vector<std::vector<T>>& A, Compare comp = Compare());
}

#endif

#ifndef SORT_HPP
#define SORT_HPP

#include <vector>

namespace internal {
    // sort1
    template <typename T, typename Compare>
    void sort1(std::vector<T>& A, Compare comp = std::less<T>{});


    // sort2
    template <typename T, typename Compare>
    void sort2(std::vector<std::vector<T>>& A, Compare comp = std::less<T>{});
}

#endif

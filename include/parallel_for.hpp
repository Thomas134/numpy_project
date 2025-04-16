#ifndef PARALLEL_FOR_HPP
#define PARALLEL_FOR_HPP

#include <vector>

namespace internal {
    // apply1
    template <typename T, typename Func>
    void apply1(std::vector<T>& A, Func func);


    // apply2
    template <typename T, typename Func>
    void apply2(std::vector<std::vector<T>>& A, Func func);
}

#endif

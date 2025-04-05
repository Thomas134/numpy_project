#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // 使用 std::for_each 将所有元素变为负数
    std::for_each(numbers.begin(), numbers.end(), [](int& num) {
        num = -num;
    });

    // 输出结果
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    // ʹ�� std::for_each ������Ԫ�ر�Ϊ����
    std::for_each(numbers.begin(), numbers.end(), [](int& num) {
        num = -num;
    });

    // ������
    for (int num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    return 0;
}
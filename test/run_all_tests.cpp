#include "test_apply.hpp"
#include "test_basic_property.hpp"
#include "test_logical.hpp"
#include "test_math.hpp"
#include "test_matrix_operations.hpp"
#include "test_shift.hpp"
#include "test_sort.hpp"


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
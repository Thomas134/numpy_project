gtest_dep = dependency('gtest', required: true)

blas_dep = dependency('blas', required: true)

openmp_dep = dependency('openmp', required: true)

test_sources = files(
  'test_apply.hpp',
  'test_basic_property.hpp',
  'test_logical.hpp',
  'test_math.hpp',
  'test_matrix_operations.hpp',
  'test_shift.hpp',
  'test_sort.hpp',
  'run_all_tests.cpp'
)

test_exe = executable(
  'run_all_tests',
  test_sources,
  include_directories: include_dirs,
  dependencies: [gtest_dep, blas_dep, openmp_dep]
)

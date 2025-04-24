project('numpy_project', 'cpp',
  version : '1.0',
  default_options : ['cpp_std=c++17', 'cpp_args=-mavx2 -fopenmp -O3']
)

blas_dep = dependency('blas', required : true)

openmp_dep = dependency('openmp', required : true)

include_dirs = include_directories('include', 'include/utils', 'include/data_structure')

sources = files(
  'include/simd_traits.cpp',
  'include/math.cpp',
  'include/logical.cpp',
  'include/matrix_operations.cpp',
  'include/xsimd_traits.cpp',
  'include/shift.cpp',
  'include/sort.cpp',
  'include/parallel_for.cpp',
  'include/utils/simd_operators.cpp',
  'include/utils/utils.cpp',
  'include/data_structure/dtype_trait.cpp',
  'include/data_structure/ndarray.cpp'
)

numpycpp_lib = static_library('numpycpp',
  sources,
  include_directories : [include_dirs],
  dependencies : [blas_dep, openmp_dep],
  install : true,
  install_dir : '/usr/local/lib'
)

install_headers('include/simd_traits.cpp', 
'include/math.cpp', 
'include/logical.cpp', 
'include/matrix_operations.cpp', 
'include/xsimd_traits.cpp', 
'include/shift.cpp', 
'include/sort.cpp', 
'include/parallel_for.cpp', 
subdir : 'numpy')


install_headers('include/utils/simd_operators.cpp', 
  'include/utils/utils.cpp', 
  subdir : 'numpy/utils'
)

install_headers('include/data_structure/dtype_trait.cpp', 
  'include/data_structure/ndarray.cpp', 
  subdir : 'numpy/data_structure'
)

subdir('test')

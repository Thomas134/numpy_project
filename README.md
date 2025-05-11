# Project Overview
This is a small-scale Numpy implemented in C++, and it is also my final year project. The project supports multiple operating systems distributions, includes multiple CI pipelines, 
and has also been ported to the RISC-V architecture. In this project, I utilized SIMD, OpenMP and CBlas technologies to implement numerous matrix operations similar to those in Numpy.
On the RISC-V architecture, since OpenEuler RISC-V currently does not support the C language RVV interface, for the SIMD portion, I switched to using native loops for implementation and
then enabled the compilation options for the RVV optimization. As for CBlas and OpenMP, since they have already been ported to OpenEuler RISC-V, they can be used normally.

# Main Directory Structure
| **File**                 | **Description**                                |
|--------------------------|------------------------------------------------|
| `ndarray.hpp`            | Implementation of major data structures.       |
| `logical.hpp`            | Implementation of logical functions.           |
| `math.cpp`               | Implementation of math functions.              |
| `matrix_operations.hpp`  | Implementation of matrix operations in CBlas.  |
| `parallel_for.hpp`       | Implementation of for loop in OpenMP mode.     |
| `shift.hpp`              | Implementation of bitwise operations.          |
| `sort.hpp`               | Implementation of sort functions.             |
| `simd_traits.hpp`        | To encapsulate the SIMD interface.            |
| `xsimd_traits.hpp`       | To encapsulate the XSIMD interface.           |
| `dtype_traits.hpp`       | For Numpy type-traits.                        |

# Build the Project
You should install the OpenBlas, OpenMP and SIMD header files first. On Ubuntu 24.04, you can do the following operations.<br>
```bash
sudo apt-get install -y cmake build-essential libopenblas-dev libxsimd-dev libboost-all-dev
```
For other distributions, you can refer to the build method in the `.github/workflows/ci.yml` file. Currently, it supports Fedora 36-42, Kali Linux, Arch Linux, Ubuntu 24.04, OpenEuler RISC-V, and debian:bookworm.
And it also provides building methods for both CMake and Meson.<br><br>
To build the project
```bash
mkdir build && cd build && cmake .. && make -j$(nproc)
```

# Run the Test
The testcases are put in the `test` directory. After building the project, you can run `run_all_tests` in the test directory.
```bash
cd test && ./run_all_tests
```
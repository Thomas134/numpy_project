#include <gtest/gtest.h>
#include "../include/data_structure/ndarray.cpp"

TEST(NDArrayMathTest, Min1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr1(shape);
    ndarray<float> arr2(shape);
    std::vector<float> data1 = {1.0f, 3.0f, 5.0f};
    std::vector<float> data2 = {2.0f, 2.0f, 6.0f};
    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<float> result = arr1.min(arr2);
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::min(data1[i], data2[i]));
    }
}

TEST(NDArrayMathTest, Min2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr1(shape);
    ndarray<float> arr2(shape);
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<std::vector<float>> data2 = {{2.0f, 1.0f}, {4.0f, 3.0f}};
    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<float> result = arr1.min(arr2);
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::min(data1[i][j], data2[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Max1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr1(shape);
    ndarray<float> arr2(shape);
    std::vector<float> data1 = {1.0f, 3.0f, 5.0f};
    std::vector<float> data2 = {2.0f, 2.0f, 6.0f};
    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<float> result = arr1.max(arr2);
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::max(data1[i], data2[i]));
    }
}

TEST(NDArrayMathTest, Max2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr1(shape);
    ndarray<float> arr2(shape);
    std::vector<std::vector<float>> data1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<std::vector<float>> data2 = {{2.0f, 1.0f}, {4.0f, 3.0f}};
    arr1.assign(data1);
    arr2.assign(data2);

    ndarray<float> result = arr1.max(arr2);
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::max(data1[i][j], data2[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Sqrt1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {1.0f, 4.0f, 9.0f};
    arr.assign(data);

    ndarray<float> result = arr.sqrt();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::sqrt(data[i]));
    }
}

TEST(NDArrayMathTest, Sqrt2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{1.0f, 4.0f}, {9.0f, 16.0f}};
    arr.assign(data);

    ndarray<float> result = arr.sqrt();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::sqrt(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Abs1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {-1.0f, 2.0f, -3.0f};
    arr.assign(data);

    ndarray<float> result = arr.abs();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::abs(data[i]));
    }
}

TEST(NDArrayMathTest, Abs2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{-1.0f, 2.0f}, {-3.0f, 4.0f}};
    arr.assign(data);

    ndarray<float> result = arr.abs();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::abs(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Rsqrt1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {1.0f, 4.0f, 9.0f};
    arr.assign(data);

    ndarray<float> result = arr.rsqrt();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], 1 / std::sqrt(data[i]));
    }
}

TEST(NDArrayMathTest, Rsqrt2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{1.0f, 4.0f}, {9.0f, 16.0f}};
    arr.assign(data);

    ndarray<float> result = arr.rsqrt();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], 1 / std::sqrt(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Ceil1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {1.2f, 2.5f, 3.7f};
    arr.assign(data);

    ndarray<float> result = arr.ceil();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::ceil(data[i]));
    }
}

TEST(NDArrayMathTest, Ceil2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{1.2f, 2.5f}, {3.7f, 4.1f}};
    arr.assign(data);

    ndarray<float> result = arr.ceil();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::ceil(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Floor1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {1.2f, 2.5f, 3.7f};
    arr.assign(data);

    ndarray<float> result = arr.floor();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::floor(data[i]));
    }
}

TEST(NDArrayMathTest, Floor2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{1.2f, 2.5f}, {3.7f, 4.1f}};
    arr.assign(data);

    ndarray<float> result = arr.floor();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::floor(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Log1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    arr.assign(data);

    ndarray<float> result = arr.log();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::log(data[i]));
    }
}

TEST(NDArrayMathTest, Log2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    arr.assign(data);

    ndarray<float> result = arr.log();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::log(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Log2_1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {1.0f, 2.0f, 4.0f};
    arr.assign(data);

    ndarray<float> result = arr.log2();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::log2(data[i]));
    }
}

TEST(NDArrayMathTest, Log2_2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{1.0f, 2.0f}, {4.0f, 8.0f}};
    arr.assign(data);

    ndarray<float> result = arr.log2();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::log2(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Log10_1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {1.0f, 10.0f, 100.0f};
    arr.assign(data);

    ndarray<float> result = arr.log10();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::log10(data[i]));
    }
}

TEST(NDArrayMathTest, Log10_2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{1.0f, 10.0f}, {100.0f, 1000.0f}};
    arr.assign(data);

    ndarray<float> result = arr.log10();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::log10(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Sin1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {0.0f, 3.14159f / 2, 3.14159f};
    arr.assign(data);

    ndarray<float> result = arr.sin();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::sin(data[i]));
    }
}

TEST(NDArrayMathTest, Sin2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{0.0f, 3.14159f / 2}, {3.14159f, 3.14159f * 3 / 2}};
    arr.assign(data);

    ndarray<float> result = arr.sin();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::sin(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Cos1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {0.0f, 3.14159f / 2, 3.14159f};
    arr.assign(data);

    ndarray<float> result = arr.cos();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::cos(data[i]));
    }
}

TEST(NDArrayMathTest, Cos2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{0.0f, 3.14159f / 2}, {3.14159f, 3.14159f * 3 / 2}};
    arr.assign(data);

    ndarray<float> result = arr.cos();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::cos(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Tan1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {0.0f, 3.14159f / 4, 3.14159f / 3};
    arr.assign(data);

    ndarray<float> result = arr.tan();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::tan(data[i]));
    }
}

TEST(NDArrayMathTest, Tan2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{0.0f, 3.14159f / 4}, {3.14159f / 3, 3.14159f / 6}};
    arr.assign(data);

    ndarray<float> result = arr.tan();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_EQ(resultData[index++], std::tan(data[i][j]));
        }
    }
}

TEST(NDArrayMathTest, Asin1DTest) {
    std::vector<size_t> shape = {3};
    ndarray<float> arr(shape);
    std::vector<float> data = {0.0f, 0.5f, 1.0f};
    arr.assign(data);

    ndarray<float> result = arr.asin();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_EQ(resultData[i], std::asin(data[i]));
    }
}

TEST(NDArrayMathTest, Asin2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{0.0f, 0.5f}, {0.707f, 1.0f}};
    arr.assign(data);

    ndarray<float> result = arr.asin();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            float expected = std::asin(data[i][j]);
            EXPECT_NEAR(resultData[index++], expected, 1e-5);
        }
    }
}

TEST(NDArrayMathTest, Acos1DTest) {
    std::vector<size_t> shape = {4};
    ndarray<float> arr(shape);
    std::vector<float> data = {0.0f, 0.5f, 0.707f, 1.0f};
    arr.assign(data);

    ndarray<float> result = arr.acos();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < data.size(); ++i) {
        float expected = std::acos(data[i]);
        EXPECT_NEAR(resultData[i], expected, 1e-5);
    }
}

TEST(NDArrayMathTest, Acos2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{0.0f, 0.5f}, {0.707f, 1.0f}};
    arr.assign(data);

    ndarray<float> result = arr.acos();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            float expected = std::acos(data[i][j]);
            EXPECT_NEAR(resultData[index], expected, 1e-5);
            index++;
        }
    }
}

TEST(NDArrayMathTest, Atan2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{0.0f, 0.5f}, {0.707f, 1.0f}};
    arr.assign(data);

    ndarray<float> result = arr.atan();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            float expected = std::atan(data[i][j]);
            EXPECT_NEAR(resultData[index], expected, 1e-5);
            index++;
        }
    }
}


TEST(NDArrayMathTest, Atan1DTest) {
    std::vector<size_t> shape = {4};
    ndarray<float> arr(shape);
    std::vector<float> data = {0.0f, 0.5f, 0.707f, 1.0f};
    arr.assign(data);

    ndarray<float> result = arr.atan();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < data.size(); ++i) {
        float expected = std::atan(data[i]);
        EXPECT_NEAR(resultData[i], expected, 1e-5);
    }
}

// TEST(NDArrayMathTest, Sincos1DTest) {
//     std::vector<size_t> shape = {4};
//     ndarray<float> arr(shape);
//     std::vector<float> data = {0.0f, 0.5f, 0.707f, 1.0f};
//     arr.assign(data);

//     ndarray<float> result = arr.sincos();
//     std::vector<float> resultData = result.data();

//     for (size_t i = 0; i < data.size(); ++i) {
//         EXPECT_NEAR(resultData[i], std::sin(data[i]), 1e-5);
//     }
// }

// TEST(NDArrayMathTest, Sincos2DTest) {
//     std::vector<size_t> shape = {2, 2};
//     ndarray<float> arr(shape);
//     std::vector<std::vector<float>> data = {{0.0f, 0.5f}, {0.707f, 1.0f}};
//     arr.assign(data);

//     ndarray<float> result = arr.sincos();
//     std::vector<float> resultData = result.data();

//     size_t index = 0;
//     for (size_t i = 0; i < shape[0]; ++i) {
//         for (size_t j = 0; j < shape[1]; ++j) {
//             EXPECT_NEAR(resultData[index++], std::sin(data[i][j]), 1e-5);
//         }
//     }
// }

TEST(NDArrayMathTest, Round1DTest) {
    std::vector<size_t> shape = {4};
    ndarray<float> arr(shape);
    std::vector<float> data = {0.1f, 0.5f, 1.7f, 2.2f};
    arr.assign(data);

    ndarray<float> result = arr.round();
    std::vector<float> resultData = result.data();

    for (size_t i = 0; i < data.size(); ++i) {
        float expected = std::round(data[i]);
        EXPECT_NEAR(resultData[i], expected, 1e-5);
    }
}

TEST(NDArrayMathTest, Round2DTest) {
    std::vector<size_t> shape = {2, 2};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data = {{0.1f, 0.5f}, {1.7f, 2.2f}};
    arr.assign(data);

    ndarray<float> result = arr.round();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            float expected = std::round(data[i][j]);
            EXPECT_NEAR(resultData[index], expected, 1e-5);
            index++;
        }
    }
}
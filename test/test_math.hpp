#include <gtest/gtest.h>
#include <random>
#include "../include/data_structure/ndarray.cpp"

TEST(NDArrayMathTest, Min1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr1(shape);
    ndarray<float> arr2(shape);
    std::vector<float> data1(shape[0]);
    std::vector<float> data2(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 10.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data1[i] = dis(gen);
        data2[i] = dis(gen);
    }
    
    arr1.assign(data1);
    arr2.assign(data2);
    
    ndarray<float> result = arr1.min(arr2);
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::min(data1[i], data2[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Min2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr1(shape);
    ndarray<float> arr2(shape);
    std::vector<std::vector<float>> data1(shape[0], std::vector<float>(shape[1]));
    std::vector<std::vector<float>> data2(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 10.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data1[i][j] = dis(gen);
            data2[i][j] = dis(gen);
        }
    }
    
    arr1.assign(data1);
    arr2.assign(data2);
    
    ndarray<float> result = arr1.min(arr2);
    std::vector<float> resultData = result.data();
    
    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::min(data1[i][j], data2[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Max1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr1(shape);
    ndarray<float> arr2(shape);
    std::vector<float> data1(shape[0]);
    std::vector<float> data2(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 10.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data1[i] = dis(gen);
        data2[i] = dis(gen);
    }
    
    arr1.assign(data1);
    arr2.assign(data2);
    
    ndarray<float> result = arr1.max(arr2);
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::max(data1[i], data2[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Max2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr1(shape);
    ndarray<float> arr2(shape);
    std::vector<std::vector<float>> data1(shape[0], std::vector<float>(shape[1]));
    std::vector<std::vector<float>> data2(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 10.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data1[i][j] = dis(gen);
            data2[i][j] = dis(gen);
        }
    }
    
    arr1.assign(data1);
    arr2.assign(data2);
    
    ndarray<float> result = arr1.max(arr2);
    std::vector<float> resultData = result.data();
    
    size_t index = 0;

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::max(data1[i][j], data2[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Sqrt1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.sqrt();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::sqrt(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Sqrt2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.sqrt();
    std::vector<float> resultData = result.data();
    
    size_t index = 0;

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::sqrt(data[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Abs1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.abs();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::abs(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Abs2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-100.0f, 100.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.abs();
    std::vector<float> resultData = result.data();
    
    size_t index = 0;

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::abs(data[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Rsqrt1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.rsqrt();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], 1 / std::sqrt(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Rsqrt2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.rsqrt();
    std::vector<float> resultData = result.data();
    
    size_t index = 0;

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], 1 / std::sqrt(data[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Ceil1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 10.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.ceil();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::ceil(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Ceil2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 10.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }

    arr.assign(data);

    ndarray<float> result = arr.ceil();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::ceil(data[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Floor1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 10.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.floor();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::floor(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Floor2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 10.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.floor();
    std::vector<float> resultData = result.data();
    
    size_t index = 0;

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::floor(data[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Log1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.log();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::log(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Log2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.log();
    std::vector<float> resultData = result.data();
    
    size_t index = 0;

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::log(data[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Log2_1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.log2();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::log2(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Log2_2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 100.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }

    arr.assign(data);

    ndarray<float> result = arr.log2();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::log2(data[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Log10_1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 1000.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.log10();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::log10(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Log10_2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1.0f, 1000.0f);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }

    arr.assign(data);

    ndarray<float> result = arr.log10();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::log10(data[i][j]), 1e-3);
        }
    }
}

const float TWO_PI = 2 * 3.14159f; 

TEST(NDArrayMathTest, Sin1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-TWO_PI, TWO_PI);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.sin();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::sin(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Sin2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-TWO_PI, TWO_PI);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }

    arr.assign(data);

    ndarray<float> result = arr.sin();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::sin(data[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Cos1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-TWO_PI, TWO_PI);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.cos();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::cos(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Cos2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-TWO_PI, TWO_PI);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }

    arr.assign(data);

    ndarray<float> result = arr.cos();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::cos(data[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Tan1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-TWO_PI + 0.1f, TWO_PI - 0.1f);

    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }
    
    arr.assign(data);
    
    ndarray<float> result = arr.tan();
    std::vector<float> resultData = result.data();
    
    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::tan(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Tan2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-TWO_PI + 0.1f, TWO_PI - 0.1f);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }

    arr.assign(data);

    ndarray<float> result = arr.tan();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            EXPECT_NEAR(resultData[index++], std::tan(data[i][j]), 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Asin1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }

    arr.assign(data);

    ndarray<float> result = arr.asin();
    std::vector<float> resultData = result.data();

    // #pragma omp simd
    for (size_t i = 0; i < shape[0]; ++i) {
        EXPECT_NEAR(resultData[i], std::asin(data[i]), 1e-3);
    }
}

TEST(NDArrayMathTest, Asin2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }

    arr.assign(data);

    ndarray<float> result = arr.asin();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            float expected = std::asin(data[i][j]);
            EXPECT_NEAR(resultData[index++], expected, 1e-3);
        }
    }
}

TEST(NDArrayMathTest, Acos1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }

    arr.assign(data);

    ndarray<float> result = arr.acos();
    std::vector<float> resultData = result.data();

    // #pragma omp simd
    for (size_t i = 0; i < data.size(); ++i) {
        float expected = std::acos(data[i]);
        EXPECT_NEAR(resultData[i], expected, 1e-3);
    }
}

TEST(NDArrayMathTest, Acos2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }

    arr.assign(data);

    ndarray<float> result = arr.acos();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            float expected = std::acos(data[i][j]);
            EXPECT_NEAR(resultData[index], expected, 1e-3);
            index++;
        }
    }
}

TEST(NDArrayMathTest, Atan1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }

    arr.assign(data);

    ndarray<float> result = arr.atan();
    std::vector<float> resultData = result.data();

    // #pragma omp simd
    for (size_t i = 0; i < data.size(); ++i) {
        float expected = std::atan(data[i]);
        EXPECT_NEAR(resultData[i], expected, 1e-3);
    }
}

TEST(NDArrayMathTest, Atan2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }

    arr.assign(data);

    ndarray<float> result = arr.atan();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            float expected = std::atan(data[i][j]);
            EXPECT_NEAR(resultData[index], expected, 1e-3);
            index++;
        }
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
//         EXPECT_NEAR(resultData[i], std::sin(data[i]), 1e-3);
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
//             EXPECT_NEAR(resultData[index++], std::sin(data[i][j]), 1e-3);
//         }
//     }
// }

TEST(NDArrayMathTest, Round1DTest) {
    std::vector<size_t> shape = {10000};
    ndarray<float> arr(shape);
    std::vector<float> data(shape[0]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        data[i] = dis(gen);
    }

    arr.assign(data);

    ndarray<float> result = arr.round();
    std::vector<float> resultData = result.data();

    // #pragma omp simd
    for (size_t i = 0; i < data.size(); ++i) {
        float expected = std::round(data[i]);
        EXPECT_NEAR(resultData[i], expected, 1e-3);
    }
}

TEST(NDArrayMathTest, Round2DTest) {
    std::vector<size_t> shape = {1000, 1000};
    ndarray<float> arr(shape);
    std::vector<std::vector<float>> data(shape[0], std::vector<float>(shape[1]));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-10.0f, 10.0f);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i][j] = dis(gen);
        }
    }

    arr.assign(data);

    ndarray<float> result = arr.round();
    std::vector<float> resultData = result.data();

    size_t index = 0;
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            float expected = std::round(data[i][j]);
            EXPECT_NEAR(resultData[index], expected, 1e-3);
            index++;
        }
    }
}

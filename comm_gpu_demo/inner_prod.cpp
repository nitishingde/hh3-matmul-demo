#include "mm.h"

#define VERIFY_MM 1

int main([[maybe_unused]]int argc, [[maybe_unused]]char *argv[]) {
    using MatrixType = double;
    constexpr Order Ord = Order::Column;

    CublasLockGuard cublasLockGuard;

    int devCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&devCount));
    std::vector<int32_t> deviceIds;
    for(int i = 0; i < devCount; ++i) deviceIds.emplace_back(i);
#if not NDEBUG
    printf("Devices: {");
        for(auto dev: deviceIds) {
            printf("%d, ", dev);
        }
        printf("\b\b}\n");
#endif

    // A => m x k
    // B => k x n
    // C => m x n
    size_t m = std::stoull(argv[1]), k = std::stoull(argv[2]), n = std::stoull(argv[3]), blockSize = std::stoull(argv[4]);
    printf("M = %zu, K = %zu, N = %zu, B = %zu\n", m, k, n, blockSize);

    std::vector<MatrixType> A(m*k), B(k*n), C(m*n, 1);

    auto matrixA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(m, k, blockSize, *A.data());
    auto matrixB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(k, n, blockSize, *B.data());
    auto matrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(m, n, blockSize, *C.data());

    // initialize matrices
    std::for_each(matrixA->data(), matrixA->data() + (m * k), [](MatrixType &val) { val = (MatrixType) fastrand(); });
    std::for_each(matrixB->data(), matrixB->data() + (k * n), [](MatrixType &val) { val = (MatrixType) fastrand(); });

    std::cout << "Done initializing matrices" << std::endl;
    {
        MMInnerProduct<MatrixType, Ord> mmInnerProduct;
        mmInnerProduct.execute(matrixA, matrixB, matrixC, deviceIds);
    }

#if VERIFY_MM
    // verify code
    std::vector<MatrixType> V(m*n, 1);
    auto testMatrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(m, n, blockSize, *V.data());

    {
        MMVerification<MatrixType, Ord> mmVerification;
        mmVerification.execute(matrixA, matrixB, testMatrixC, deviceIds);
    }
    for(size_t i = 0; i < m*n; ++i) {
        if(0.01 < std::abs(testMatrixC->data()[i]-matrixC->data()[i])) {
#if not NDEBUG
            std::cout << *matrixA;
            std::cout << *matrixB;
            std::cout << *matrixC;
            std::cout << *testMatrixC;
#endif
            throw std::runtime_error(
                std::string("Matrix multiplication output is wrong!\n") +
                "@index = " + std::to_string(i) + "\n" +
                "{cublasXt = " + std::to_string(testMatrixC->data()[i]) + ", calculated = " + std::to_string(matrixC->data()[i]) + "}\n" +
                "diff = " + std::to_string(std::abs(testMatrixC->data()[i]-matrixC->data()[i])) + "\n"
            );
        }
    }
#if not NDEBUG
    std::cout << *matrixA;
    std::cout << *matrixB;
    std::cout << *matrixC;
    std::cout << *testMatrixC;
#endif

#endif

    return 0;
}
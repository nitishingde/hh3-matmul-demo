#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include "data/matrix_data.h"
#include "mm.h"

#define VERIFY_MM 1

int main([[maybe_unused]]int32_t argc, [[maybe_unused]]char **argv) {
    using MatrixType = double;
    constexpr Order Ord = Order::Column;
    using namespace std::chrono_literals;

    CublasLockGuard cublasLockGuard;

#if not NDEBUG
    printf("[%d]\n", argc);
    for(int i = 0; i < argc; ++i) {
        printf("\t[%s]\n", argv[i]);
    }
    printf("\n");
#endif

    // A => m x k
    // B => k x n
    // C => m x n
    size_t m = std::stoull(argv[1]), k = std::stoull(argv[2]), n = std::stoull(argv[3]), blockSize = std::stoull(argv[4]);
    printf("M = %zu, K = %zu, N = %zu, B = %zu\n", m, k, n, blockSize);

    std::vector<MatrixType> A(m*k), B(k*n), C(m*n);

    auto matrixA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(m, k, blockSize, *A.data());
    auto matrixB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(k, n, blockSize, *B.data());
    auto matrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(m, n, blockSize, *C.data());
#if VERIFY_MM
    std::vector<MatrixType> V(m*n);
    auto testMatrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(m, n, blockSize, *V.data());
#endif

    // Mersenne Twister Random Generator
    uint64_t timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> (uint64_t) 32)};
    std::mt19937_64 rng(ss);
    // Choose your distribution depending on the type of MatrixType
    std::uniform_real_distribution<MatrixType> unif(0, 10);

    // initialize matrices
    std::for_each(matrixA->data(), matrixA->data() + (m * k), [&unif, &rng](MatrixType &val) { val = (MatrixType) unif(rng); });
    std::for_each(matrixB->data(), matrixB->data() + (k * n), [&unif, &rng](MatrixType &val) { val = (MatrixType) unif(rng); });
    std::for_each(matrixC->data(), matrixC->data() + (m * n), [&unif, &rng](MatrixType &val) { val = 1;(MatrixType) unif(rng); });
#if VERIFY_MM
    for(size_t i = 0; i < m*n; ++i) {
        testMatrixC->data()[i] = matrixC->data()[i];
    }
#endif

    MMOuterProduct<MatrixType, Ord> mmOuterProduct(m, k, n, blockSize);
    mmOuterProduct.execute(matrixA, matrixB, matrixC);

#if VERIFY_MM
    // verify code
    MMVerification<MatrixType, Ord> mmVerification;
    mmVerification.execute(matrixA, matrixB, testMatrixC);
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
                "{original = " + std::to_string(testMatrixC->data()[i]) + ", calculated = " + std::to_string(matrixC->data()[i]) + "}\n" +
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

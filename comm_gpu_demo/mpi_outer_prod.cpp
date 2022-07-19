#include <random>
#include <atomic>
#include <thread>

#include "data/matrix_data.h"
#include "mm.h"

#define VERIFY_MM 1

int main([[maybe_unused]]int32_t argc, [[maybe_unused]]char **argv) {
    using MatrixType = double;
    constexpr Order Ord = Order::Column;
    using namespace std::chrono_literals;

    int32_t mpiNodeId = -1, mpiNumNodes = -1;
    bool isRootNode = false;
    if(MPI_Init(&argc, &argv) == MPI_SUCCESS) {
        int32_t flag = false;
        if(auto status = MPI_Initialized(&flag); status == MPI_SUCCESS and flag) {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiNodeId);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiNumNodes);
            isRootNode = (mpiNodeId == 0);
        } else {
            return -1;
        }
    }

    CublasLockGuard cublasLockGuard;

    std::vector<int32_t> deviceIds{mpiNodeId};
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
    int parameterIdx = 0;
    for(size_t i = 0; i < argc; ++i) {
        if(strcmp(argv[i], "--params") == 0) break;
        parameterIdx++;
    }
    size_t m = std::stoull(argv[parameterIdx+1]), k = std::stoull(argv[parameterIdx+2]), n = std::stoull(argv[parameterIdx+3]), blockSize = std::stoull(argv[parameterIdx+4]);
    printf("[Debug][Process %d] M = %zu, K = %zu, N = %zu, B = %zu\n", mpiNodeId, m, k, n, blockSize);

    auto subMatA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(m, k, blockSize, *(new MatrixType[m*k]), true);
    auto subMatB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(k, n, blockSize, *(new MatrixType[m*k]), true);
    auto matrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(m, n, blockSize, *(new MatrixType[m*k]), true);

    // initialize matrices
#if not NDEBUG
    std::for_each(subMatA->data(), subMatA->data() + (m * k), [&mpiNodeId](MatrixType &val) { val = mpiNodeId+1; });
    std::for_each(subMatB->data(), subMatB->data() + (k * n), [&mpiNodeId](MatrixType &val) { val = mpiNodeId+1; });
#else
    // Mersenne Twister Random Generator
    uint64_t timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> (uint64_t) 32)};
    std::mt19937_64 rng(ss);
    // Choose your distribution depending on the type of MatrixType
    std::uniform_real_distribution<MatrixType> unif(0, 10);

    std::for_each(subMatA->data(), subMatA->data() + (m * k), [&unif, &rng](MatrixType &val) { val = (MatrixType) unif(rng); });
    std::for_each(subMatB->data(), subMatB->data() + (k * n), [&unif, &rng](MatrixType &val) { val = (MatrixType) unif(rng); });
#endif
    if(isRootNode) {
        std::for_each(matrixC->data(), matrixC->data() + (m * n), [](MatrixType &val) { val = 1; });
        std::cout << "[Process 0] Done initializing matrices" << std::endl;
    }

    {
        MMMPIOuterProduct<MatrixType, Ord> mpiOuterProduct;
        mpiOuterProduct.execute(subMatA, subMatB, matrixC, deviceIds);
    }

#if VERIFY_MM
    // initialize matrices for verification
    std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> testMatrixC = nullptr;
    if(isRootNode) {
        testMatrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(m, n, blockSize, *(new MatrixType[m*n]), true);
        std::for_each(testMatrixC->data(), testMatrixC->data() + (m * n), [](MatrixType &val) { val = 1; });
    }
    else {
        testMatrixC = matrixC;
        std::for_each(testMatrixC->data(), testMatrixC->data() + (m * n), [](MatrixType &val) { val = 0; });
    }

    {
        MMCommVerification<MatrixType, Ord> mmCommVerification;
        mmCommVerification.execute(subMatA, subMatB, testMatrixC, deviceIds);
    }

    if(isRootNode) {
        for(size_t i = 0; i < m*n; ++i) {
            if(0.01 < std::abs(testMatrixC->data()[i]-matrixC->data()[i])) {
                throw std::runtime_error(
                    std::string("Matrix multiplication output is wrong!\n") +
                    "@index = " + std::to_string(i) + "\n" +
                    "{original = " + std::to_string(testMatrixC->data()[i]) + ", calculated = " + std::to_string(matrixC->data()[i]) + "}\n" +
                    "diff = " + std::to_string(std::abs(testMatrixC->data()[i]-matrixC->data()[i])) + "\n"
                );
            }
        }
    }
#endif

    if(MPI_Finalize() != MPI_SUCCESS) {
        return -1;
    }

    return 0;
}

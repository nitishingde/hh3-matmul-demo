#include <comm/comm.h>
#include "mm.h"


template<class MatrixType, Order Ord>
void reset(std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> matrixC) {
    size_t m = matrixC->matrixHeight(), n = matrixC->matrixWidth();
    if(comm::isMpiRootPid()) {
        std::for_each(matrixC->data(), matrixC->data() + (m * n), [](MatrixType &val) { val = 1; });
    }
    else {
        std::for_each(matrixC->data(), matrixC->data() + (m * n), [](MatrixType &val) { val = 0; });
    }
}

int main([[maybe_unused]]int32_t argc, [[maybe_unused]]char **argv) {
    using MatrixType = double;
    constexpr Order Ord = Order::Column;
    using namespace std::chrono_literals;

    CublasLockGuard cublasLockGuard;

    // initialize comm library
    comm::CommLockGuard commLockGuard(&argc, &argv);

    std::vector<int32_t> deviceIds{comm::getMpiNodeId()};
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
    printf("[Debug][Process %d] M = %zu, K = %zu, N = %zu, B = %zu\n", comm::getMpiNodeId(), m, k, n, blockSize);

    auto subMatA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(m, k, blockSize, *(new MatrixType[m*k]), true);
    auto subMatB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(k, n, blockSize, *(new MatrixType[m*k]), true);
    auto matrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(m, n, blockSize, *(new MatrixType[m*k]), true);

    // initialize matrices
#if not NDEBUG
    std::for_each(subMatA->data(), subMatA->data() + (m * k), [](MatrixType &val) { val = comm::getMpiNodeId()+1; });
    std::for_each(subMatB->data(), subMatB->data() + (k * n), [](MatrixType &val) { val = comm::getMpiNodeId()+1; });
#else
    std::for_each(subMatA->data(), subMatA->data() + (m * k), [](MatrixType &val) { val = (MatrixType) fastrand(); });
    std::for_each(subMatB->data(), subMatB->data() + (k * n), [](MatrixType &val) { val = (MatrixType) fastrand(); });
#endif

    {
        reset(matrixC);
        comm::barrier();
        MM_CommOuterProduct<MatrixType, Ord>().execute(subMatA, subMatB, matrixC, deviceIds);
    }

    comm::barrier();
    comm::stopDaemon();

    {
        reset(matrixC);
        MPI_Barrier(MPI_COMM_WORLD);
        MM_CommOuterProduct2<MatrixType, Ord>().execute(subMatA, subMatB, matrixC, deviceIds);
    }

    {
        reset(matrixC);
        MPI_Barrier(MPI_COMM_WORLD);
        MM_MpiOuterProduct<MatrixType, Ord>().execute(subMatA, subMatB, matrixC, deviceIds);
    }

    {
        reset(matrixC);
        MPI_Barrier(MPI_COMM_WORLD);
        MM_MpiOuterProduct2<MatrixType, Ord>().execute(subMatA, subMatB, matrixC, deviceIds);
    }

    {
        reset(matrixC);
        MPI_Barrier(MPI_COMM_WORLD);
        MM_MpiVerification<MatrixType, Ord>().execute(subMatA, subMatB, matrixC, deviceIds);
    }

    return 0;
}

#include "data/matrix_order.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"
#include "data/contiguous_sub_matrix_container.h"
#include "matrix_utility.h"
#include "mmd.h"
#include <cstdio>

#define VERIFY_MMD true
#define DUMP_DATA  false

int main([[maybe_unused]]int32_t argc, [[maybe_unused]]char **argv) {
    using MatrixType = float;
    constexpr Order Ord = Order::Col;
    using namespace std::chrono_literals;

    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv);

    MPI_Comm matrixComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &matrixComm);
    MPI_Barrier(matrixComm);

    auto [M, K, N, tileSize, productThreads, commThreads, path] = parseArgs(argc, argv);
    printf("[Process %d] M = %lu, K = %lu, N = %lu, tileSize = %lu\n", getNodeId(), M, K, N, tileSize);

    int32_t devCount = 0;
    cudaGetDeviceCount(&devCount);
    std::vector<int32_t> deviceIds{getNodeId()};
    cudaDeviceProp cudaDeviceProp{};
    cudaGetDeviceProperties(&cudaDeviceProp, deviceIds[0]);
    printf("[Process %d][GPU %d/%d][%s][RAM = %zuGB][#AyncEngines = %d]\n",
           getNodeId(),
           deviceIds[0], devCount,
           cudaDeviceProp.name,
           cudaDeviceProp.totalGlobalMem/(1<<30),
           cudaDeviceProp.asyncEngineCount
    );

    auto subMatA = std::make_shared<ContiguousSubMatrixContainer<Order::Col, MatrixType, 'a', Ord>>(0, M, K, tileSize, matrixComm);
    auto subMatB = std::make_shared<ContiguousSubMatrixContainer<Order::Row, MatrixType, 'b', Ord>>(1, K, N, tileSize, matrixComm);
    init(subMatA);
    init(subMatB);

    auto matrixC = std::make_shared<Cyclic2dMatrixContainer<MatrixType, 'c', Ord>>(2, M, N, tileSize, matrixComm);
    init(matrixC);

    MPI_Barrier(MPI_COMM_WORLD);
    if(isRootNodeId()) printf("Matrices initialized on every node\n");

    {
        MPI_Barrier(MPI_COMM_WORLD);
        MMD_MpiOuterProduct1<MatrixType, 'a', 'b', 'c', Ord>(productThreads, commThreads).execute(subMatA, subMatB, matrixC, deviceIds);
        matrixC->shrink();
    }

#if VERIFY_MMD
    verifySolution(std::move(subMatA), std::move(subMatB), matrixC, deviceIds, matrixComm);
    subMatA = nullptr;
    subMatB = nullptr;
#endif

#if DUMP_DATA
    dumpData(matrixC, path);
#endif

    MPI_Comm_free(&matrixComm);

    return 0;
}

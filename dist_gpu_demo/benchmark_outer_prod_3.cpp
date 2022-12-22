#include "data/matrix_order.h"
#include "data/cyclic2d_matrix_container.h"
#include "matrix_utility.h"
#include "mmd.h"
#include <cstdio>

#define VERIFY_MMD false
#define DUMP_DATA  false

int main([[maybe_unused]]int32_t argc, [[maybe_unused]]char **argv) {
    using MatrixType = float;
    constexpr Order Ord = Order::Col;
    using namespace std::chrono_literals;

    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv);
    CublasGlobalLockGuard cublasGlobalLockGuard;

    MPI_Comm matrixComm;
    checkMpiErrors(MPI_Comm_dup(MPI_COMM_WORLD, &matrixComm));
    checkMpiErrors(MPI_Barrier(matrixComm));

    auto [M, K, N, tileSize, productThreads, commThreads, path] = parseArgs(argc, argv);
    printf("[Process %d] M = %lu, K = %lu, N = %lu, tileSize = %lu, path = %s\n", getNodeId(), M, K, N, tileSize, path.c_str());

    int32_t devCount = 0;
    cudaGetDeviceCount(&devCount);
    std::vector<int32_t> deviceIds;
    deviceIds.reserve(8);
    for(int32_t i = 0; i < devCount; ++i) {
        deviceIds.emplace_back(i);
        cudaDeviceProp cudaDeviceProp{};
        cudaGetDeviceProperties(&cudaDeviceProp, i);
        printf("[Process %d][GPU %d/%d][%s][RAM = %zuGB][#AyncEngines = %d]\n",
           getNodeId(),
           i, devCount,
           cudaDeviceProp.name,
           cudaDeviceProp.totalGlobalMem/(1<<30),
           cudaDeviceProp.asyncEngineCount
        );
    }

    auto subMatA = std::make_shared<TiledSubMatrixContainer<Order::Col, MatrixType, 'a', Ord>>(0, M, K, tileSize, matrixComm);
    auto subMatB = std::make_shared<TiledSubMatrixContainer<Order::Row, MatrixType, 'b', Ord>>(1, K, N, tileSize, matrixComm);
    init(subMatA);
    init(subMatB);

    auto matrixC = std::make_shared<Cyclic2dMatrixContainer<MatrixType, 'c', Ord>>(2, M, N, tileSize, matrixComm);
    init(matrixC);

    if(isRootNodeId()) printf("Matrices initialized on every node\n");

    auto strategy = MMD_MpiOuterProduct3<MatrixType, 'a', 'b', 'c', Ord>(productThreads, commThreads);
    constexpr uint32_t ITER = 10;
    double time[ITER] = {0.};
    for(uint32_t iter = 0; iter < ITER; ++iter) {
        checkMpiErrors(MPI_Barrier(MPI_COMM_WORLD));
        auto start = std::chrono::high_resolution_clock::now();
        strategy.executeImpl(subMatA, subMatB, matrixC, deviceIds, path + "MMD_Unified_tile"+std::to_string(tileSize)+"_iter"+std::to_string(iter)+"_node");
        checkMpiErrors(MPI_Barrier(MPI_COMM_WORLD));
        auto end = std::chrono::high_resolution_clock::now();
        time[iter] = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1.e9;
        if(isRootNodeId()) {
            printf("\r[Iterations: %d/%d][ Perf " GREEN("%9.3f") " gflops ][ Time " BLUE("%8.3f") " secs]",
               iter+1, ITER,
               (double(M) * double(K) * double(N) * double(2)) / (1.e9 * time[iter]),
               time[iter]
           );
            fflush(stdout);
        }
        init(matrixC);
    }

    if(isRootNodeId()) {
        double gflop = (double(M) * double(K) * double(N) * double(2)) / 1.e9;
        double minTime = *std::min_element(time, time+ITER);
        double avgTime = std::accumulate(time, time+ITER, 0.0)/double(ITER);
        double maxTime = *std::max_element(time, time+ITER);
        printf(
            "\r[ Iterations " YELLOW("%d") " ][ " MAGENTA("Hedgehog+MPI") " ][ M, N, K, T = (" MAGENTA("%lu, %lu, %lu, %lu") ") ][ Max " GREEN("%9.3f") " gflops ][ Avg " CYAN("%9.3f") " gflops ][ Min " RED("%9.3f") " gflops ][ Min " GREEN("%8.3f") " secs ][ Avg " CYAN("%8.3f") " secs ][ Max " RED("%8.3f") " secs ]\n",
            ITER,
            M, N, K, tileSize,
            gflop/minTime,
            gflop/avgTime,
            gflop/maxTime,
            minTime,
            avgTime,
            maxTime
        );
    }

    matrixC->shrink();
    checkMpiErrors(MPI_Barrier(MPI_COMM_WORLD));

#if VERIFY_MMD
    verifySolution(std::move(subMatA), std::move(subMatB), matrixC, deviceIds, matrixComm);
    subMatA = nullptr;
    subMatB = nullptr;
#endif

#if DUMP_DATA
    dumpData(matrixC, path);
#endif

    checkMpiErrors(MPI_Comm_free(&matrixComm));

    return 0;
}

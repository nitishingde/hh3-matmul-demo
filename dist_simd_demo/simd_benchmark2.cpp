#include <filesystem>
#include <openblas/cblas.h>
#include "common_matrix_utility.h"
#include "mmd.h"
#include "utility.h"

int main(int argc, char *argv[]) {
    auto [p, q, M, K, N, T, l, gp, gq, _wh, _ww, d, productThreads, verbose, path, resultsFile] = parseArgs(argc, argv);
    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv, p, q, MPI_THREAD_SERIALIZED);

    using MatrixType = float;

    constexpr char IdA        = 'A';
    constexpr char IdB        = 'B';
    constexpr char IdC        = 'C';
    constexpr auto memoryType = MemoryType::HOST;

    MPI_Comm gridComm   = MPI_COMM_NULL;
    MPI_Comm groupCommA = MPI_COMM_NULL;
    MPI_Comm groupCommB = MPI_COMM_NULL;
    const auto     dims     = std::array{static_cast<int32_t>(p), static_cast<int32_t>(q), 1};
    constexpr auto periods  = std::array{1, 1, 1};
    if constexpr(true) {
        auto mpiLG = std::lock_guard(mpiMutex);
        checkMpiErrors(MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods.data(), true, &gridComm));
        auto remains = std::array{0, 1, 0};
        checkMpiErrors(MPI_Cart_sub(gridComm, remains.data(), &groupCommA));
        remains = {1, 0, 0};
        checkMpiErrors(MPI_Cart_sub(gridComm, remains.data(), &groupCommB));
    }
    MpiGlobalLockGuard::init(gridComm);

    std::ofstream csvFile;


    auto matrixA = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdA>>(memoryType, M, K, T, p, q, gridComm);
    auto matrixB = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdB>>(memoryType, K, N, T, p, q, gridComm);
    auto matrixC = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdC>>(memoryType, M, N, T, p, q, gridComm);

    if(isRootNodeId()) {
        printf("[Node %ld][p %ld][q %ld][M %ld][K %ld][N %ld][T %ld][l %ld][productThreads %ld][verbosity level %ld]\n", getNodeId(), p, q, M, K, N, T, l, productThreads, verbose);
        printDataDistribution<MatrixType, IdA, IdB, IdC>(matrixA, matrixB, matrixC);
        std::filesystem::remove_all(path);
        std::filesystem::create_directory(path);
        csvFile.open(resultsFile);
        csvFile << "iteration, gflops, time" << std::endl;
    }

    auto strategy = MMD_Simd2<MatrixType, IdA, IdB, IdC>(groupCommA, groupCommB);
    //warmup
    auto time = strategy.builder(static_cast<int32_t>(productThreads), static_cast<int32_t>(l)).executeImpl(matrixA, matrixB, matrixC, gridComm, path + "window" + "_" + std::to_string(getNodeId()) + ".dot");
    if(isRootNodeId()) {
        printf("[Warmup][ Perf " GREEN("%9.3f") " gflops ][ Time " BLUE("%8.3f") " secs]\n",
            (static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N) * 2.0) / (1.e9 * time),
            time
        );
    }


    constexpr int32_t ITER = 10;
    double times[ITER];
    for(int32_t iter = 0; iter < ITER; ++iter) {
        times[iter]  = strategy.builder(static_cast<int32_t>(productThreads), static_cast<int32_t>(l)).executeImpl(matrixA, matrixB, matrixC, gridComm, path + "window" + std::to_string(iter) + "_" + std::to_string(getNodeId()) + ".dot");
        if(isRootNodeId()) {
            double gflops = (static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N) * 2.0) / (1.e9 * times[iter]);
            csvFile << iter+1 << ", " << gflops << ", " << times[iter] << std::endl;
            printf("[%s][Iterations: %3d/%d][ Perf " GREEN("%9.3f") " gflops ][ Time " BLUE("%8.3f") " secs]\n",
                   "WindowStrategyBatched_v3",
                   iter+1, ITER,
                   gflops,
                   times[iter]
            );
            fflush(stdout);
        }
    }

    if(isRootNodeId()) {
        double gflop = (static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N) * 2.0) / 1.e9;
        double minTime = *std::min_element(times, times+ITER);
        double avgTime = std::accumulate(times, times+ITER, 0.0)/static_cast<double>(ITER);
        double maxTime = *std::max_element(times, times+ITER);
        printf("[%s][Iterations: %3d/%d][ Max " GREEN("%9.3f") " gflops ][ Avg " CYAN("%9.3f") " gflops ][ Min " RED("%9.3f") " gflops ][ Min " GREEN("%8.3f") " secs ][ Avg " CYAN("%8.3f") " secs ][ Max " RED("%8.3f") " secs ]\n",
           "WindowStrategyBatched_v3",
           ITER, ITER,
           gflop/minTime,
           gflop/avgTime,
           gflop/maxTime,
           minTime,
           avgTime,
           maxTime
        );
        fflush(stdout);
    }
    return 0;
}

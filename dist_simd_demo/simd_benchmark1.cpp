#include <filesystem>
#include "common_matrix_utility.h"
#include "mmd.h"
#include "utility.h"

int main(int argc, char *argv[]) {
    auto [p, q, M, K, N, T, lookAhead, _gp, _gq, _wh, _ww, _d, productThreads, verbose, path, resultsFile] = parseArgs(argc, argv);
    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv, p, q, MPI_THREAD_SERIALIZED);

    using MatrixType = float;

    constexpr char       IdA        = 'a';
    constexpr char       IdB        = 'b';
    constexpr char       IdC        = 'c';
    constexpr MemoryType memoryType = MemoryType::HOST;
    MPI_Comm             mpiComm    = MPI_COMM_WORLD;

    MPI_Comm             mpiCommA   = MPI_COMM_WORLD;
    MPI_Comm             mpiCommB   = MPI_COMM_WORLD;
    MPI_Comm             mpiCommC   = MPI_COMM_WORLD;
    checkMpiErrors(MPI_Comm_dup(mpiComm, &mpiCommB));

    std::ofstream csvFile;

    printf("[Node %ld][p %ld][q %ld][M %ld][K %ld][N %ld][T %ld][l %ld][productThreads %ld][verbosity level %ld]\n", getNodeId(), p, q, M, K, N, T, lookAhead, productThreads, verbose);
    fflush(stdout);

    auto matrixA = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdA>>(memoryType, M, K, T, p, q, mpiCommA);
    auto matrixB = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdB>>(memoryType, K, N, T, p, q, mpiCommB);
    auto matrixC = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdC>>(memoryType, M, N, T, p, q, mpiCommC);

    if(isRootNodeId()) {
        printDataDistribution<MatrixType, IdA, IdB, IdC>(matrixA, matrixB, matrixC);
        std::filesystem::remove_all(path);
        std::filesystem::create_directory(path);
        csvFile.open(resultsFile);
        csvFile << "iteration, gflops, time" << std::endl;
    }

    auto strategy = MMD_Simd1<MatrixType, IdA, IdB, IdC>();
    //warmup
    auto time = strategy.builder(productThreads, lookAhead).executeImpl(matrixA, matrixB, matrixC, mpiComm, path + "window" + "_" + std::to_string(getNodeId()) + ".dot");
    if(isRootNodeId()) {
        printf("[Warmup][ Perf " GREEN("%9.3f") " gflops ][ Time " BLUE("%8.3f") " secs]\n",
            (static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N) * static_cast<double>(2)) / (1.e9 * time),
            time
        );
    }


    constexpr int32_t ITER = 10;
    double times[ITER];
    for(int32_t iter = 0; iter < ITER; ++iter) {
        times[iter]  = strategy.builder(productThreads, lookAhead).executeImpl(matrixA, matrixB, matrixC, mpiComm, path + "window" + std::to_string(iter) + "_" + std::to_string(getNodeId()) + ".dot");
        if(isRootNodeId()) {
            double gflops = (static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N) * static_cast<double>(2)) / (1.e9 * times[iter]);
            csvFile << iter+1 << ", " << gflops << ", " << times[iter] << std::endl;
            printf("[%s][Iterations: %3d/%d][ Perf " GREEN("%9.3f") " gflops ][ Time " BLUE("%8.3f") " secs]\n",
                   "WindowStrategyBroadcast_v3",
                   iter+1, ITER,
                   gflops,
                   times[iter]
            );
            fflush(stdout);
        }
    }

    if(isRootNodeId()) {
        double gflop = (static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(N) * static_cast<double>(2)) / 1.e9;
        double minTime = *std::min_element(times, times+ITER);
        double avgTime = std::accumulate(times, times+ITER, 0.0)/static_cast<double>(ITER);
        double maxTime = *std::max_element(times, times+ITER);
        printf("[%s][Iterations: %3d/%d][ Max " GREEN("%9.3f") " gflops ][ Avg " CYAN("%9.3f") " gflops ][ Min " RED("%9.3f") " gflops ][ Min " GREEN("%8.3f") " secs ][ Avg " CYAN("%8.3f") " secs ][ Max " RED("%8.3f") " secs ]\n",
           "WindowStrategyBroadcast_v3",
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

#include "mmd.h"
#include "matrix_utility.h"

int main(int argc, char *argv[]) {
    auto [p, q, M, K, N, T, prodThreads, windowSize, path, host] = parseArgs(argc, argv);
    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv, p, q);

    using MatrixType = double;

    constexpr char       IdA        = 'a';
    constexpr char       IdB        = 'b';
    constexpr char       IdC        = 'c';
    constexpr MemoryType memoryType = MemoryType::HOST;
    MPI_Comm             mpiComm    = MPI_COMM_WORLD;

    std::ofstream csvFile;

    windowSize = genWindowSize<MatrixType>(T, prodThreads, windowSize);
    printf("[Node %ld][p %ld][q %ld][M %ld][K %ld][N %ld][T %ld][prodThreads %ld][windowSize %ld]\n", getNodeId(), p, q, M, K, N, T, prodThreads, windowSize);
    fflush(stdout);

    int32_t gpuCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&gpuCount));
    auto deviceIds = std::vector<int32_t>();
    deviceIds.reserve(gpuCount);
    for(int32_t i = 0; i < gpuCount; ++i) {
        deviceIds.emplace_back(i);
        cudaDeviceProp cudaDeviceProp{};
        cudaGetDeviceProperties(&cudaDeviceProp, i);
        printf("[Process %ld][GPU %d/%d][%s][RAM = %zuGB][#AyncEngines = %d][Compute capability = %d.%d]\n",
            getNodeId(),
            i, gpuCount,
            cudaDeviceProp.name,
            cudaDeviceProp.totalGlobalMem/(1<<30),
            cudaDeviceProp.asyncEngineCount,
            cudaDeviceProp.major, cudaDeviceProp.minor
        );
    }
    CublasGlobalLockGuard cublasGlobalLockGuard(deviceIds);

    auto matrixA = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdA>>(memoryType, M, K, T, p, q, mpiComm);
    auto matrixB = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdB>>(memoryType, K, N, T, p, q, mpiComm);
    auto matrixC = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdC>>(memoryType, M, N, T, p, q, mpiComm);

    if(isRootNodeId()) {
        printDataDistribution<MatrixType, IdA, IdB, IdC>(matrixA, matrixB, matrixC);
        auto fileName = "./results" + std::to_string(M/1024) + std::to_string(T/1024) + "k_" + std::to_string(getNumNodes()) + ".csv";
        csvFile.open(fileName);
        csvFile << "iteration, gflops, time" << std::endl;
    }

    auto strategy = MMD_WindowStrategy<MatrixType, IdA, IdB, IdC>();

    constexpr int32_t ITER = 10;
    double times[ITER];
    for(int32_t iter = 0; iter < ITER; ++iter) {
        times[iter]  = strategy.builder(prodThreads, windowSize).executeImpl(matrixA, matrixB, matrixC, deviceIds, mpiComm, path + "window" + std::to_string(iter) + "_" + std::to_string(getNodeId()) + ".dot");
        if(isRootNodeId()) {
            double gflops = (double(M) * double(K) * double(N) * double(2)) / (1.e9 * times[iter]);
            csvFile << iter+1 << ", " << gflops << ", " << times[iter] << std::endl;
            printf("[Iterations: %3d/%d][ Perf " GREEN("%9.3f") " gflops ][ Time " BLUE("%8.3f") " secs]\n",
                iter+1, ITER,
                gflops,
                times[iter]
            );
            fflush(stdout);
        }
    }

    if(isRootNodeId()) {
        double gflop = (double(M) * double(K) * double(N) * double(2)) / 1.e9;
        double minTime = *std::min_element(times, times+ITER);
        double avgTime = std::accumulate(times, times+ITER, 0.0)/double(ITER);
        double maxTime = *std::max_element(times, times+ITER);
        printf("[Iterations: %3d/%d][ Max " GREEN("%9.3f") " gflops ][ Avg " CYAN("%9.3f") " gflops ][ Min " RED("%9.3f") " gflops ][ Min " GREEN("%8.3f") " secs ][ Avg " CYAN("%8.3f") " secs ][ Max " RED("%8.3f") " secs ]\n",
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

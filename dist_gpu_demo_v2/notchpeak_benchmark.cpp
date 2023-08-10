#include "graphs.h"
#include "tasks.h"
#include <hedgehog/hedgehog.h>

int main(int argc, char *argv[]) {
    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv);

    constexpr char       IdA        = 'a';
    constexpr char       IdB        = 'b';
    constexpr char       IdC        = 'c';
    constexpr char       IdP        = 'p';
    constexpr MemoryType memoryType = MemoryType::CUDA_UNIFIED_MEMORY;
    MPI_Comm             mpiComm    = MPI_COMM_WORLD;

    using MatrixType = float;
    using MatrixA    = MatrixContainer<MatrixType, IdA>;
    using MatrixB    = MatrixContainer<MatrixType, IdB>;
    using MatrixC    = MatrixContainer<MatrixType, IdC>;
    using Triplet    = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>;
    using TileA      = MatrixTile<MatrixType, IdA>;
    using TileB      = MatrixTile<MatrixType, IdB>;
    using TileC      = MatrixTile<MatrixType, IdC>;
    using TileP      = MatrixTile<MatrixType, IdP>;
    using Pair       = std::tuple<std::shared_ptr<TileC>, std::shared_ptr<TileP>>;
    using Job        = GpuJob<MatrixType, IdA, IdB, IdC>;

    auto [p, q, M, K, N, T, prodThreads, path, host] = parseArgs(argc, argv);
    printf("[Node %ld][p %ld][q %ld][M %ld][K %ld][N %ld][T %ld][prodThreads %ld]\n", getNodeId(), p, q, M, K, N, T, prodThreads);
    fflush(stdout);

    int32_t gpuCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&gpuCount));
    cudaDeviceProp cudaDeviceProp = {};
    checkCudaErrors(cudaGetDeviceProperties(&cudaDeviceProp, 0));
    auto deviceIds = std::vector<int32_t>();
    deviceIds.reserve(gpuCount);
    for(int32_t i = 0; i < gpuCount; ++i) {
        deviceIds.emplace_back(i);
    }
    CublasGlobalLockGuard cublasGlobalLockGuard(deviceIds);

    auto matrixA = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdA>>(memoryType, M, K, T, p, q, mpiComm);
    auto matrixB = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdB>>(memoryType, K, N, T, p, q, mpiComm);
    auto matrixC = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdC>>(memoryType, M, N, T, p, q, mpiComm);

    if(isRootNodeId()) {
        printf("Data distribution:\n");
        fflush(stdout);
        for(int64_t row = 0; row < matrixB->matrixNumRowTiles(); ++row) {
            for(int64_t col = 0; col < matrixA->matrixNumColTiles(); ++col) {
                printf("   ");
            }
            printf("  ");
            for(int64_t col = 0; col < matrixB->matrixNumColTiles(); ++col) {
                printf("%2ld ", matrixB->owner(row, col));
            }
            printf("\b\n");
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);

        for(int64_t row = 0; row < matrixA->matrixNumRowTiles(); ++row) {
            for(int64_t col = 0; col < matrixA->matrixNumColTiles(); ++col) {
                printf("%2ld ", matrixB->owner(row, col));
            }
            printf("  ");
            for(int64_t col = 0; col < matrixC->matrixNumColTiles(); ++col) {
                printf("%2ld ", matrixC->owner(row, col));
            }
            printf("\b\n");
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
    }

    auto demo = [&deviceIds, &cudaDeviceProp, T, prodThreads, mpiComm](std::shared_ptr<MatrixA> matrixA, std::shared_ptr<MatrixB> matrixB, std::shared_ptr<MatrixC> matrixC, const int32_t iter) {
        auto MT = matrixC->matrixNumRowTiles(), KT = matrixA->matrixNumColTiles(), NT = matrixC->matrixNumColTiles();

        // Generate graph
        auto graph = hh::Graph<1, Triplet, TileC>("MM");

        auto inputStateManager  = std::make_shared<hh::StateManager<1, Triplet, MatrixA, MatrixB, MatrixC, Triplet>>(
            std::make_shared<InputState<MatrixType, IdA, IdB, IdC>>(),
            "InputStateManager",
            false
        );
        auto jobGenTask         = std::make_shared<GpuJobGeneratorTask<MatrixType, IdA, IdB, IdC>>(cudaDeviceProp.totalGlobalMem, prodThreads);
        jobGenTask->connectMemoryManager(std::make_shared<GpuTokenMemoryManager>(deviceIds));
        auto execPipeline       = std::make_shared<OuterProductExecutionPipeline<MatrixType, IdA, IdB, IdC, IdP>>(
            std::make_shared<OuterProductGpuGraph<MatrixType, IdA, IdB, IdC, IdP>>(T, prodThreads), deviceIds
        );
        auto compStateManager   = std::make_shared<OuterProductComputationStateManager<MatrixType, IdA, IdB, IdC, IdP>>(
            std::make_shared<OuterProductComputationState<MatrixType, IdA, IdB, IdC, IdP>>(MT, KT, NT)
        );
        auto accTask            = std::make_shared<AccumulateTask<MatrixType, IdC, IdP>>(prodThreads);
        auto dwTaskA            = std::make_shared<MatrixWarehouseTask<MatrixType, IdA>>();
        dwTaskA->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileA, int64_t, MemoryType>>(MT, T, MemoryType::CUDA_UNIFIED_MEMORY)
        );
        auto dwTaskB            = std::make_shared<MatrixWarehouseTask<MatrixType, IdB>>();
        dwTaskB->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileB, int64_t, MemoryType>>(NT, T, MemoryType::CUDA_UNIFIED_MEMORY)
        );

        graph.input<Triplet>(inputStateManager);
        graph.edge<MatrixA>(inputStateManager, dwTaskA);
        graph.edge<MatrixB>(inputStateManager, dwTaskB);
        graph.edge<MatrixC>(inputStateManager, compStateManager);
        graph.edge<Triplet>(inputStateManager, jobGenTask);
        graph.edge<DbRequest<IdA>>(jobGenTask, dwTaskA);
        graph.edge<DbRequest<IdB>>(jobGenTask, dwTaskB);
        graph.edge<TileA>(dwTaskA, jobGenTask);
        graph.edge<TileB>(dwTaskB, jobGenTask);
        graph.edge<Job>(jobGenTask, execPipeline);
        graph.edge<TileP>(execPipeline, compStateManager);
        graph.edge<Pair>(compStateManager, accTask);
        graph.edge<TileC>(accTask, compStateManager);
        graph.output<TileC>(compStateManager);
        graph.executeGraph();

        MPI_Barrier(mpiComm);
        graph.pushData(std::make_shared<Triplet>(std::make_tuple(matrixA, matrixB, matrixC)));
        graph.finishPushingData();

        graph.waitForTermination();
        graph.createDotFile(
            "./demo_v2_" + std::to_string(iter) + "_" + std::to_string(getNodeId()) + ".dot",
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::QUEUE,
            hh::DebugOptions::ALL,
            std::make_unique<hh::JetColor>(),
            false
        );

        double time = double((graph.core()->executionDuration() == std::chrono::nanoseconds::zero()? std::chrono::system_clock::now() - graph.core()->startExecutionTimeStamp(): graph.core()->executionDuration()).count())/1e9;
        double maxTime = 0;
        checkMpiErrors(MPI_Reduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiComm));
        return maxTime;
    };

    double times[10], gflops[10];
    for(int32_t iter = 0; iter < 10; ++iter) {
        gflops[iter]  = (2.*double(M)*double(K)*double(N))/1.e9;
        times[iter]   = demo(matrixA, matrixB, matrixC, iter);
        gflops[iter] /= times[iter];
        if(isRootNodeId()) printf("[%3d][ " CYAN("%9.3f") " gflops]["  RED("%8.3f") " secs ]\n", iter, gflops[iter], times[iter]);
    }

    if(isRootNodeId()) printf("[AVG][ " CYAN("%9.3f") " gflops]["  RED("%8.3f") " secs ]\n", std::accumulate(gflops, gflops+10, 0.)/10, std::accumulate(times, times+10, 0.)/10);

    return 0;
}

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
    printf("[p %ld][q %ld][M %ld][K %ld][N %ld][T %ld][prodThreads %ld]\n", p, q, M, K, N, T, prodThreads);
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
    auto dbTaskA            = std::make_shared<MatrixDbTask<MatrixType, IdA>>();
    auto dbTaskB            = std::make_shared<MatrixDbTask<MatrixType, IdB>>();

    graph.input<Triplet>(inputStateManager);
//    graph.edge<MatrixA>(inputStateManager, dbTaskA);
//    graph.edge<MatrixB>(inputStateManager, dbTaskB);
    graph.edge<MatrixC>(inputStateManager, compStateManager);
    graph.edge<Triplet>(inputStateManager, jobGenTask);
//    graph.edge<DbRequest<IdA>>(jobGenTask, dbTaskA);
//    graph.edge<DbRequest<IdB>>(jobGenTask, dbTaskB);
//    graph.edge<TileA>(dbTaskA, jobGenTask);
//    graph.edge<TileB>(dbTaskB, jobGenTask);
    graph.edge<Job>(jobGenTask, execPipeline);
    graph.edge<TileP>(execPipeline, compStateManager);
    graph.edge<Pair>(compStateManager, accTask);
    graph.edge<TileC>(accTask, compStateManager);
    graph.output<TileC>(compStateManager);
    graph.executeGraph();

    graph.pushData(std::make_shared<Triplet>(std::make_tuple(
        std::static_pointer_cast<MatrixA>(matrixA),
        std::static_pointer_cast<MatrixB>(matrixB),
        std::static_pointer_cast<MatrixC>(matrixC)
    )));
    graph.finishPushingData();

    LOG();
    graph.createDotFile(
        "./demo_v2_temp_" + std::to_string(getNodeId()) + ".dot",
        hh::ColorScheme::EXECUTION,
        hh::StructureOptions::QUEUE,
        hh::DebugOptions::ALL,
        std::make_unique<hh::JetColor>(),
        false
    );

    // clean up
    for(auto result = graph.getBlockingResult(); result != nullptr; result = graph.getBlockingResult()) {
        std::visit(hh::ResultVisitor{
            [&graph](std::shared_ptr<MatrixTile<MatrixType, IdC>> &tile) {
                if(!isRootNodeId()) return;
                printf("[node %ld] tileC(%ld, %ld)>>\n", getNodeId(), tile->rowIdx(), tile->colIdx());
                auto pData = (MatrixType*)tile->data();
                for(uint32_t r = 0; r < tile->height(); ++r) {
                    for(uint32_t c = 0; c < tile->width(); ++c) {
                        printf("%f ", pData[c*tile->height()+r]);
                    }
                    printf("\n");
                }
                printf("\n");
                fflush(stdout);
                graph.createDotFile(
                    "./demo_v2_temp_" + std::to_string(getNodeId()) + ".dot",
                    hh::ColorScheme::EXECUTION,
                    hh::StructureOptions::QUEUE,
                    hh::DebugOptions::ALL,
                    std::make_unique<hh::JetColor>(),
                    false
                );
            }},
        *result
        );
    }

    LOG();
    graph.waitForTermination();
    LOG();
    graph.createDotFile(
        "./demo_v2_" + std::to_string(getNodeId()) + ".dot",
        hh::ColorScheme::EXECUTION,
        hh::StructureOptions::QUEUE,
        hh::DebugOptions::ALL,
        std::make_unique<hh::JetColor>(),
        false
    );

    return 0;
}

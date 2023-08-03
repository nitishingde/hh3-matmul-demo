#include "tasks.h"
#include "utility.h"

int main(int argc, char *argv[]) {
    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv);

    constexpr char       IdA        = 'a';
    constexpr char       IdB        = 'b';
    constexpr char       IdC        = 'c';
    constexpr char       IdP        = 'p';
    constexpr MemoryType memoryType = MemoryType::HOST;
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

    auto matrixA = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdA>>(memoryType, M, K, T, p, q, mpiComm);
    auto matrixB = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdB>>(memoryType, K, N, T, p, q, mpiComm);

    printf("[p %ld][q %ld][M %ld][K %ld][N %ld][T %ld][MT %ld][KT %ld][NT %ld]\n", p, q, M, K, N, T, matrixA->matrixNumRowTiles(), matrixA->matrixNumColTiles(), matrixB->matrixNumColTiles());

    auto graph = hh::Graph<4, MatrixA, MatrixB, DbRequest<IdA>, DbRequest<IdB>, TileA, TileB>();
    auto taskA = std::make_shared<MatrixWarehouseTask<MatrixType, IdA>>();
    taskA->connectMemoryManager(std::make_shared<hh::StaticMemoryManager<TileA, int64_t, MemoryType>>(4, T, memoryType));
    auto taskB = std::make_shared<MatrixWarehouseTask<MatrixType, IdB>>();
    taskB->connectMemoryManager(std::make_shared<hh::StaticMemoryManager<TileB, int64_t, MemoryType>>(4, T, memoryType));

    graph.input<DbRequest<IdA>>(taskA);
    graph.input<DbRequest<IdB>>(taskB);
    graph.input<MatrixA>(taskA);
    graph.input<MatrixB>(taskB);
    graph.output<TileA>(taskA);
    graph.output<TileB>(taskB);
    graph.executeGraph();

    graph.pushData(std::static_pointer_cast<MatrixA>(matrixA));
    graph.pushData(std::static_pointer_cast<MatrixB>(matrixB));
    for(int64_t k = 0; k < matrixA->matrixNumRowTiles(); ++k) {
        for(int64_t row = 0; row < matrixB->matrixNumColTiles(); ++row) {
            graph.pushData(std::make_shared<DbRequest<IdA>>(row, k));
        }
        for(int64_t col = 0; col < matrixB->matrixNumColTiles(); ++col) {
            graph.pushData(std::make_shared<DbRequest<IdB>>(k, col));
        }
    }
    graph.pushData(std::make_shared<DbRequest<IdA>>(-1, -1, true));
    graph.pushData(std::make_shared<DbRequest<IdB>>(-1, -1, true));
    graph.finishPushingData();

    int32_t count = 0;
    for(auto result = graph.getBlockingResult(); result != nullptr; result = graph.getBlockingResult()) {
        std::visit(hh::ResultVisitor{
            [&graph, &count](std::shared_ptr<MatrixTile<MatrixType, IdA>> &tile) {
                if(isRootNodeId()) {
                    printf("[node %ld] tileA(%ld, %ld)>>\n", getNodeId(), tile->rowIdx(), tile->colIdx());
//                    auto pData = (MatrixType *) tile->data();
//                    for(uint32_t r = 0; r < tile->height(); ++r) {
//                        for(uint32_t c = 0; c < tile->width(); ++c) {
//                            printf("%f ", pData[c * tile->height() + r]);
//                        }
//                        printf("\n");
//                    }
//                    printf("\n");
                    fflush(stdout);
                }
                count++;
                if(tile->isMemoryManagerConnected()) tile->returnToMemoryManager();
            },
            [&graph, &count](std::shared_ptr<MatrixTile<MatrixType, IdB>> &tile) {
                if(isRootNodeId()) {
                    printf("[node %ld] tileB(%ld, %ld)>>\n", getNodeId(), tile->rowIdx(), tile->colIdx());
//                    auto pData = (MatrixType *) tile->data();
//                    for(uint32_t r = 0; r < tile->height(); ++r) {
//                        for(uint32_t c = 0; c < tile->width(); ++c) {
//                            printf("%f ", pData[c * tile->height() + r]);
//                        }
//                        printf("\n");
//                    }
//                    printf("\n");
                    fflush(stdout);
                }
                if(tile->isMemoryManagerConnected()) tile->returnToMemoryManager();
                count++;
            }},
           *result
        );
    }

    printf("[Mode %ld][Count = %d]\n", getNodeId(), count);
    fflush(stdout);
    graph.waitForTermination();

    graph.createDotFile(
        "./demo_v2_tests_" + std::to_string(getNodeId()) + ".dot",
        hh::ColorScheme::EXECUTION,
        hh::StructureOptions::QUEUE,
        hh::DebugOptions::ALL,
        std::make_unique<hh::JetColor>(),
        false
    );

    return 0;
}

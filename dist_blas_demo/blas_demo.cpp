#include "tasks.h"
#include "states.h"

//void test_db_task(int *argc, char **argv[]) {
//    MpiGlobalLockGuard mpiGlobalLockGuard(argc, argv);
//    using MatrixType = float;
//    constexpr char IdA = 'a';
//    constexpr char IdB = 'b';
//    constexpr uint32_t M = 16, N = 24, K = 22, T = 2;
////    constexpr char IdC = 'c';
//
//    auto graph = hh::Graph<4, MatrixContainer<MatrixType, IdA>, MatrixContainer<MatrixType, IdB>, DbRequest<IdA>, DbRequest<IdB>,  MatrixTile<MatrixType, IdA>, MatrixTile<MatrixType, IdB>>("Test DbTask");
//    auto dbTaskA = std::make_shared<MatrixDbTask<MatrixType, IdA>>();
//    auto dbTaskB = std::make_shared<MatrixDbTask<MatrixType, IdB>>();
//
//    auto mmA = std::make_shared<hh::StaticMemoryManager<MatrixTile<MatrixType, IdA>, uint64_t, MemoryType>>(1, T, MemoryType::HOST);
//    dbTaskA->connectMemoryManager(mmA);
//    auto mmB = std::make_shared<hh::StaticMemoryManager<MatrixTile<MatrixType, IdB>, uint64_t, MemoryType>>(1, T, MemoryType::HOST);
//    dbTaskB->connectMemoryManager(mmB);
//
//    graph.input<MatrixContainer<MatrixType, IdA>>(dbTaskA);
//    graph.input<MatrixContainer<MatrixType, IdB>>(dbTaskB);
//    graph.input<DbRequest<IdA>>(dbTaskA);
//    graph.input<DbRequest<IdB>>(dbTaskB);
//    graph.output<MatrixTile<MatrixType, IdA>>(dbTaskA);
//    graph.output<MatrixTile<MatrixType, IdB>>(dbTaskB);
//    graph.executeGraph();
//
//    auto matrixA = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdA>>(M, K, T);
//    graph.pushData(std::shared_ptr<MatrixContainer<MatrixType, IdA>>(matrixA));
//
//    auto matrixB = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdB>>(K, N, T);
//    graph.pushData(std::shared_ptr<MatrixContainer<MatrixType, IdB>>(matrixB));
//
//    graph.pushData(std::make_shared<DbRequest<IdA>>(0, 5, false));
//    graph.pushData(std::make_shared<DbRequest<IdA>>(3, 2, false));
//    graph.pushData(std::make_shared<DbRequest<IdB>>(2, 2, false));
//    graph.pushData(std::make_shared<DbRequest<IdB>>(5, 9, false));
//    graph.pushData(std::make_shared<DbRequest<IdA>>(-1, -1, true));
//    graph.pushData(std::make_shared<DbRequest<IdB>>(-1, -1, true));
//
//    graph.finishPushingData();
//    for(auto result = graph.getBlockingResult(); result != nullptr; result = graph.getBlockingResult()) {
//        std::visit(hh::ResultVisitor{
//            [](std::shared_ptr<MatrixTile<MatrixType, IdA>> &tile) {
//                if(isRootNodeId()) {
//                    printf("%c>>\n", IdA);
//                    auto pData = (MatrixType*)tile->data();
//                    for(uint32_t c = 0; c < tile->width(); ++c) {
//                        for(uint32_t r = 0; r < tile->height(); ++r) {
//                            printf("%f ", pData[c*tile->height()+r]);
//                        }
//                        printf("\n");
//                    }
//                    printf("\n");
//                }
//                fflush(stdout);
//                using namespace std::chrono_literals;
//                std::this_thread::sleep_for(1s);
//                if(tile->isMemoryManagerConnected()) tile->returnToMemoryManager();
//            },
//            [](std::shared_ptr<MatrixTile<MatrixType, IdB>> &tile) {
//                if(isRootNodeId()) {
//                    printf("%c>>\n", IdB);
//                    auto pData = (MatrixType*)tile->data();
//                    for(uint32_t c = 0; c < tile->width(); ++c) {
//                        for(uint32_t r = 0; r < tile->height(); ++r) {
//                            printf("%f ", pData[c*tile->height()+r]);
//                        }
//                        printf("\n");
//                    }
//                    printf("\n");
//                }
//                fflush(stdout);
//                using namespace std::chrono_literals;
//                std::this_thread::sleep_for(1s);
//                if(tile->isMemoryManagerConnected()) tile->returnToMemoryManager();
//            }},
//            *result
//        );
//    }
//    graph.waitForTermination();
//    graph.createDotFile("/home/ngs/google_drive/projects/hh3-matmul-demo/blas_demo/dbTask.dot");
//}

int main(int argc, char *argv[]) {
    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv);
    using MatrixType = float;
    constexpr char IdA = 'a';
    constexpr char IdB = 'b';
    constexpr char IdC = 'c';
    auto [P, Q, M, K, N, T, gemmThreads, path, host] = parseArgs(argc, argv);
    MPI_Comm mpiComm = MPI_COMM_WORLD;

    using MatrixA = MatrixContainer<MatrixType, IdA>;
    using MatrixB = MatrixContainer<MatrixType, IdB>;
    using MatrixC = MatrixContainer<MatrixType, IdC>;
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using TileC   = MatrixTile<MatrixType, IdC>;
    using Triplet = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>, std::shared_ptr<TileC>>;

    auto matrixA = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdA>>(M, K, T, P, Q, mpiComm);
    auto matrixB = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdB>>(K, N, T, P, Q, mpiComm);
    auto matrixC = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdC>>(M, N, T, P, Q, mpiComm);

    auto MT = matrixA->matrixNumRowTiles(), KT = matrixA->matrixNumColTiles(), NT = matrixB->matrixNumColTiles();
    printf("[node %ld] MT = %ld, KT = %ld, NT = %ld\n", getNodeId(), MT, KT, NT);

    auto graph = hh::Graph<3, MatrixA, MatrixB, MatrixC, TileC>("OpenBlas MM");

    // Initialize Tasks
    auto dbTaskA  = std::make_shared<MatrixDbTask<MatrixType, IdA>>();
    auto mmA = std::make_shared<hh::StaticMemoryManager<MatrixTile<MatrixType, IdA>, uint64_t, MemoryType>>(2+(M+T-1)/T, T, MemoryType::HOST);
    dbTaskA->connectMemoryManager(mmA);

    auto dbTaskB  = std::make_shared<MatrixDbTask<MatrixType, IdB>>();
    auto mmB = std::make_shared<hh::StaticMemoryManager<MatrixTile<MatrixType, IdB>, uint64_t, MemoryType>>(2+(N+T-1)/T, T, MemoryType::HOST);
    dbTaskB->connectMemoryManager(mmB);

    auto prodTask = std::make_shared<ProductTask<MatrixType, IdA, IdB, IdC>>(gemmThreads);

    // Initialize state and state-managers
    auto inputState        = std::make_shared<InputState<MatrixType, IdA, IdB, IdC>>(KT);
    auto inputStateManager = std::make_shared<InputStateManager<MatrixType, IdA, IdB, IdC>>(inputState);

    auto computationState        = std::make_shared<ComputationState<MatrixType, IdA, IdB, IdC>>(MT, KT, NT);
    auto computationStateManager = std::make_shared<ComputationStateManager<MatrixType, IdA, IdB, IdC>>(computationState);

    // build graphs
    graph.input<MatrixA>(dbTaskA);
    graph.input<MatrixB>(dbTaskB);
    graph.input<MatrixA>(inputStateManager);
    graph.input<MatrixB>(inputStateManager);
    graph.input<MatrixC>(inputStateManager);
    graph.edge<DbRequest<IdA>>(inputStateManager, dbTaskA);
    graph.edge<TileA>(dbTaskA, inputStateManager);
    graph.edge<DbRequest<IdB>>(inputStateManager, dbTaskB);
    graph.edge<TileB>(dbTaskB, inputStateManager);
    graph.edge<TileA>(inputStateManager, computationStateManager);
    graph.edge<TileB>(inputStateManager, computationStateManager);
    graph.edge<TileC>(inputStateManager, computationStateManager);
    graph.edge<Triplet>(computationStateManager, prodTask);
    graph.edge<Triplet>(prodTask, computationStateManager);
    graph.output<TileC>(computationStateManager);
    graph.executeGraph();

    // push data to the graph
    graph.pushData(std::static_pointer_cast<MatrixA>(matrixA));
    graph.pushData(std::static_pointer_cast<MatrixB>(matrixB));
    graph.pushData(std::static_pointer_cast<MatrixC>(matrixC));

    graph.finishPushingData();

    // clean up
    for(auto result = graph.getBlockingResult(); result != nullptr; result = graph.getBlockingResult()) {
        std::visit(hh::ResultVisitor{
            [](std::shared_ptr<MatrixTile<MatrixType, IdC>> &tile) {
//                if(!isRootNodeId()) return;
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
            }},
            *result
        );
    }
    graph.waitForTermination();
    graph.createDotFile(
        "./blas" + std::to_string(getNodeId()) + ".dot",
        hh::ColorScheme::EXECUTION,
        hh::StructureOptions::QUEUE,
        hh::DebugOptions::NONE,
        std::make_unique<hh::JetColor>(),
        false
    );

    return 0;
}

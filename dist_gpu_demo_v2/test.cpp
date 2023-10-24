#include "tasks.h"
#include "utility.h"
#include "matrix_utility.h"

void testMatrixWarehouseTask(int64_t p, int64_t q, int64_t M, int64_t K, int64_t N, int64_t T) {
    constexpr char       IdA        = 'a';
    constexpr char       IdB        = 'b';
    constexpr MemoryType memoryType = MemoryType::HOST;
    MPI_Comm             mpiComm    = MPI_COMM_WORLD;

    using MatrixType = float;
    using MatrixA    = MatrixContainer<MatrixType, IdA>;
    using MatrixB    = MatrixContainer<MatrixType, IdB>;
    using TileA      = MatrixTile<MatrixType, IdA>;
    using TileB      = MatrixTile<MatrixType, IdB>;

    auto matrixA = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdA>>(memoryType, M, K, T, p, q, mpiComm);
    auto matrixB = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdB>>(memoryType, K, N, T, p, q, mpiComm);

    printf("[p %ld][q %ld][M %ld][K %ld][N %ld][T %ld][MT %ld][KT %ld][NT %ld]\n", p, q, M, K, N, T, matrixA->matrixNumRowTiles(), matrixA->matrixNumColTiles(), matrixB->matrixNumColTiles());
    fflush(stdout);
    MPI_Barrier(mpiComm);

    if(isRootNodeId()) {
        printDataDistribution<MatrixType, IdA>(matrixA);
        printDataDistribution<MatrixType, IdB>(matrixB);
    }
    MPI_Barrier(mpiComm);

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
    for(int64_t k = 0; k < matrixA->matrixNumColTiles(); ++k) {
        for(int64_t row = 0; row < matrixB->matrixNumRowTiles(); ++row) {
            graph.pushData(std::make_shared<DbRequest<IdA>>(row, k));
        }
        for(int64_t col = 0; col < matrixB->matrixNumColTiles(); ++col) {
            graph.pushData(std::make_shared<DbRequest<IdB>>(k, col));
        }
    }
    graph.finishPushingData();

    int32_t countA = 0, countB = 0;
    for(auto result = graph.getBlockingResult(); result != nullptr; result = graph.getBlockingResult()) {
        std::visit(hh::ResultVisitor{
            [&countA](std::shared_ptr<MatrixTile<MatrixType, IdA>> &tile) {
                if(isRootNodeId()) {
                    printf("[node %ld][count %2d] tileA(%ld, %ld)>>\n", getNodeId(), countA, tile->rowIdx(), tile->colIdx());
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
                countA++;
                tile->ttl(0);
                if(tile->isMemoryManagerConnected()) tile->returnToMemoryManager();
            },
            [&countB](std::shared_ptr<MatrixTile<MatrixType, IdB>> &tile) {
                if(isRootNodeId()) {
                    printf("[node %ld][count %2d] tileB(%ld, %ld)>>\n", getNodeId(), countB, tile->rowIdx(), tile->colIdx());
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
                tile->ttl(0);
                if(tile->isMemoryManagerConnected()) tile->returnToMemoryManager();
                countB++;
            }},
            *result
        );
    }

    printf("[Mode %ld][CountA = %d][CountB = %d]\n", getNodeId(), countA, countB);
    fflush(stdout);
    graph.waitForTermination();

    graph.createDotFile(
        "./demo_v2_tests_" + std::to_string(getNodeId()) + ".dot",
        hh::ColorScheme::EXECUTION,
        hh::StructureOptions::QUEUE,
        hh::InputOptions::GATHERED,
        hh::DebugOptions::ALL,
        std::make_unique<hh::JetColor>(),
        false
    );
}

void testMatrixWarehousePrefetchTask(int64_t p, int64_t q, int64_t M, int64_t K, int64_t N, int64_t T) {
    constexpr char       IdA        = 'a';
    constexpr char       IdB        = 'b';
    constexpr char       IdC        = 'c';
    constexpr MemoryType memoryType = MemoryType::HOST;
    MPI_Comm             mpiComm    = MPI_COMM_WORLD;

    using MatrixType = float;
    using MatrixA    = MatrixContainer<MatrixType, IdA>;
    using MatrixB    = MatrixContainer<MatrixType, IdB>;
    using MatrixC    = MatrixContainer<MatrixType, IdC>;
    using MatAMatC   = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixC>>;
    using MatBMatC   = std::tuple<std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>;
    using TileA      = MatrixTile<MatrixType, IdA>;
    using TileB      = MatrixTile<MatrixType, IdB>;
    using TileC      = MatrixTile<MatrixType, IdC>;

    auto matrixA = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdA>>(memoryType, M, K, T, p, q, mpiComm);
    auto matrixB = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdB>>(memoryType, K, N, T, p, q, mpiComm);
    auto matrixC = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdC>>(memoryType, M, N, T, p, q, mpiComm);

    auto matrixInit = [](auto matrix) {
        for(int64_t rowIdx = 0; rowIdx < matrix->matrixNumRowTiles(); ++rowIdx) {
            for(int64_t colIdx = 0; colIdx < matrix->matrixNumColTiles(); ++colIdx) {
                if(auto tile = matrix->tile(rowIdx, colIdx)) {
                    auto pData = (MatrixType *)tile->data();
                    for(int64_t i = 0; i < tile->width()*tile->height(); ++i) pData[i] = MatrixType(getNodeId()*1000+rowIdx*100+colIdx*10);
                }
            }
        }
    };

    matrixInit(matrixA);
    matrixInit(matrixB);
    matrixInit(matrixC);

    std::mutex mpiMutex;
    std::atomic_int32_t stop = 2;

    auto graph = hh::Graph<4, MatAMatC, MatBMatC, DbRequest<IdA>, DbRequest<IdB>, TileA, TileB>();
    auto taskA = std::make_shared<MatrixWarehousePrefetchTask<MatrixType, IdA>>(&mpiMutex, &stop);
    taskA->connectMemoryManager(std::make_shared<hh::StaticMemoryManager<TileA, int64_t, MemoryType>>(4, T, memoryType));
    auto taskB = std::make_shared<MatrixWarehousePrefetchTask<MatrixType, IdB>>(&mpiMutex, &stop);
    taskB->connectMemoryManager(std::make_shared<hh::StaticMemoryManager<TileB, int64_t, MemoryType>>(4, T, memoryType));

    graph.input<MatAMatC>(taskA);
    graph.input<DbRequest<IdA>>(taskA);
    graph.input<MatBMatC>(taskB);
    graph.input<DbRequest<IdB>>(taskB);
    graph.output<TileA>(taskA);
    graph.output<TileB>(taskB);
    graph.executeGraph();

    graph.pushData(std::make_shared<MatAMatC>(std::make_tuple(
        std::static_pointer_cast<MatrixA>(matrixA),
        std::static_pointer_cast<MatrixC>(matrixC)
    )));
    graph.pushData(std::make_shared<MatBMatC>(std::make_tuple(
        std::static_pointer_cast<MatrixB>(matrixB),
        std::static_pointer_cast<MatrixC>(matrixC)
    )));

    std::set<int64_t> rowIndices = {};
    std::set<int64_t> colIndices = {};
    for(int64_t rowIdx = 0; rowIdx < matrixC->matrixNumRowTiles(); ++rowIdx) {
        for(int64_t colIdx = 0; colIdx < matrixC->matrixNumColTiles(); ++colIdx) {
            if(matrixC->tile(rowIdx, colIdx) != nullptr) {
                rowIndices.emplace(rowIdx);
                colIndices.emplace(colIdx);
            }
        }
    }

    std::vector<std::pair<int64_t, int64_t>> requestOrderA;
    std::vector<std::pair<int64_t, int64_t>> requestOrderB;
    for(int64_t k = 0; k < matrixA->matrixNumColTiles(); ++k) {
        for(auto rowIdx: rowIndices) {
            graph.pushData(std::make_shared<DbRequest<IdA>>(rowIdx, k));
            if(isRootNodeId()) {
                requestOrderA.emplace_back(std::move(std::make_pair(rowIdx, k)));
            }
        }
        for(auto colIdx: colIndices) {
            graph.pushData(std::make_shared<DbRequest<IdB>>(k, colIdx));
            if(isRootNodeId()) {
                requestOrderB.emplace_back(std::move(std::make_pair(k, colIdx)));
            }
        }
    }
    graph.finishPushingData();

    if(isRootNodeId()) {
        printDataDistribution<MatrixType, IdA, IdB, IdC>(matrixA, matrixB, matrixC);
        printf("A: ");
        for(const auto &vec2: requestOrderA) {
            printf("(%ld, %ld), ", vec2.first, vec2.second);
        }
        printf("\n");

        printf("B: ");
        for(const auto &vec2: requestOrderB) {
            printf("(%ld, %ld), ", vec2.first, vec2.second);
        }
        printf("\n");
        fflush(stdout);
    }

    int32_t countA = 0, countB = 0;
    for(auto result = graph.getBlockingResult(); result != nullptr; result = graph.getBlockingResult()) {
        std::visit(hh::ResultVisitor {
            [&countA](std::shared_ptr<MatrixTile<MatrixType, IdA>> &tile) {
                if(isRootNodeId()) {
                    printf("[node %ld][count %2d] tileA(%ld, %ld)>>\n", getNodeId(), countA, tile->rowIdx(), tile->colIdx());
                    auto pData = (MatrixType *) tile->data();
                    for(uint32_t r = 0; r < tile->height(); ++r) {
                        for(uint32_t c = 0; c < tile->width(); ++c) {
                            printf("%f ", pData[c * tile->height() + r]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                    fflush(stdout);
                }
                countA++;
                tile->ttl(0);
                if(tile->isMemoryManagerConnected()) tile->returnToMemoryManager();
            },
            [&countB](std::shared_ptr<MatrixTile<MatrixType, IdB>> &tile) {
                if(isRootNodeId() and false) {
                    printf("[node %ld][count %2d] tileB(%ld, %ld)>>\n", getNodeId(), countB, tile->rowIdx(), tile->colIdx());
                    auto pData = (MatrixType *) tile->data();
                    for(uint32_t r = 0; r < tile->height(); ++r) {
                        for(uint32_t c = 0; c < tile->width(); ++c) {
                            printf("%f ", pData[c * tile->height() + r]);
                        }
                        printf("\n");
                    }
                    printf("\n");
                    fflush(stdout);
                }
                tile->ttl(0);
                if(tile->isMemoryManagerConnected()) tile->returnToMemoryManager();
                countB++;
            }},
            *result
        );
    }

    printf("[Mode %ld][CountA = %d][CountB = %d]\n", getNodeId(), countA, countB);
    fflush(stdout);
    graph.waitForTermination();

    graph.createDotFile(
        "./demo_v2_tests_" + std::to_string(getNodeId()) + ".dot",
        hh::ColorScheme::EXECUTION,
        hh::StructureOptions::QUEUE,
        hh::InputOptions::GATHERED,
        hh::DebugOptions::ALL,
        std::make_unique<hh::JetColor>(),
        false
    );

}

int main(int argc, char *argv[]) {
    auto [p, q, M, K, N, T, productThreads, accumulateThreads, windowSize, lookAhead, computeTiles, path, host, resultsFile] = parseArgs(argc, argv);
    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv, p, q);

    testMatrixWarehouseTask(p, q, M, K, N, T);
    testMatrixWarehousePrefetchTask(p, q, M, K, N, T);

    return 0;
}

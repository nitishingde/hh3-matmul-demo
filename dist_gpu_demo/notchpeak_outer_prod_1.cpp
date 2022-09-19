#include "data/matrix_order.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"
#include "data/contiguous_sub_matrix_container.h"
#include "matrix_init.h"
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

    auto [M, K, N, tileSize, path] = parseArgs(argc, argv);
    printf("[Process %d] M = %d, K = %d, N = %d, tileSize = %d\n", getNodeId(), M, K, N, tileSize);

    int32_t devCount = 0;
    cudaGetDeviceCount(&devCount);
    devCount--;
    std::vector<int32_t> deviceIds;
    deviceIds.reserve(8);
    for(int32_t i = 0; i < devCount; ++i) deviceIds.emplace_back(i);
    printf("[Process %d] GPUs = %zu\n", getNodeId(), deviceIds.size());

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
        MMD_MpiOuterProduct1<MatrixType, 'a', 'b', 'c', Ord>().execute(subMatA, subMatB, matrixC, deviceIds);
        matrixC->shrink();
    }

#if VERIFY_MMD
    auto redundantMatrixC = std::make_shared<RedundantMatrixContainer<MatrixType, 'c', Ord>>(3, M, N, tileSize, matrixComm, isRootNodeId());
    init(redundantMatrixC);

    {
        MPI_Barrier(MPI_COMM_WORLD);
        MMD_VerifyCublas<MatrixType, 'a', 'b', 'c', Ord>().execute(std::move(subMatA), std::move(subMatB), redundantMatrixC, deviceIds);
        subMatA = nullptr;
        subMatB = nullptr;
    }

    MPI_Bcast(redundantMatrixC->data(), M*N, std::is_same_v<MatrixType, double>? MPI_DOUBLE: MPI_FLOAT, 0, redundantMatrixC->mpiComm());
    MPI_Barrier(MPI_COMM_WORLD);
    if(isRootNodeId()) printf("Verifying solution.\n");

    for(uint64_t i = 0; i < redundantMatrixC->matrixNumRowTiles(); ++i) {
        for(uint64_t j = 0; j < redundantMatrixC->matrixNumColTiles(); ++j) {
            if(auto tile = matrixC->getTile(i, j); tile == nullptr or tile->sourceNodeId() != getNodeId()) continue;
            else if(*tile != *redundantMatrixC->getTile(i, j)) {
                std::cerr << "[Error] tile @[" + std::to_string(i) + ", " + std::to_string(j) + "] don't match.\n";
#if not NDEBUG
                std::cout << GREEN("Actual  : ") << *redundantMatrixC->getTile(i, j);
                std::cout <<   RED("Computed: ") << *tile;
#endif
            }
        }
    }
#endif

#if DUMP_DATA
    for(uint64_t i = 0; i < matrixC->matrixNumColTiles(); ++i) {
        for(uint64_t j = 0; j < matrixC->matrixNumRowTiles(); ++j) {
            if(auto tile = matrixC->getTile(i, j); tile->sourceNodeId() == getNodeId()) {
                std::string fileName = path+"/matrix_tile_"+std::to_string(i)+"_"+std::to_string(j)+".dat";
                printf("[Process %d] Writing to file %s\n", getNodeId(), fileName.c_str());
                auto pFile = fopen(fileName.c_str(), "w");
                fwrite(tile->data(), sizeof(MatrixType), tile->dataSize(), pFile);
                fclose(pFile);
            }
        }
    }
#endif

    MPI_Comm_free(&matrixComm);

    return 0;
}

#include "data/matrix_order.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"
#include "data/contiguous_sub_matrix_container.h"
#include "mmd.h"
#include <cstdio>

template<Order SubMatrixOrder, class MatrixType, char Id, Order Ord>
void reset(std::shared_ptr<ContiguousSubMatrixContainer<SubMatrixOrder, MatrixType, Id, Ord>> subMat) {
    MatrixType i = 0;
    std::for_each(
        subMat->data(),
        subMat->data() + subMat->subMatrixHeight()*subMat->subMatrixWidth(),
        [&i](MatrixType &val) { val = i++; }
    );
}

int main([[maybe_unused]]int32_t argc, [[maybe_unused]]char **argv) {
    using MatrixType = double;
    constexpr Order Ord = Order::Col;
    using namespace std::chrono_literals;

    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv);

    MPI_Comm matrixComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &matrixComm);
    MPI_Barrier(matrixComm);

    auto [M, K, N, tileSize] = parseArgs(argc, argv);
    int32_t devCount = 0;
    cudaGetDeviceCount(&devCount);
    std::vector<int32_t> deviceIds;
    deviceIds.reserve(8);
    for(int32_t i = 0; i < devCount; ++i) deviceIds.emplace_back(i);
    printf("[Process %d] GPUs = %zu\n", getNodeId(), deviceIds.size());
    printf("[Process %d] M = %d, K = %d, N = %d, tileSize = %d\n", getNodeId(), M, K, N, tileSize);
    auto subMatA = std::make_shared<ContiguousSubMatrixContainer<Order::Col, MatrixType, 'a', Ord>>(0, M, K, tileSize, matrixComm);
    auto subMatB = std::make_shared<ContiguousSubMatrixContainer<Order::Row, MatrixType, 'b', Ord>>(1, K, N, tileSize, matrixComm);
    reset(subMatA);
    reset(subMatB);

    auto matrixC = std::make_shared<Cyclic2dMatrixContainer<MatrixType, 'c', Ord>>(2, M, N, tileSize, matrixComm);
    for(uint32_t i = 0; i < matrixC->matrixNumColTiles(); ++i) {
        for(uint32_t j = 0; j < matrixC->matrixNumRowTiles(); ++j) {
            if(auto tile = matrixC->getTile(i, j); tile != nullptr) {
                std::memset(tile->data(), 0, sizeof(MatrixType)*tile->dataSize());
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(isRootNodeId()) printf("Matrices initialized on every node\n");

    {
        MPI_Barrier(MPI_COMM_WORLD);
        MMD_MpiOuterProductCyclic2d<MatrixType, 'a', 'b', 'c', Ord>().execute(subMatA, subMatB, matrixC, deviceIds);
    }

    for(uint32_t i = 0; i < matrixC->matrixNumColTiles(); ++i) {
        for(uint32_t j = 0; j < matrixC->matrixNumRowTiles(); ++j) {
            if(auto tile = matrixC->getTile(i, j); tile->sourceNodeId() == getNodeId()) {
                std::string fileName = "matrix_tile_"+std::to_string(i)+"_"+std::to_string(j)+".dat";
                printf("[Process %d] Writing to file %s\n", getNodeId(), fileName.c_str());
                auto pFile = fopen(fileName.c_str(), "w");
                fwrite(tile->data(), sizeof(MatrixType), tile->dataSize(), pFile);
                fclose(pFile);
            }
        }
    }

    MPI_Comm_free(&matrixComm);

    return 0;
}

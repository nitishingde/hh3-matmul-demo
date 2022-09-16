#include "data/matrix_order.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"
#include "data/contiguous_sub_matrix_container.h"
#include "mmd.h"
#include <cstdio>

#define VERIFY_MMD true
#define DUMP_DATA  false

template<Order SubMatrixOrder, class MatrixType, char Id, Order Ord>
void init(std::shared_ptr<ContiguousSubMatrixContainer<SubMatrixOrder, MatrixType, Id, Ord>> subMat) {
    std::for_each(
        subMat->data(),
        subMat->data() + subMat->subMatrixHeight()*subMat->subMatrixWidth(),
        [](MatrixType &val) { val = fastrand()%10; }
    );
}

template<class MatrixType, char Id, Order Ord>
void init(std::shared_ptr<Cyclic2dMatrixContainer<MatrixType, Id, Ord>> cyclic2dMatrix) {
    for(uint32_t i = 0; i < cyclic2dMatrix->matrixNumColTiles(); ++i) {
        for(uint32_t j = 0; j < cyclic2dMatrix->matrixNumRowTiles(); ++j) {
            if(auto tile = cyclic2dMatrix->getTile(i, j); (tile != nullptr) and (tile->sourceNodeId() == getNodeId())) {
                std::for_each(tile->data(), tile->data()+tile->dataSize(), [](MatrixType &val) { val = getNodeId(); });
            }
        }
    }
}

template<class MatrixType, char Id, Order Ord>
void init(std::shared_ptr<RedundantMatrixContainer<MatrixType, Id, Ord>> redundantMatrix) {
    if(isRootNodeId()) {
        for(uint32_t idx = 0; idx < redundantMatrix->matrixNumRowTiles()*redundantMatrix->matrixNumColTiles(); ++idx) {
            uint32_t rowIdx = idx/redundantMatrix->matrixNumColTiles(), colIdx = idx%redundantMatrix->matrixNumColTiles();
            if(auto tile = redundantMatrix->getTile(rowIdx, colIdx); tile != nullptr) {
                if constexpr(Ord == Order::Col) {
                    uint32_t value = idx%getNumNodes();
                    for(uint32_t jj = 0; jj < tile->width(); ++jj) {
                        std::for_each(
                            &tile->data()[jj*tile->leadingDimension()],
                            &tile->data()[jj*tile->leadingDimension()+tile->height()],
                            [value](MatrixType &val) { val = value; }
                        );
                    }
                }
                else {
                    throw;
                }
            }
        }
    }
    else {
        std::memset(redundantMatrix->data(), 0, redundantMatrix->dataSize()*sizeof(MatrixType));
    }
}

int main([[maybe_unused]]int32_t argc, [[maybe_unused]]char **argv) {
    using MatrixType = double;
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
    for(uint32_t i = 0; i < redundantMatrixC->matrixNumRowTiles(); ++i) {
        for(uint32_t j = 0; j < redundantMatrixC->matrixNumColTiles(); ++j) {
            if(auto tile = matrixC->getTile(i, j); tile->sourceNodeId() != getNodeId()) continue;
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
    for(uint32_t i = 0; i < matrixC->matrixNumColTiles(); ++i) {
        for(uint32_t j = 0; j < matrixC->matrixNumRowTiles(); ++j) {
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

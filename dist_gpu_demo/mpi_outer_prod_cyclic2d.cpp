#include "data/matrix_order.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"
#include "data/contiguous_sub_matrix_container.h"
#include "mmd.h"

#define VERIFY_MM 1

template<Order SubMatrixOrder, class MatrixType, char Id, Order Ord>
void reset(std::shared_ptr<ContiguousSubMatrixContainer<SubMatrixOrder, MatrixType, Id, Ord>> subMat) {
    std::for_each(
        subMat->data(),
        subMat->data() + subMat->subMatrixHeight()*subMat->subMatrixWidth(),
        [](MatrixType &val) { val = fastrand()%10; }
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
    std::vector<int32_t> deviceIds = {getNodeId()};
//#if not NDEBUG
    printf("[Process %d] M = %d, K = %d, N = %d, tileSize = %d\n", getNodeId(), M, K, N, tileSize);
//#endif
    auto subMatA = std::make_shared<ContiguousSubMatrixContainer<Order::Col, MatrixType, 'a', Ord>>(0, M, K, tileSize, matrixComm);
    auto subMatB = std::make_shared<ContiguousSubMatrixContainer<Order::Row, MatrixType, 'b', Ord>>(1, K, N, tileSize, matrixComm);
    reset(subMatA);
    reset(subMatB);

    auto matrixC = std::make_shared<Cyclic2dMatrixContainer<MatrixType, 'c', Ord>>(2, M, N, tileSize, matrixComm);

    {
        MMD_MpiOuterProductCyclic2d<MatrixType, 'a', 'b', 'c', Ord>().execute(subMatA, subMatB, matrixC, deviceIds);
#if not NDEBUG
        if(isRootNodeId()) {
            for (int i = 0; i < matrixC->matrixNumRowTiles(); ++i) {
                for (int j = 0; j < matrixC->matrixNumColTiles(); ++j) {
                    if(auto tile = matrixC->getTile(i, j); tile->sourceNodeId() == getNodeId()) {
                        std::cout << *tile;
                    }
                }
            }
        }
        else {
            std::this_thread::sleep_for(100ms);
            printf("\n-------------------------------------------------------------------------------------------\n\n");
            for (int i = 0; i < matrixC->matrixNumRowTiles(); ++i) {
                for (int j = 0; j < matrixC->matrixNumColTiles(); ++j) {
                    if(auto tile = matrixC->getTile(i, j); tile->sourceNodeId() == getNodeId()) {
                        std::cout << *tile;
                    }
                }
            }
        }
#endif
    }

#if VERIFY_MM
    auto redundantMatrixC = std::make_shared<RedundantMatrixContainer<MatrixType, 'c', Ord>>(3, M, N, tileSize, matrixComm, isRootNodeId());

    {
        std::memset(redundantMatrixC->data(), 0, sizeof(MatrixType)*M*N);
        MMD_VerifyCublas<MatrixType, 'a', 'b', 'c', Ord>().execute(subMatA, subMatB, redundantMatrixC, deviceIds);
    }

    MPI_Bcast(redundantMatrixC->data(), M*N, std::is_same_v<MatrixType, double>? MPI_DOUBLE: MPI_FLOAT, 0, redundantMatrixC->mpiComm());
    if(getNodeId() == 1)
    for(uint32_t i = 0; i < redundantMatrixC->matrixNumRowTiles(); ++i) {
        for(uint32_t j = 0; j < redundantMatrixC->matrixNumColTiles(); ++j) {
            if(auto tile = matrixC->getTile(i, j); tile->sourceNodeId() != getNodeId()) continue;
            else if(*tile != *redundantMatrixC->getTile(i, j)) {
#if not NDEBUG
                if(isRootNodeId()) {
                    std::cout << *redundantMatrixC->getTile(i, j);
                    std::cout << *tile;
                }
#endif
                std::cout<<"[Error] tile @["+std::to_string(i)+", "+std::to_string(j)+"] don't match.\n";
            }
        }
    }
#endif

    MPI_Comm_free(&matrixComm);

    return 0;
}

#include "data/contiguous_sub_matrix_container.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"

int main(int argc, char **argv) {
    using MatrixType = double;
    constexpr Order Ord = Order::Col;
    using namespace std::chrono_literals;

    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv);
    MPI_Comm matrixComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &matrixComm);
    MPI_Barrier(matrixComm);

    auto subMatA = ContiguousSubMatrixContainer<Order::Row, MatrixType, 'A', Ord>(0, 128, 128, 32, matrixComm);
    auto subMatB = ContiguousSubMatrixContainer<Order::Col, MatrixType, 'B', Ord>(1, 128, 128, 32, matrixComm);
    auto redundantMatrix = RedundantMatrixContainer<MatrixType, 'C', Ord>(2, 128, 128, 32, matrixComm);
    auto cyclic2dMatrix = Cyclic2dMatrixContainer<MatrixType, 'C', Ord>(3, 128, 128, 32, matrixComm);

    auto tile1 = subMatA.getTile(0, 0);
    auto tile2 = subMatB.getTile(0, 0);
    auto tile3 = redundantMatrix.getTile(0, 0);
    auto tile4 = cyclic2dMatrix.getTile(0, 0);

    return 0;
}
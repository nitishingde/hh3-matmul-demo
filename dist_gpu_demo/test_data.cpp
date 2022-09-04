#undef NDEBUG

#include "data/contiguous_sub_matrix_container.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"

int main(int argc, char **argv) {
    using MatrixType = double;
    constexpr Order Ord = Order::Col;
    using namespace std::chrono_literals;

    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv);
    assert(getNumNodes() == 4);

    auto matrixComm = MPI_COMM_WORLD;
    MPI_Barrier(matrixComm);

    auto subMatA = std::make_shared<ContiguousSubMatrixContainer<Order::Col, MatrixType, 'A', Ord>>(0, 75, 78, 8, matrixComm);
    for(uint32_t i = 0; i < subMatA->subMatrixWidth()*subMatA->subMatrixHeight(); ++i) {
        subMatA->data()[i] = i;
    }

    auto subMatB = std::make_shared<ContiguousSubMatrixContainer<Order::Row, MatrixType, 'B', Ord>>(1, 78, 75, 8, matrixComm);
    for(uint32_t i = 0; i < subMatB->subMatrixWidth()*subMatB->subMatrixHeight(); ++i) {
        subMatB->data()[i] = i;
    }

    switch(getNodeId()) {
        case 0:
            assert(subMatA->subMatrixHeight() == 75);
            assert(subMatA->subMatrixWidth()  == 24);
            assert(subMatA->subMatrixNumRowTiles() == 10);
            assert(subMatA->subMatrixNumColTiles() == 3);
            assert(std::make_tuple(uint32_t(0), uint32_t(10)) == subMatA->subMatrixRowTileRange());
            assert(std::make_tuple(uint32_t(0), uint32_t(3)) == subMatA->subMatrixColTileRange());

            assert(subMatB->subMatrixHeight() == 24);
            assert(subMatB->subMatrixWidth()  == 75);
            assert(subMatB->subMatrixNumRowTiles() == 3);
            assert(subMatB->subMatrixNumColTiles() == 10);
            assert(std::make_tuple(uint32_t(0), uint32_t(3)) == subMatB->subMatrixRowTileRange());
            assert(std::make_tuple(uint32_t(0), uint32_t(10)) == subMatB->subMatrixColTileRange());
            break;

        case 1:
            assert(subMatA->subMatrixHeight() == 75);
            assert(subMatA->subMatrixWidth()  == 24);
            assert(subMatA->subMatrixNumRowTiles() == 10);
            assert(subMatA->subMatrixNumColTiles() == 3);
            assert(std::make_tuple(uint32_t(0), uint32_t(10)) == subMatA->subMatrixRowTileRange());
            assert(std::make_tuple(uint32_t(3), uint32_t(6)) == subMatA->subMatrixColTileRange());

            assert(subMatB->subMatrixHeight() == 24);
            assert(subMatB->subMatrixWidth()  == 75);
            assert(subMatB->subMatrixNumRowTiles() == 3);
            assert(subMatB->subMatrixNumColTiles() == 10);
            assert(std::make_tuple(uint32_t(3), uint32_t(6)) == subMatB->subMatrixRowTileRange());
            assert(std::make_tuple(uint32_t(0), uint32_t(10)) == subMatB->subMatrixColTileRange());
            break;

        case 2:
            assert(subMatA->subMatrixHeight() == 75);
            assert(subMatA->subMatrixWidth()  == 16);
            assert(subMatA->subMatrixNumRowTiles() == 10);
            assert(subMatA->subMatrixNumColTiles() == 2);
            assert(std::make_tuple(uint32_t(0), uint32_t(10)) == subMatA->subMatrixRowTileRange());
            assert(std::make_tuple(uint32_t(6), uint32_t(8)) == subMatA->subMatrixColTileRange());

            assert(subMatB->subMatrixHeight() == 16);
            assert(subMatB->subMatrixWidth()  == 75);
            assert(subMatB->subMatrixNumRowTiles() == 2);
            assert(subMatB->subMatrixNumColTiles() == 10);
            assert(std::make_tuple(uint32_t(6), uint32_t(8)) == subMatB->subMatrixRowTileRange());
            assert(std::make_tuple(uint32_t(0), uint32_t(10)) == subMatB->subMatrixColTileRange());
            break;

        case 3:
            assert(subMatA->subMatrixHeight() == 75);
            assert(subMatA->subMatrixWidth()  == 14);
            assert(subMatA->subMatrixNumRowTiles() == 10);
            assert(subMatA->subMatrixNumColTiles() == 2);
            assert(std::make_tuple(uint32_t(0), uint32_t(10)) == subMatA->subMatrixRowTileRange());
            assert(std::make_tuple(uint32_t(8), uint32_t(10)) == subMatA->subMatrixColTileRange());

            assert(subMatB->subMatrixHeight() == 14);
            assert(subMatB->subMatrixWidth()  == 75);
            assert(subMatB->subMatrixNumRowTiles() == 2);
            assert(subMatB->subMatrixNumColTiles() == 10);
            assert(std::make_tuple(uint32_t(8), uint32_t(10)) == subMatB->subMatrixRowTileRange());
            assert(std::make_tuple(uint32_t(0), uint32_t(10)) == subMatB->subMatrixColTileRange());
            break;
    }

    if(isRootNodeId()) printf("Test passed!\n");

    return 0;
}
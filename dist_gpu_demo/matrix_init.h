#ifndef HH3_MATMUL_MATRIX_INIT_H
#define HH3_MATMUL_MATRIX_INIT_H

#include "data/contiguous_sub_matrix_container.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"

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
            if(auto tile = cyclic2dMatrix->getTile(i, j); tile != nullptr) {
                if(tile->sourceNodeId() == getNodeId()) {
                    std::for_each(tile->data(), tile->data()+tile->dataSize(), [](MatrixType &val) { val = getNodeId(); });
                }
                else {
                    std::for_each(tile->data(), tile->data()+tile->dataSize(), [](MatrixType &val) { val = 0; });
                }
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


#endif //HH3_MATMUL_MATRIX_INIT_H

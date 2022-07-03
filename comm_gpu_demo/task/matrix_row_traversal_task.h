#ifndef HH3_MATMUL_MATRIX_ROW_TRAVERSAL_TASK_H
#define HH3_MATMUL_MATRIX_ROW_TRAVERSAL_TASK_H

#include "../data/matrix_block_data.h"

template<class MatrixType, char Id, Order Ord>
class MatrixRowTraversalTask: public hh::AbstractTask<1,
        MatrixData<MatrixType, Id, Ord>,       //inp1
        MatrixBlockData<MatrixType, Id, Ord>   //out1
    > {
public:
    explicit MatrixRowTraversalTask():
        hh::AbstractTask<1, MatrixData<MatrixType, Id, Ord>, MatrixBlockData<MatrixType, Id, Ord>>(
            "Row Traversal Task",
            1,
            false
        ) {}

    void execute(std::shared_ptr<MatrixData<MatrixType, Id, Ord>> matrixDataB) override {
        for(size_t row = 0; row < matrixDataB->numBlocksRows(); ++row) {
            for(size_t col = 0; col < matrixDataB->numBlocksCols(); ++col) {
                if constexpr(Ord == Order::Column) {
                    this->addResult(std::make_shared<MatrixBlockData<MatrixType, Id, Ord>>(
                        row, col,
                        std::min(matrixDataB->blockSize(), matrixDataB->matrixHeight() - (row * matrixDataB->blockSize())),
                        std::min(matrixDataB->blockSize(), matrixDataB->matrixWidth() - (col * matrixDataB->blockSize())),
                        matrixDataB->leadingDimension(),
                        *matrixDataB->data(),
                        *(matrixDataB->data() + (col * matrixDataB->blockSize()) * matrixDataB->leadingDimension() + row * matrixDataB->blockSize())
                    ));
                }
            }
        }
    }
};

#endif //HH3_MATMUL_MATRIX_ROW_TRAVERSAL_TASK_H

#ifndef HH3_MATMUL_MATRIX_COLUMN_TRAVERSAL_TASK_H
#define HH3_MATMUL_MATRIX_COLUMN_TRAVERSAL_TASK_H

#include "../data/matrix_block_data.h"

template<class MatrixType, char Id, Order Ord>
class MatrixColumnTraversalTask: public hh::AbstractTask<1,
        MatrixData<MatrixType, Id, Ord>,       //inp1
        MatrixBlockData<MatrixType, Id, Ord>   //out1
    > {
public:
    explicit MatrixColumnTraversalTask():
        hh::AbstractTask<1, MatrixData<MatrixType, Id, Ord>, MatrixBlockData<MatrixType, Id, Ord>>(
            "Column Traversal Task",
            1,
            false
        ) {}

    void execute(std::shared_ptr<MatrixData<MatrixType, Id, Ord>> matrixDataA) override {
        for(size_t col = 0; col < matrixDataA->numBlocksCols(); ++col) {
            for(size_t row = 0; row < matrixDataA->numBlocksRows(); ++row) {
                if constexpr(Ord == Order::Column) {
                    this->addResult(std::make_shared<MatrixBlockData<MatrixType, Id, Ord>>(
                        row, col,
                        std::min(matrixDataA->blockSize(), matrixDataA->matrixHeight() - (row * matrixDataA->blockSize())),
                        std::min(matrixDataA->blockSize(), matrixDataA->matrixWidth() - (col * matrixDataA->blockSize())),
                        matrixDataA->leadingDimension(),
                        *matrixDataA->data(),
                        *(matrixDataA->data() + (col * matrixDataA->blockSize()) * matrixDataA->leadingDimension() + row * matrixDataA->blockSize())
                    ));
                }
            }
        }
    }
};

#endif //HH3_MATMUL_MATRIX_COLUMN_TRAVERSAL_TASK_H

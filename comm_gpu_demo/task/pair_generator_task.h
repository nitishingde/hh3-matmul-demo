// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.

#ifndef HH3_MATMUL_PAIR_GENERATOR_TASK_H
#define HH3_MATMUL_PAIR_GENERATOR_TASK_H

#include <hedgehog/hedgehog.h>
#include "../data/matrix_block_data.h"

template<class MatrixType, Order Ord>
using BlockTriplets = std::tuple<
        std::shared_ptr<MatrixBlockData<MatrixType, 'a', Ord>>,
        std::shared_ptr<MatrixBlockData<MatrixType, 'b', Ord>>,
        std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>
    >;

template<class MatrixType, Order Ord>
class PairGeneratorTask:
    public hh::AbstractTask<1,
        MatrixBlockData<MatrixType, 'c', Ord>,  //inp1
        BlockTriplets<MatrixType, Ord>          //out1
    > {
private:
    const std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> matrixA_ = nullptr;
    const std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> matrixB_ = nullptr;
public:
    explicit PairGeneratorTask(
        const std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
        const std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB
    ):  matrixA_(matrixA),
        matrixB_(matrixB),
        hh::AbstractTask<1, MatrixBlockData<MatrixType, 'c', Ord>, BlockTriplets<MatrixType, Ord>>(
            "PairGenerator Task",
            1,
            false
        ) {}

    void execute(std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>> blockC) override {
        size_t row = blockC->rowIdx(), col = blockC->colIdx(), blockSize = blockC->blockSizeWidth();
        size_t K = matrixA_->matrixWidth() / blockSize;
        for(int k = 0; k < K; ++k) {
            auto blockP = std::static_pointer_cast<MatrixBlockData<MatrixType, 'p', Ord>>(this->getManagedMemory());
            blockP->rowIdx(row);
            blockP->colIdx(col);
            blockP->blockSizeWidth(blockSize);
            blockP->blockSizeHeight(blockSize);
            blockP->leadingDimension(blockSize);
            blockP->ttl(1);

            this->addResult(std::make_shared<BlockTriplets<MatrixType, Ord>>(std::make_tuple(
                std::make_shared<MatrixBlockData<MatrixType, 'a', Ord>>(
                    row, k,
                    std::min(blockSize, matrixA_->matrixHeight() - (row * blockSize)),
                    std::min(blockSize, matrixA_->matrixWidth() - (k * blockSize)),
                    matrixA_->leadingDimension(),
                    *matrixA_->data(),
                    *(matrixA_->data() + (k * blockSize) * matrixA_->leadingDimension() + row * blockSize)
                ),
                std::make_shared<MatrixBlockData<MatrixType, 'b', Ord>>(
                    k, col,
                    std::min(blockSize, matrixB_->matrixHeight() - (k * blockSize)),
                    std::min(blockSize, matrixB_->matrixWidth() - (col * blockSize)),
                    matrixB_->leadingDimension(),
                    *matrixB_->data(),
                    *(matrixB_->data() + (col * blockSize) * matrixB_->leadingDimension() + k * blockSize)
                ),
                blockP
            )));
        }
    }

    std::shared_ptr<hh::AbstractTask<1, MatrixBlockData<MatrixType, 'c', Ord>, BlockTriplets<MatrixType, Ord>>>
    copy() override {
        return std::make_shared<PairGeneratorTask>(matrixA_, matrixB_);
    }
};

#endif //HH3_MATMUL_PAIR_GENERATOR_TASK_H

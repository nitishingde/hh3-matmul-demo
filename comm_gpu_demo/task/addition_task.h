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


#ifndef HH3_MATMUL_ADDITION_TASK_H
#define HH3_MATMUL_ADDITION_TASK_H

#include "../data/matrix_block_data.h"

template<class MatrixType, Order Ord>
class AdditionTask: public hh::AbstractTask<1,
        std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>,
        MatrixBlockData<MatrixType, 'c', Ord>
    > {
public:
    explicit AdditionTask(size_t threadCount):
        hh::AbstractTask<1,
            std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>,
            std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>, MatrixBlockData<MatrixType, 'c', Ord>
        >("Addition Task", threadCount, false) {}

    void execute(std::shared_ptr<std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>> blockPair) override {
        // TODO
        auto blockC = blockPair->first;
        auto blockP = blockPair->second;
        assert(blockC->blockSizeWidth() == blockP->blockSizeWidth());
        assert(blockC->blockSizeHeight() == blockP->blockSizeHeight());

        if constexpr(Ord == Order::Row) {
            for(size_t i = 0; i < blockC->blockSizeHeight(); ++i) {
                for(size_t j = 0; j < blockC->blockSizeWidth(); ++j) {
                    blockC->blockData()[i * blockC->leadingDimension() + j] += blockP->blockData()[i * blockP->leadingDimension() + j];
                }
            }
        } else {
            for(size_t j = 0; j < blockC->blockSizeWidth(); ++j) {
                for(size_t i = 0; i < blockC->blockSizeHeight(); ++i) {
                    blockC->blockData()[j * blockC->leadingDimension() + i] += blockP->blockData()[j * blockP->leadingDimension() + i];
                }
            }
        }

        blockP->used();
        blockP->returnToMemoryManager();

        this->addResult(blockC);
    }

    std::shared_ptr<hh::AbstractTask<1,
        std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>,
        MatrixBlockData<MatrixType, 'c', Ord>
    >>
    copy() override {
        return std::make_shared<AdditionTask>(this->numberThreads());
    };
};

#endif //HH3_MATMUL_ADDITION_TASK_H

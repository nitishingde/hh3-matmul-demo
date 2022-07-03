
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

        delete[] blockP->blockData();
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

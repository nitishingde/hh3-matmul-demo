#ifndef HEDGEHOG_TUTORIALS_ACCUMULATE_TASK_H
#define HEDGEHOG_TUTORIALS_ACCUMULATE_TASK_H

#include <hedgehog/hedgehog.h>
#include "../data/matrix_block_data.h"


template<class Type, char Id, Order Ord>
class AccumulateTask: public hh::AbstractTask<1, MatrixBlockData<Type, Id, Ord>, void*> {
private:
    std::vector<std::vector<std::shared_ptr<MatrixBlockData<Type, Id, Ord>>>> rootMatrix_;
    std::vector<std::shared_ptr<MatrixBlockData<Type, Id, Ord>>> queue_;
    std::atomic_uint32_t count_ = 0;
    uint32_t rootCount_ = 0;
    uint32_t rootSize;
    uint32_t expectedCount_;

    bool doesRootMatrixExist(size_t row, size_t col) {
        return rootMatrix_[row][col] != nullptr;
    }

    void addAndUpdate(const std::shared_ptr<MatrixBlockData<Type, Id, Ord>> &other) {
        uint32_t row = other->rowIdx(), col = other->colIdx();
        auto &c = rootMatrix_[row][col];

        if constexpr (Ord == Order::Row) {
            for (size_t i = 0; i < c->blockSizeHeight(); ++i) {
                for (size_t j = 0; j < c->blockSizeWidth(); ++j) {
                    c->blockData()[i * c->leadingDimension() + j] += other->get(i, j);
                }
            }
            count_++;
        }
        else {
            for (size_t j = 0; j < c->blockSizeWidth(); ++j) {
                for (size_t i = 0; i < c->blockSizeHeight(); ++i) {
                    c->blockData()[j * c->leadingDimension() + i] += other->get(i, j);
                }
            }
            count_++;
        }
    }
public:
    explicit AccumulateTask(size_t rows, size_t cols, size_t partialCopies):
        hh::AbstractTask<1, MatrixBlockData<Type, Id, Ord>, void*>("AccumulateTask", 1, false),
        rootSize(rows*cols),
        expectedCount_(rows*cols*partialCopies) {
        rootMatrix_.resize(rows, std::vector<std::shared_ptr<MatrixBlockData<Type, Id, Ord>>>(cols, nullptr));
    }

    void execute(std::shared_ptr<MatrixBlockData<Type, Id, Ord>> data) override {
        auto rowIdx = data->rowIdx();
        auto colIdx = data->colIdx();

        if(data.get()->fullMatrixData() != nullptr) {
            rootMatrix_[rowIdx][colIdx] = data;
            rootCount_++;
        }
        else if(doesRootMatrixExist(rowIdx, colIdx)) {
            addAndUpdate(data);
        } else {
            queue_.emplace_back(data);
        }

        // one time clean up
        if(!queue_.empty() and rootCount_ == rootSize) {
            for(const auto &blockData: queue_) {
                addAndUpdate(blockData);
            }
            queue_.clear();
        }
        if(canTerminate()) {
            this->addResult(std::make_shared<void*>(nullptr));//FIXME
        }
    }

    bool canTerminate() const override {
        return count_ == expectedCount_;
    }
};


#endif //HEDGEHOG_TUTORIALS_ACCUMULATE_TASK_H

#ifndef HEDGEHOG_TUTORIALS_ACCUMULATE_TASK_H
#define HEDGEHOG_TUTORIALS_ACCUMULATE_TASK_H

#include <hedgehog/hedgehog.h>
#include "../data/matrix_block_data.h"


template<class Type, char Id, Order Ord>
class AccumulateTask: public hh::AbstractTask<1, MatrixBlockData<Type, Id, Ord>, void*> {
private:
    std::vector<std::vector<std::shared_ptr<MatrixBlockData<Type, Id, Ord>>>> rootMatrix_;
    std::vector<std::vector<std::vector<std::shared_ptr<MatrixBlockData<Type, Id, Ord>>>>> queue_;
    std::atomic_uint32_t count_ = 0;
    uint32_t expectedCount_;
    std::mutex mutex_;

    bool doesRootMatrixExist(size_t row, size_t col) {
        return rootMatrix_[row][col] != nullptr;
    }

    void execute(size_t row, size_t col) {
        std::lock_guard lockGuard(mutex_);
        auto &c = rootMatrix_[row][col];
        if constexpr (Ord == Order::Row) {
            for(auto &p: queue_[row][col]) {
                for (size_t i = 0; i < c->blockSizeHeight(); ++i) {
                    for (size_t j = 0; j < c->blockSizeWidth(); ++j) {
                        c->blockData()[i * c->leadingDimension() + j] += p->get(i, j);
                    }
                }
                count_++;
            }
        }
        else {
            for(auto &p: queue_[row][col]) {
                for (size_t j = 0; j < c->blockSizeWidth(); ++j) {
                    for (size_t i = 0; i < c->blockSizeHeight(); ++i) {
                        c->blockData()[j * c->leadingDimension() + i] += p->get(j, i);
                    }
                }
                count_++;
            }
        }
        queue_[row][col].clear();
    }

public:
    explicit AccumulateTask(size_t rows, size_t cols, size_t expectedCount):
        hh::AbstractTask<1, MatrixBlockData<Type, Id, Ord>, void*>("AccumulateTask", 1, false),
        expectedCount_(expectedCount) {
        rootMatrix_.resize(rows, std::vector<std::shared_ptr<MatrixBlockData<Type, Id, Ord>>>(cols, nullptr));
        queue_.resize(rows, std::vector<std::vector<std::shared_ptr<MatrixBlockData<Type, Id, Ord>>>>(cols));
    }

    void execute(std::shared_ptr<MatrixBlockData<Type, Id, Ord>> data) override {
        auto rowIdx = data->rowIdx();
        auto colIdx = data->colIdx();

        if(data.get()->fullMatrixData() != nullptr) {
            rootMatrix_[rowIdx][colIdx] = data;
        }
        else {
            std::lock_guard lockGuard(mutex_);
            queue_[rowIdx][colIdx].emplace_back(data);
        }

        if(doesRootMatrixExist(rowIdx, colIdx)) {
            execute(rowIdx, colIdx);
        }

        uint32_t temp = count_;
        if(canTerminate()) {
            this->addResult(std::make_shared<void*>(nullptr));//FIXME
        }
    }

    bool canTerminate() const override {
        return count_ == expectedCount_;
    }
};


#endif //HEDGEHOG_TUTORIALS_ACCUMULATE_TASK_H

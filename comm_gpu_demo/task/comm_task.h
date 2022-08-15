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


#ifndef HH3_MATMUL_COMM_TASK_H
#define HH3_MATMUL_COMM_TASK_H

#include "comm/comm.h"
#include "../data/serialization.h"
#include <atomic>
#include <list>

template<class Type>
uint32_t getTypeId() {
    return uint32_t(typeid(Type).hash_code());
}

template<class T>
class TypeResolver {
private:
    static_assert(std::is_base_of_v<Serialization, T>, "ReceiverTask template parameters needs to inherit from Serialization!");
public:
    std::shared_ptr<T> get(const comm::SignalType &recvData) {
        if(getTypeId<T>() == recvData->id) {
            auto data = std::make_shared<T>();
            data.get()->deserialize(std::istringstream(recvData->serializedData));
            return data;
        }

        return std::shared_ptr<T>(nullptr);
    }
};

template<typename ...Args>
class ReceiverTask:
        public hh::AbstractTask<1, void*, Args...>,
        public TypeResolver<Args> ... {
private:
    std::atomic_uint32_t count_ = 0;
    uint32_t expectedCount_;
    template<class T>
    void addIfNotNull(T data) {
        if(data != nullptr) {
            this->count_++;
            this->addResult(data);
        }
    }

public:
    explicit ReceiverTask(uint32_t expectedCountOfMessages):
        hh::AbstractTask<1, void*, Args...>("ReceiverTask", 1, true),
        expectedCount_(expectedCountOfMessages) {}

    ~ReceiverTask() override = default;

    void execute([[maybe_unused]]std::shared_ptr<void*> data) override {
        comm::connectReceiver(&ReceiverTask::commSlot, this);
        using namespace std::chrono_literals;
        while(!canTerminate()) {
            std::this_thread::sleep_for(6ms);
        }
    }

    void commSlot(comm::SignalType recvData) {
        (addIfNotNull(TypeResolver<Args>::get(recvData)), ...);
    }

    [[nodiscard]] bool canTerminate() const override {
        return count_ == expectedCount_;
    }
};

template<class Data>
class SenderTask: public hh::AbstractTask<1, Data, void*> {
private:
    uint32_t destId_;
    std::atomic_uint32_t count_ = 0;
    uint32_t expectedCount_;
public:
    SenderTask(uint32_t destId, uint32_t expectedCount):
        hh::AbstractTask<1, Data, void*>("SenderTask", 1, false),
        destId_(destId),
        expectedCount_(expectedCount) {}

    void execute(std::shared_ptr<Data> data) override {
        static_assert(std::is_base_of_v<Serialization, Data>, "SenderTask template parameters needs to inherit from Serialization!");
        comm::sendMessage(getTypeId<Data>(), data.get()->serialize(), destId_);
        count_++;
        if(count_ == expectedCount_) {
            this->addResult(std::make_shared<void*>(nullptr));
        }
    }

    [[nodiscard]] bool canTerminate() const override {
        return count_ == expectedCount_;
    }
};

template<class MatrixType, char Id, Order Ord>
class MatrixBlockReceiverTask: public hh::AbstractTask<1, void*, MatrixBlockData<MatrixType, Id, Ord>> {
private:
    size_t M_ = 0;
    size_t N_ = 0;
    size_t blockSize_ = 0;
    int32_t expectedCount_ = 0;
    struct Request {
        std::shared_ptr<MatrixBlockData<MatrixType, Id, Ord>> block_ = nullptr;
        MPI_Request mpiRequest_ {};
    };
    std::list<Request> requests_;
    std::mutex mutex_ {};

private:
    void daemon() {
        using namespace std::chrono_literals;
        while(!canTerminate()) {
            std::this_thread::sleep_for(1ms);
            std::lock_guard lc(mutex_);
            for(auto req = requests_.begin(); req != requests_.end();) {
                int32_t flag;
                MPI_Status status;
                MPI_Test(&req->mpiRequest_, &flag, &status);

                if(flag) {
                    auto block = req->block_;
                    int32_t tag = status.MPI_TAG;
                    block->colIdx(tag & 0xffff);
                    tag >>= 16;
                    block->rowIdx(tag);
                    block->blockSizeHeight(std::min(blockSize_, M_ - block->rowIdx() * blockSize_));
                    block->blockSizeWidth(std::min(blockSize_, N_ - block->colIdx() * blockSize_));
                    req = requests_.erase(req);
                    this->addResult(block);
                }
                else if(status.MPI_ERROR) {
                    int errorLen;
                    char error[256];
                    MPI_Error_string(status.MPI_ERROR, error, &errorLen);
                    printf("[Process 0][Error %s]\n", error);
                }
            }
        }
    }

public:
    explicit MatrixBlockReceiverTask(size_t M, size_t N, size_t blockSize, size_t expectedCount):
        hh::AbstractTask<1, void *, MatrixBlockData<MatrixType, Id, Ord>>("MatrixBlock Receiver", 1, true),
        M_(M), N_(N), blockSize_(blockSize),
        expectedCount_(expectedCount) {}

    void execute(std::shared_ptr<void*> data) override {
        std::thread daemon(&MatrixBlockReceiverTask::daemon, this);
        for(; expectedCount_; --expectedCount_) {
            // actually collect data
            auto block = std::static_pointer_cast<MatrixBlockData<MatrixType, Id, Ord>>(this->getManagedMemory());
            std::lock_guard lc(mutex_);
            requests_.template emplace_back(Request{});
            auto &request = requests_.back();
            request.block_ = block;
            block->ttl(1);
            MPI_Irecv(
                block->blockData(),
                blockSize_*blockSize_, MPI_DOUBLE,
                MPI_ANY_SOURCE, MPI_ANY_TAG,
                MPI_COMM_WORLD,
                &request.mpiRequest_
            );
        }
        daemon.join();
    }

    [[nodiscard]] bool canTerminate() const override {
        if(expectedCount_ < 0) std::runtime_error("MatrixBlockReceiverTask::expectedCount_ < 0\n");
        return expectedCount_ == 0 and requests_.empty();
    }
};

template<class MatrixType, char Id, Order Ord>
class MatrixBlockSenderTask: public hh::AbstractTask<1, MatrixBlockData<MatrixType, Id, Ord>, void*> {
private:
    size_t blockSize_ = 0;
    int32_t expectedCount_ = 0;//FIXME: is it needed?

public:
    explicit MatrixBlockSenderTask(size_t numOfThreads, size_t blockSize, size_t expectedCount):
            hh::AbstractTask<1, MatrixBlockData<MatrixType, Id, Ord>, void*>("MatrixBlock Sender", numOfThreads, false),
            blockSize_(blockSize),
            expectedCount_(expectedCount) {}

    void execute(std::shared_ptr<MatrixBlockData<MatrixType, Id, Ord>> block) override {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto tempBlock = std::static_pointer_cast<MatrixBlockData<MatrixType, Id, Ord>>(this->getManagedMemory());
        for(int j = 0; j < block->blockSizeWidth(); ++j) {
            std::memcpy(&tempBlock->blockData()[j*block->blockSizeHeight()], &block->blockData()[j*block->leadingDimension()], sizeof(MatrixType)*block->blockSizeHeight());
        }
        int32_t tag = int16_t(block->rowIdx());
        tag <<= 16;
        tag |= int16_t(block->colIdx());
        MPI_Send(
            tempBlock->blockData(),
            blockSize_*blockSize_, MPI_DOUBLE,
            0, tag,
            MPI_COMM_WORLD
        );
        tempBlock->returnToMemoryManager();
    }

    std::shared_ptr<hh::AbstractTask<1, MatrixBlockData<MatrixType, Id, Ord>, void*>>
    copy() override {
        return std::make_shared<MatrixBlockSenderTask>(this->numberThreads(), blockSize_, expectedCount_);
    }
};

#endif //HH3_MATMUL_COMM_TASK_H

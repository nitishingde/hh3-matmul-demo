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


#ifndef HH3_MATMUL_COMM_TASKS_H
#define HH3_MATMUL_COMM_TASKS_H

#include "../data/matrix_tile.h"
#include <hedgehog/hedgehog.h>

template<class MatrixType, char Id, Order Ord>
class Cyclic2dReceiverTask: public hh::AbstractTask<1, void*, MatrixTile<MatrixType, Id, Ord>> {
public:
    explicit Cyclic2dReceiverTask(uint64_t expectedCount):
        hh::AbstractTask<1, void *, MatrixTile<MatrixType, Id, Ord>>("Cyclic2d Receiver Task", 1, true),
        expectedCount_(expectedCount) {}

    void execute(std::shared_ptr<void*> data) override {
        std::thread daemon(&Cyclic2dReceiverTask::daemon, this);
        for(; expectedCount_; --expectedCount_) {
            // actually collect data
            auto tile = std::static_pointer_cast<MatrixTile<MatrixType, Id, Ord>>(this->getManagedMemory());
            auto dataPacket = tile->dataPacket();
            std::lock_guard lc(mutex_);
            requests_.template emplace_back(Request{});
            auto &request = requests_.back();
            request.tile_ = tile;
            checkMpiErrors(MPI_Irecv(
                dataPacket->data(), dataPacket->size(), MPI_UINT8_T,
                MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                &request.mpiRequest_
            ));
        }
        daemon.join();
    }

    [[nodiscard]] bool canTerminate() const override {
//        printf("[Process %d] canTerminate %d\n", getNodeId(), expectedCount_ == 0 and requests_.empty());
        if(expectedCount_ < 0) throw std::runtime_error("MatrixBlockReceiverTask::expectedCount_ < 0\n");
        return expectedCount_ == 0 and requests_.empty();
    }

private:
    void daemon() {
        using namespace std::chrono_literals;
        while(!canTerminate()) {
            std::this_thread::sleep_for(1ms);
            std::lock_guard lc(mutex_);
            for(auto req = requests_.begin(); req != requests_.end();) {
                int32_t flag;
                MPI_Status status;
                checkMpiErrors(MPI_Test(&req->mpiRequest_, &flag, &status));
                if(flag) {
                    auto tile = req->tile_;
                    tile->dataPacket()->contextId(status.MPI_TAG);
                    tile->unPackDataPacket();
                    tile->ttl(1);
                    req = requests_.erase(req);
                    this->addResult(tile);
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

private:
    int64_t expectedCount_ = 0;
    struct Request {
        std::shared_ptr<MatrixTile<MatrixType, Id, Ord>> tile_ = nullptr;
        MPI_Request mpiRequest_ {};
    };
    std::list<Request> requests_;
    std::mutex mutex_ {};
};

template<class MatrixType, char Id, Order Ord>
class Cyclic2dSenderTask: public hh::AbstractTask<1, MatrixTile<MatrixType, Id, Ord>, void*> {
public:
    explicit Cyclic2dSenderTask(uint32_t numOfThreads, uint64_t expectedCount):
            hh::AbstractTask<1, MatrixTile<MatrixType, Id, Ord>, void*>("Cyclic2d Sender Task", numOfThreads, false),
            expectedCount_(expectedCount) {}

    void execute(std::shared_ptr<MatrixTile<MatrixType, Id, Ord>> matrixTile) override {
        if(matrixTile->sourceNodeId() == getNodeId()) return;
        auto dataPacket = matrixTile->dataPacket();

        auto start = std::chrono::high_resolution_clock::now();
        checkMpiErrors(MPI_Send(
            dataPacket->data(), dataPacket->size(), MPI_UINT8_T,
            matrixTile->sourceNodeId(), matrixTile->contextId(), MPI_COMM_WORLD
        ));
        auto end = std::chrono::high_resolution_clock::now();
        double bandWidth = double(dataPacket->size())/(double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1.e9);
        bandWidth /= (1024*1024);// MB/s

        minBW_ = std::min(minBW_, bandWidth);
        totalSent_++;
        avgBW_ = (avgBW_*(totalSent_-1) + bandWidth)/totalSent_;
        maxBW_ = std::max(maxBW_, bandWidth);
    }

    std::string extraPrintingInformation() const override {
        if(maxBW_ < 0.0) return "";

        auto min = std::to_string(minBW_);
        auto avg = std::to_string(avgBW_);
        auto max = std::to_string(maxBW_);
        return "Min Bandwidth: " + min.substr(0, min.find('.', 0)+4) + " MB/s\\n"
            + "Avg Bandwidth: " + avg.substr(0, avg.find('.', 0)+4) + " MB/s\\n"
            + "Max Bandwidth: " + max.substr(0, max.find('.', 0)+4) + " MB/s";
    }

    std::shared_ptr<hh::AbstractTask<1, MatrixTile<MatrixType, Id, Ord>, void*>>
    copy() override {
        return std::make_shared<Cyclic2dSenderTask>(this->numberThreads(), expectedCount_);
    }

private:
    double minBW_ = 1e9, avgBW_ = 0.0, maxBW_ = -1.0;
    int64_t totalSent_ = 0;
    int64_t expectedCount_ = 0;//FIXME: is it needed?
};


#endif //HH3_MATMUL_COMM_TASKS_
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


#endif //HH3_MATMUL_COMM_TASK_H

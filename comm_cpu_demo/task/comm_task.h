#ifndef HEDGEHOG_TUTORIALS_COMM_TASK_H
#define HEDGEHOG_TUTORIALS_COMM_TASK_H


#include "../../cmake-build-debug/_deps/uintah2-src/include/comm/comm.h" //FIXME
#include "../data/serialization.h"
#include <hedgehog/hedgehog.h>

template<class Type>
uint32_t getTypeId() {
    return uint32_t(typeid(Type).hash_code());
}

template<class T>
class TypeResolver {
private:
    static_assert(std::is_base_of_v<Serialization, T>, "ReceiverTask template parameters needs to inherit from Serialization!");
public:
    std::shared_ptr<T> get(const std::shared_ptr<comm::CommPacket> &recvData) {
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

    ~ReceiverTask() override {
        comm::Communicator::signal.disconnect();
    }

    void execute([[maybe_unused]]std::shared_ptr<void*> data) override {
        comm::Communicator::signal.connect(&ReceiverTask::commSlot, this);
        using namespace std::chrono_literals;
        while(!canTerminate()) {
            std::this_thread::sleep_for(100ms);
        }
    }

    void commSlot(std::shared_ptr<comm::CommPacket> recvData) {
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
        comm::Communicator::sendMessage(getTypeId<Data>(), data.get()->serialize(), destId_);
        count_++;
    }

    bool canTerminate() const override {
        return count_ == expectedCount_;
    }
};


#endif //HEDGEHOG_TUTORIALS_COMM_TASK_H

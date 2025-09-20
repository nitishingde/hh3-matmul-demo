#ifndef HH3_MATMUL_COMMON_TASKS_H
#define HH3_MATMUL_COMMON_TASKS_H

#include <array>
#include "common_data.h"

using namespace std::chrono_literals;

template<typename MatrixType, char Id>
class MatrixWarehouseTask: public hh::AbstractTask<2, MatrixContainer<MatrixType, Id>, DbRequest<Id>, MatrixTile<MatrixType, Id>> {
private:
    using Matrix     = MatrixContainer<MatrixType, Id>;
    using DB_Request = DbRequest<Id>;
    using Tile       = MatrixTile<MatrixType, Id>;

    class InterNodeRequest {
    private:
        using MetaDataBuffer = std::vector<int64_t>;
        enum {
            SIZE   = 0,
            QUIT   = 1,
            BEGIN  = 2,
            STRIDE = 3,
            ROW    = 0,
            COL    = 1,
            TAG    = 2,
            BUFFER = 2048,
        };

    public:
        explicit InterNodeRequest() {
            metaDataBuffer_.resize(BUFFER, 0);
        }

        explicit InterNodeRequest(const int64_t otherNode, const std::vector<std::tuple<int64_t, int64_t>> &&indices, bool quit = false):
            otherNode_(otherNode) {
            metaDataBuffer_.resize(BUFFER, 0);
            assert((indices.size()*STRIDE + BEGIN) < metaDataBuffer_.size());

            metaDataBuffer_[SIZE] = indices.size();
            metaDataBuffer_[QUIT] = quit;
            int32_t i = BEGIN;
            for(const auto [rowIdx, colIdx]: indices) {
                metaDataBuffer_[i + ROW]  = rowIdx;
                metaDataBuffer_[i + COL]  = colIdx;
                metaDataBuffer_[i + TAG]  = tagGenerator();
                i                        += STRIDE;
            }
        }

        // Getters
        [[nodiscard]] bool                       quit()             const { return metaDataBuffer_[QUIT]; }
        [[nodiscard]] int64_t                    batchSize()        const { return metaDataBuffer_[SIZE]; }
        [[nodiscard]] auto&                      metaDataBuffer()         { return metaDataBuffer_;       }
        [[nodiscard]] int64_t                    otherNode()        const { return otherNode_;            }
        [[nodiscard]] MPI_Request*               mpiRequestHandle()       { return &mpiRequest_;          }

        //Setters
        void quit(int64_t node) {
            otherNode_            = node;
            metaDataBuffer_[SIZE] = 0;
            metaDataBuffer_[QUIT] = true;
        }

        std::tuple<int64_t, int64_t, int64_t> operator[](int64_t idx) {
            assert(idx < this->batchSize());
            auto ptr = &metaDataBuffer_[BEGIN + idx*STRIDE];
            return std::make_tuple(ptr[ROW], ptr[COL], ptr[TAG]);
        }

        class Iterator {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type        = std::tuple<int64_t, int64_t, int64_t>;
            using difference_type   = std::ptrdiff_t;
            using pointer           = std::tuple<int64_t, int64_t, int64_t>*;
            using reference         = std::tuple<int64_t, int64_t, int64_t>&;

            explicit Iterator(int64_t *pContainerData): pContainerData_(pContainerData) {}

            // pre-increment
            Iterator& operator++() {
                pContainerData_ += STRIDE;
                return *this;
            }

            // post-increment
            Iterator& operator++(int) {
                Iterator ret = *this;
                pContainerData_ += STRIDE;
                return ret;
            }

            bool operator==(const Iterator &other) const {
                return pContainerData_ == other.pContainerData_;
            }
            bool operator!=(const Iterator &other) const {
                return pContainerData_ != other.pContainerData_;
            }

            value_type operator*() const {
                return std::make_tuple(pContainerData_[ROW], pContainerData_[COL], pContainerData_[TAG]);
            }

        private:
            int64_t *pContainerData_ = nullptr;
        };

        Iterator begin() { return Iterator(&metaDataBuffer_[BEGIN]);                            }
        Iterator end()   { return Iterator(&metaDataBuffer_[BEGIN + this->batchSize()*STRIDE]); }

    private:
        MetaDataBuffer        metaDataBuffer_ = {};
        int64_t               otherNode_      = -1;
        MPI_Request           mpiRequest_     = {};
    };

    struct InterNodeResponse {
        MPI_Request           mpiRequest = {};
        std::shared_ptr<Tile> pData      = nullptr;
    };

    struct IncomingResponse {
        InterNodeRequest  *pInterNodeRequest = nullptr;
        int64_t           currentIndex       = -1;
    };

public:
    explicit MatrixWarehouseTask(): hh::AbstractTask<2, Matrix, DB_Request, Tile>("Matrix DB Task", 1, false) {
        liveNodeCounter_.store(-1);
        canTerminate_.store(false);
    }

    ~MatrixWarehouseTask() override = default;

    void execute(std::shared_ptr<Matrix> matrix) override {
        assert(matrix_ == nullptr);

        matrix_ = matrix;
        daemon_ = std::thread(&MatrixWarehouseTask::daemon, this);
        liveNodeCounter_.store(matrix_->numNodes());
        canTerminate_.store(false);

        for(; !dbRequests_.empty(); dbRequests_.pop_front()) {
            handleDbRequest(dbRequests_.front());
        }
    }

    void execute(std::shared_ptr<DB_Request> dbRequest) override {
        if(matrix_ == nullptr) {
            dbRequests_.emplace_back(dbRequest);
            return;
        }

        handleDbRequest(dbRequest);
    }

    [[nodiscard]] bool canTerminate() const override {
        return hh::AbstractTask<2, Matrix, DB_Request, Tile>::canTerminate() and canTerminate_.load();
    }

private:
    void handleDbRequest(std::shared_ptr<DB_Request> dbRequest) {
        // scoped lock: mutex_, mpiMutex
        /** send meta data asynchronously and enqueue the future **/{
            auto request   = InterNodeRequest(dbRequest->srcNode, std::move(dbRequest->indices), dbRequest->quit);
            auto &mdBuffer = request.metaDataBuffer();
            if(dbRequest->srcNode != matrix_->nodeId()) {
                // metadata is pretty small (16KB) therefore eager protocol will be used by MPI while sending this
                // buffer, hence a blocking send is good enough here.
                auto mpiLg = std::lock_guard(mpiMutex);
                checkMpiErrors(MPI_Isend(&mdBuffer[0], mdBuffer.size(), MPI_INT64_T, request.otherNode(), Id, matrix_->mpiComm(), request.mpiRequestHandle()));
            }
            auto lg = std::lock_guard(mutex_);
            outgoingRequests_.emplace_back(std::move(request));
        }

        if(!dbRequest->quit) return;

        /** send quit requests to the other nodes **/{
            auto lg    = std::lock_guard(mutex_);
            auto mpiLg = std::lock_guard(mpiMutex);
            for(int64_t node = 0; node < matrix_->numNodes(); ++node) {
                if(node == matrix_->nodeId() or node == dbRequest->srcNode) continue;

                auto request   = InterNodeRequest();
                auto &mdBuffer = request.metaDataBuffer();
                request.quit(node);
                checkMpiErrors(MPI_Isend(&mdBuffer[0], mdBuffer.size(), MPI_INT64_T, request.otherNode(), Id, matrix_->mpiComm(), request.mpiRequestHandle()));
                outgoingRequests_.emplace_back(std::move(request));
            }
            liveNodeCounter_.fetch_sub(1);
        }
        daemon_.join();
    }

    void processOutgoingRequests() {
        // scoped lock: mutex_, mpiMutex
        if(incomingResponse_.pInterNodeRequest != nullptr) return;

        auto lg = std::lock_guard(mutex_);
        if(auto it = outgoingRequests_.begin(); it != outgoingRequests_.end()) {
            auto &request = *it;

            // tiles are available locally
            if(request.otherNode() == getNodeId()) {
                for(auto [rowIdx, colIdx, tag]: request) {
                    this->addResult(matrix_->tile(rowIdx, colIdx));
                }
                outgoingRequests_.pop_front();
                return;
            }

            int32_t    done      = false;
            MPI_Status mpiStatus = {};
            auto       mpiLg     = std::lock_guard(mpiMutex);
            checkMpiErrors(MPI_Test(request.mpiRequestHandle(), &done, &mpiStatus));

            if(done) {
                incomingResponse_.pInterNodeRequest = &request;
                incomingResponse_.currentIndex      = 0;
            }
        }
    }

    void processCurrentIncomingResponse() {
        // no scoped locking
        if(incomingResponse_.pInterNodeRequest == nullptr or this->memoryManager()->currentSize() == 0) return;

        auto& request = *incomingResponse_.pInterNodeRequest;
        if(incomingResponse_.currentIndex < request.batchSize()) {
            MPI_Status mpiStatus               = {};
            auto       [rowIdx, colIdx, tagId] = request[incomingResponse_.currentIndex];
            auto       tile                    = std::static_pointer_cast<Tile>(this->getManagedMemory());
            tile->init(rowIdx, colIdx, matrix_->tileHeight(rowIdx, colIdx), matrix_->tileWidth(rowIdx, colIdx));
            tile->memoryState(MemoryState::ON_HOLD);
            auto mpiLg = std::lock_guard(mpiMutex);
            checkMpiErrors(MPI_Recv(
                tile->data(),
                (int)tile->byteSize(),
                MPI_BYTE,
                request.otherNode(),
                tagId,
                matrix_->mpiComm(),
                &mpiStatus
            ));
            tile->memoryState(MemoryState::SHARED);
            this->addResult(tile);
            incomingResponse_.currentIndex++;
        }

        if(incomingResponse_.currentIndex == request.batchSize()) {
            auto lg = std::lock_guard(mutex_);
            outgoingRequests_.pop_front();
            incomingResponse_.pInterNodeRequest = nullptr;
            incomingResponse_.currentIndex      = -1;
        }
    }

    void processIncomingRequests() {
        // no scoped locking
        int32_t          flag           = false;
        InterNodeRequest request          {};
        auto&            metaDataBuffer = request.metaDataBuffer();
        MPI_Status       mpiStatus      = {};

        auto mpiLg = std::lock_guard(mpiMutex);
        checkMpiErrors(MPI_Iprobe(MPI_ANY_SOURCE, Id, matrix_->mpiComm(), &flag, &mpiStatus));

        if(flag) {
            MPI_Status mpiRecvStatus;
            checkMpiErrors(MPI_Recv(&metaDataBuffer[0], metaDataBuffer.size(), MPI_INT64_T, mpiStatus.MPI_SOURCE, Id, matrix_->mpiComm(), &mpiRecvStatus));
            checkMpiErrors(mpiRecvStatus.MPI_ERROR);

            if(request.quit()) {
                liveNodeCounter_.fetch_sub(1);
            }
            for(const auto [rowIdx, colIdx, tagId]: request) {
                InterNodeResponse response = {};
                response.pData = matrix_->tile(rowIdx, colIdx);
                checkMpiErrors(MPI_Issend(
                    response.pData->data(),
                    response.pData->byteSize(),
                    MPI_BYTE,
                    mpiStatus.MPI_SOURCE,
                    tagId,
                    matrix_->mpiComm(),
                    &response.mpiRequest
                ));
                outgoingResponses_.emplace_back(response);
            }
        }
    }

    void processOutgoingResponses() {
        // no scoped locking
        auto mpiLg = std::lock_guard(mpiMutex);
        for(auto it = outgoingResponses_.begin(); it != outgoingResponses_.end(); ) {
            auto& mpiRequest = it->mpiRequest;
            int32_t flag;
            MPI_Status mpiStatus;
            if(checkMpiErrors(MPI_Test(&mpiRequest, &flag, &mpiStatus)); flag) {
                it = outgoingResponses_.erase(it);
            }
            else {
                it++;
            }
        }
    }

    void updateState() {
        // no scoped locking
        auto lg = std::lock_guard(mutex_);
        canTerminate_.store(true
            and dbRequests_.empty()
            and outgoingRequests_.empty()
            and outgoingResponses_.empty()
            and (liveNodeCounter_.load() == 0)
            and (incomingResponse_.pInterNodeRequest == nullptr)
        );
    }

    void daemon() {
        using namespace std::chrono_literals;
        while(!canTerminate()) {
            processIncomingRequests();
            processOutgoingResponses();
            processOutgoingRequests();
            processCurrentIncomingResponse();
            updateState();
            std::this_thread::sleep_for(4ms);
        }
    }

private:
    std::atomic_bool                       canTerminate_       = false;
    std::shared_ptr<Matrix>                matrix_             = nullptr;
    std::thread                            daemon_             = {};
    std::mutex                             mutex_              = {};
    std::list<std::shared_ptr<DB_Request>> dbRequests_         = {};// may need to use mutex_
    std::list<InterNodeRequest>            outgoingRequests_   = {};// need to use mutex_
    std::list<InterNodeResponse>           outgoingResponses_  = {};
    std::atomic_int64_t                    liveNodeCounter_    = {};
    IncomingResponse                       incomingResponse_   = {};
};

template<typename MatrixType, char Id>
class MatrixWarehouseBatchedTask: public hh::AbstractTask<2, MatrixContainer<MatrixType, Id>, DwBatchRequest<Id>, MatrixTile<MatrixType, Id>> {
private:
    using Matrix    = MatrixContainer<MatrixType, Id>;
    using DwRequest = DwBatchRequest<Id>;
    using Tile      = MatrixTile<MatrixType, Id>;

    enum State {
        STALL  = 0,
        FREE   = 0,
        LOCAL  = 1,
        REMOTE = 2,
    };

    class InterNodeRequest {
    private:
        using MetaDataBuffer = std::vector<int32_t>;
        enum {
            SIZE   = 0,
            QUIT   = 1,
            BEGIN  = 2,
            STRIDE = 3,
            ROW    = 0,
            COL    = 1,
            TAG    = 2,
        };

    public:
        explicit InterNodeRequest() {
            init();
        }

        // Getters
        [[nodiscard]] bool                       quit()             const { return metaDataBuffer_[QUIT]; }
        [[nodiscard]] int64_t                    batchSize()        const { return metaDataBuffer_[SIZE]; }
        [[nodiscard]] auto&                      metaDataBuffer()         { return metaDataBuffer_;       }
        [[nodiscard]] int64_t                    otherNode()        const { return otherNode_;            }
        [[nodiscard]] MPI_Request*               mpiRequestHandle()       { return &mpiRequest_;          }

        //Setters
        void init() {
            metaDataBuffer_.resize(16*1024, 0);
            otherNode_ = -1;
        }

        void otherNode(int64_t val) { otherNode_ = val; }

        void addIndex(int32_t rowIdx, int32_t colIdx, int32_t tagId) {
            auto *pData = &metaDataBuffer_[BEGIN+batchSize()*STRIDE];
            pData[ROW] = rowIdx;
            pData[COL] = colIdx;
            pData[TAG] = tagId;
            metaDataBuffer_[SIZE] += 1;
        }

        void quit(bool val) {
            metaDataBuffer_[QUIT] = val;
        }

        std::tuple<int64_t, int64_t, int64_t> operator[](int64_t idx) {
            assert(idx < this->batchSize());
            auto ptr = &metaDataBuffer_[BEGIN + idx*STRIDE];
            return std::make_tuple(ptr[ROW], ptr[COL], ptr[TAG]);
        }

        //Getters
        MPI_Datatype getMpiDataType() { return MPI_INT32_T; }

        class Iterator {
        public:
            using iterator_category = std::random_access_iterator_tag;
            using value_type        = std::tuple<int32_t, int32_t, int32_t>;
            using difference_type   = std::ptrdiff_t;
            using pointer           = std::tuple<int32_t, int32_t, int32_t>*;
            using reference         = std::tuple<int32_t, int32_t, int32_t>&;

            explicit Iterator(int32_t *pContainerData): pContainerData_(pContainerData) {}

            // pre-increment
            Iterator& operator++() {
                pContainerData_ += STRIDE;
                return *this;
            }

            // post-increment
            Iterator& operator++(int) {
                Iterator ret = *this;
                pContainerData_ += STRIDE;
                return ret;
            }

            bool operator==(const Iterator &other) const {
                return pContainerData_ == other.pContainerData_;
            }
            bool operator!=(const Iterator &other) const {
                return pContainerData_ != other.pContainerData_;
            }

            value_type operator*() const {
                return std::make_tuple(pContainerData_[ROW], pContainerData_[COL], pContainerData_[TAG]);
            }

        private:
            int32_t *pContainerData_ = nullptr;
        };

        Iterator begin() { return Iterator(&metaDataBuffer_[BEGIN]);                            }
        Iterator end()   { return Iterator(&metaDataBuffer_[BEGIN + this->batchSize()*STRIDE]); }

    private:
        MetaDataBuffer        metaDataBuffer_ = {};
        int64_t               otherNode_      = -1;
        MPI_Request           mpiRequest_     = {};
    };

public:
    explicit MatrixWarehouseBatchedTask(): hh::AbstractTask<2, Matrix, DwRequest, Tile>("Matrix Warehouse", 1, false) {
        canTerminate_.store(State::STALL);
        liveNodeCounter_.store(getNumNodes());
    }

    void execute(std::shared_ptr<Matrix> matrix) override {
        matrix_         = matrix;
        daemon_         = std::thread(&MatrixWarehouseBatchedTask::daemon, this);
        receiverDaemon_ = std::thread(&MatrixWarehouseBatchedTask::receiverDaemon, this);
        canTerminate_.store(State::STALL);
        liveNodeCounter_.store(matrix->numNodes());
    }

    void execute(std::shared_ptr<DwRequest> dwBatchRequest) override {
        using namespace std::chrono_literals;
        assert(currentRequest_ == nullptr);
        std::vector<bool>             metaDataReceiverMask(matrix_->numNodes(), 0);
        std::vector<InterNodeRequest> outgoingRequests(matrix_->numNodes());//FIXME: use hash map like std::unordered_map?
        std::vector<int32_t>          priority;

        for(const auto [rowIdx, colIdx, tagId]: dwBatchRequest->data) {
            auto otherNode = matrix_->owner(rowIdx, colIdx);
            if(otherNode == getNodeId()) continue;
            outgoingRequests[otherNode].addIndex(rowIdx, colIdx, tagId);
            if(!metaDataReceiverMask[otherNode]) priority.emplace_back(otherNode);
            metaDataReceiverMask[otherNode] = true;
        }

        currentRequest_ = dwBatchRequest;
        canProcessResponses_.store(State::LOCAL);

        std::vector<MPI_Request> mpiRequests = {};
        std::vector<MPI_Status>  mpiStatuses = {};

        if(const auto receiverCount = priority.size(); 0 < receiverCount) {
            mpiRequests.resize(receiverCount, MPI_REQUEST_NULL);
            /** start sending meta data **/{
                auto mpiLg = std::lock_guard(mpiMutex);
                int32_t i = 0;
                for(const auto otherNode: priority) {
                    auto &request = outgoingRequests[otherNode];
                    auto &buffer = request.metaDataBuffer();
                    request.quit(dwBatchRequest->quit);
                    checkMpiErrors(MPI_Isend(
                        buffer.data(),
                        buffer.size(),
                        request.getMpiDataType(),
                        otherNode,
                        Id,
                        matrix_->mpiComm(),
                        &mpiRequests[i]
                    ));
                    i++;
                }
            }

            // test the 1st (top priority) metadata send separately
            for(int flag = 0; !flag;) {
                std::this_thread::sleep_for(4ms);
                MPI_Status mpiStatus;
                auto mpiLg = std::lock_guard(mpiMutex);
                checkMpiErrors(MPI_Test(&mpiRequests.front(), &flag, &mpiStatus));
            }
            canProcessResponses_.store(State::REMOTE);

            mpiStatuses.resize(receiverCount);
            for(int flag = 0; !flag;) {
                std::this_thread::sleep_for(4ms);
                auto mpiLg = std::lock_guard(mpiMutex);
                checkMpiErrors(MPI_Testall(mpiRequests.size(), mpiRequests.data(), &flag, mpiStatuses.data()));
            }
        }

        if(dwBatchRequest->quit) {
            {
                mpiRequests.resize(0);
                auto mpiLg = std::lock_guard(mpiMutex);
                for(int64_t otherNode = 0; otherNode < getNumNodes(); ++otherNode) {
                    if(metaDataReceiverMask[otherNode] or otherNode == getNodeId()) continue;

                    auto &request = outgoingRequests[otherNode];
                    auto &buffer  = request.metaDataBuffer();
                    request.quit(true);
                    mpiRequests.emplace_back(MPI_Request{});
                    checkMpiErrors(MPI_Isend(
                        buffer.data(),
                        buffer.size(),
                        request.getMpiDataType(),
                        otherNode,
                        Id,
                        matrix_->mpiComm(),
                        &mpiRequests.back()
                    ));
                }
            }
            liveNodeCounter_.fetch_sub(1);
            mpiStatuses.resize(mpiRequests.size());
            for(int flag = 0; !flag;) {
                std::this_thread::sleep_for(4ms);
                auto mpiLg = std::lock_guard(mpiMutex);
                checkMpiErrors(MPI_Testall(mpiRequests.size(), mpiRequests.data(), &flag, mpiStatuses.data()));
            }
            receiverDaemon_.join();
            daemon_.join();
        }
    }

    [[nodiscard]] bool canTerminate() const override {
        return canTerminate_.load() and hh::AbstractTask<2, Matrix, DwRequest, Tile>::canTerminate();
    }

    [[nodiscard]] std::string extraPrintingInformation() const override {
        auto dotTimer = this->dotTimer_;
        auto suffix = "MB/s";

        double size = (matrix_->tileDim()*matrix_->tileDim()*sizeof(MatrixType))/(1024.*1024.);
        auto min = std::to_string(size/dotTimer.max());
        auto avg = std::to_string(size/dotTimer.avg());
        auto max = std::to_string(size/dotTimer.min());
        return "#Tiles received: " + std::to_string(dotTimer.count()) + "\\n"
            "BW:\\n"
            "Min: " + min.substr(0, min.find('.', 0)+4) + suffix + "\\n"
            "Avg: " + avg.substr(0, avg.find('.', 0)+4) + suffix + "\\n"
            "Max: " + max.substr(0, max.find('.', 0)+4) + suffix + "\\n"
            "Total time spent: " + std::to_string(dotTimer.totalTime()) + "s"
            ;
    }

private:
    void updateState() {
        canTerminate_.store(true
            and (canProcessResponses_.load() == State::FREE)
            and (liveNodeCounter_.load() == 0)
            and outgoingResponses_.empty()
        );
    }

    void processIncomingRequests() {
        int        found     = 0;
        MPI_Status mpiStatus = {};
        auto       mpiLg     = std::lock_guard(mpiMutex);
        checkMpiErrors(MPI_Iprobe(MPI_ANY_SOURCE, Id, matrix_->mpiComm(), &found, &mpiStatus));
        if(!found) return;

        InterNodeRequest interNodeRequest   {};
        MPI_Status       mpiReceiveStatus = {};
        auto             &buffer          = interNodeRequest.metaDataBuffer();
        checkMpiErrors(MPI_Recv(buffer.data(), buffer.size(), MPI_INT64_T, mpiStatus.MPI_SOURCE, Id, matrix_->mpiComm(), &mpiReceiveStatus));
        for(const auto [rowIdx, colIdx, tagId]: interNodeRequest) {
            auto tile = matrix_->tile(rowIdx, colIdx);
            outgoingResponses_.emplace_back(MPI_Request{});
            checkMpiErrors(MPI_Issend(
                tile->data(),
                tile->byteSize(),
                MPI_BYTE,
                mpiStatus.MPI_SOURCE,
                tagId,
                matrix_->mpiComm(),
                &outgoingResponses_.back()
            ));
        }
        if(interNodeRequest.quit()) liveNodeCounter_.fetch_sub(1);
    }

    void processOutgoingResponses() {
//        int32_t flag  = 0;
//        auto    mpiLg = std::lock_guard(mpiMutex);
//        checkMpiErrors(MPI_Testall(outgoingResponses_.size(), outgoingResponses_.data(), &flag, MPI_STATUS_IGNORE));
//        if(flag) outgoingResponses_.clear();

        auto mpiLg = std::lock_guard(mpiMutex);
        for(auto it = outgoingResponses_.begin(); it != outgoingResponses_.end(); ) {
            MPI_Request& mpiRequest = *it;
            int32_t flag;
            MPI_Status mpiStatus;
            if(checkMpiErrors(MPI_Test(&mpiRequest, &flag, &mpiStatus)); flag) {
                it = outgoingResponses_.erase(it);
            }
            else {
                it++;
            }
        }
    }

    void receiverDaemon() {
        using namespace std::chrono_literals;
        while(true) {
            while(canProcessResponses_.load() == State::STALL) {
                std::this_thread::sleep_for(4ms);
            }

            for(const auto [rowIdx, colIdx, tagId]: currentRequest_->data) {
                if(auto tile = matrix_->tile(rowIdx, colIdx); tile != nullptr) {
                    this->addResult(tile);
                    continue;
                }

                MPI_Status mpiStatus = {};
                auto       tile      = std::static_pointer_cast<Tile>(this->getManagedMemory());
                tile->init(rowIdx, colIdx, matrix_->tileHeight(rowIdx, colIdx), matrix_->tileWidth(rowIdx, colIdx));
                tile->memoryState(MemoryState::SHARED);

                do {
                    std::this_thread::sleep_for(4ms);
                } while(canProcessResponses_.load() != State::REMOTE);

                auto mpiLg = std::lock_guard(mpiMutex);
                dotTimer_.start();
                checkMpiErrors(MPI_Recv(
                    tile->data(),
                    (int)tile->byteSize(),
                    MPI_BYTE,
                    matrix_->owner(rowIdx, colIdx),
                    tagId,
                    matrix_->mpiComm(),
                    &mpiStatus
                ));
                dotTimer_.stop();
                this->addResult(tile);
            }

            canProcessResponses_.store(State::FREE);
            if(currentRequest_->quit) {
                currentRequest_ = nullptr;
                return;
            }
            currentRequest_ = nullptr;
        }
    }

    void daemon() {
        using namespace std::chrono_literals;
        while(!canTerminate()) {
            processIncomingRequests();
            processOutgoingResponses();
            updateState();
            std::this_thread::sleep_for(4ms);
        }
    }

private:
    std::atomic_bool           canTerminate_        = false;
    std::atomic_int64_t        liveNodeCounter_     = {};
    std::shared_ptr<Matrix>    matrix_              = nullptr;
    std::thread                daemon_              = {};
    std::thread                receiverDaemon_      = {};
    std::list<MPI_Request>     outgoingResponses_   = {};
    std::atomic_int32_t        canProcessResponses_ = false;
    std::shared_ptr<DwRequest> currentRequest_      = nullptr;
    DotTimer                   dotTimer_              {};
};

/**
 * A specialized MPI task.
 * It supports only 1 CommType type, which is the same for both input and output.
 * It relies on memory manager to get buffer for receiving data.
 * An MPI edge is formed based on the tuple (MPI_Comm, messageTag). This edge is similar to the switch rule observed in execution pipeline.
 *
 * FIXME: global variable dependency: mpiMutex
 *
 * @tparam CommType
 * @tparam OtherInputTypes
 */
template<class CommType, class ...OtherInputTypes>
class AbstractMpiCommTask: public hh::AbstractTask<sizeof...(OtherInputTypes)+1, CommType, OtherInputTypes..., CommType> {
private:
    class CommQueues {
    public:
        void lock() {
            mutex_.lock();
        }

        void unlock() {
            this->sendQueueSize.store(this->sendQueue.size());
            this->recvQueueSize.store(this->recvQueue.size());
            mutex_.unlock();
        }

        [[nodiscard]] bool try_lock() noexcept {
            return mutex_.try_lock();
        }

        void logBandwidth(const std::chrono::time_point<std::chrono::system_clock> &startTime, const std::chrono::time_point<std::chrono::system_clock> &endTime, const int32_t sizeInBytes) {
            const auto time      = double(std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count())/1.e9;
            const auto bandwidth = sizeInBytes/time;

            receiveCount_++;
            avgBandwidth_  = (avgBandwidth_*(receiveCount_-1) + bandwidth)/receiveCount_;
            minBandwidth_  = std::min(minBandwidth_, bandwidth);
            maxBandwidth_  = std::max(maxBandwidth_, bandwidth);
        }

        [[nodiscard]] double avgBw() const {
            return avgBandwidth_;
        }

        [[nodiscard]] double minBw() const {
            return minBandwidth_;
        }

        [[nodiscard]] double maxBw() const {
            return maxBandwidth_;
        }

        [[nodiscard]] int32_t receiveCount() const {
            return receiveCount_;
        }

    public:
        std::vector<std::tuple<std::shared_ptr<CommType>, MPI_Request, std::chrono::time_point<std::chrono::system_clock>>> recvQueue     = {};
        std::atomic_int32_t                                                                                                 recvQueueSize = -1;
        std::vector<std::tuple<std::shared_ptr<CommType>, MPI_Request>>                                                     sendQueue     = {};
        std::atomic_int32_t                                                                                                 sendQueueSize = -1;

    private:
        std::mutex mutex_         = {};
        double     minBandwidth_  = std::numeric_limits<double>::max();
        double     maxBandwidth_  = -1;
        double     avgBandwidth_  = 0;
        int32_t    receiveCount_  = 0;
    };

public:
    explicit AbstractMpiCommTask(const std::string &name, MPI_Comm mpiComm = MPI_COMM_WORLD, bool autoMaticStart = false):
            hh::AbstractTask<sizeof...(OtherInputTypes)+1, CommType, OtherInputTypes..., CommType>(name, 1, autoMaticStart), mpiComm_(mpiComm) {

        std::lock_guard mpiLg(mpiMutex);
        checkMpiErrors(MPI_Comm_rank(mpiComm, &mpiNodeId_));
        checkMpiErrors(MPI_Comm_size(mpiComm, &mpiNumNodes_));
    }

    virtual void execute([[maybe_unused]] std::shared_ptr<CommType> data) {}

    virtual void initializeComm() {}

    void initialize() final {
        consumerDaemon_ = std::thread(&AbstractMpiCommTask::consumerDaemon, this);
        receiverDaemon_ = std::thread(&AbstractMpiCommTask::receiverDaemon, this);

        initializeComm();
    }

    virtual void shutdownComm() {}

    void shutdown() final {
        shutdownComm();

        if(consumerDaemon_.joinable()) consumerDaemon_.join();
        if(receiverDaemon_.joinable()) receiverDaemon_.join();
    }

    // Getters
    [[nodiscard]] int32_t mpiNodeId()               const { return mpiNodeId_;                            }
    [[nodiscard]] int32_t mpiNumNodes()             const { return mpiNumNodes_;                          }
    [[nodiscard]] bool    isCommSendQueueEmpty()    const { return commQueues_.sendQueueSize.load() == 0; }
    [[nodiscard]] bool    isCommReceiveQueueEmpty() const { return commQueues_.recvQueueSize.load() == 0; }

    void sendToViaMpiAsync(std::shared_ptr<CommType> &data, const int32_t destinationNodeId, const int32_t tagId = 0) {
        assert(0 <= destinationNodeId);
        assert(0 <= tagId);

        std::lock_guard commLg(commQueues_);
        auto &sendQueue = commQueues_.sendQueue;
        sendQueue.emplace_back(data, MPI_Request{});
        auto &mpiRequest = std::get<1>(sendQueue.back());

        std::lock_guard mpiLg(mpiMutex);
        checkMpiErrors(MPI_Issend(getBuffer(*data), getBufferSizeInBytes(*data), MPI_BYTE, destinationNodeId, std::max(tagId, 0), mpiComm_, &mpiRequest));
    }

    virtual void postProcessMpiSentData([[maybe_unused]] std::shared_ptr<CommType> &data, [[maybe_unused]] int32_t destinationNodeId, [[maybe_unused]] int32_t tagId, [[maybe_unused]] int32_t sizeInBytes) {}

    void joinCommThreads() {
        consumerDaemon_.join();
        receiverDaemon_.join();
    }

    [[nodiscard]] virtual std::tuple<std::shared_ptr<CommType>, int32_t, int32_t, bool> mpiReceiveProtocol() {
        assert(this->memoryManager() != nullptr);
        return std::make_tuple(std::dynamic_pointer_cast<CommType>(this->getManagedMemory()), MPI_ANY_SOURCE, MPI_ANY_TAG, false);
    }

    // WARNING: has to be non-blocking
    [[nodiscard]] virtual std::tuple<std::shared_ptr<CommType>, int32_t, int32_t> mpiSendProtocol() {
        return std::make_tuple(nullptr, MPI_ANY_SOURCE, MPI_ANY_TAG);
    }

    virtual void preProcessMpiReceivedData([[maybe_unused]] std::shared_ptr<CommType> &data, [[maybe_unused]] int32_t sourceNodeId, [[maybe_unused]] int32_t tagId, [[maybe_unused]] int32_t sizeInBytes) {}

    [[nodiscard]] virtual bool canTerminateComm() const = 0;

    [[nodiscard]] bool canTerminate() const final {
        return canTerminateComm() and hh::AbstractTask<sizeof...(OtherInputTypes)+1, CommType, OtherInputTypes..., CommType>::canTerminate();
    }

    [[nodiscard]] std::string extraPrintingInformation() const override {
        //std::lock_guard lg(commQueues_);FIXME: needed?
        constexpr auto suffix = "MB/s";

        auto min = std::to_string(commQueues_.minBw()/(1024.*1024.));
        auto avg = std::to_string(commQueues_.avgBw()/(1024.*1024.));
        auto max = std::to_string(commQueues_.maxBw()/(1024.*1024.));

        return "#Elements received via MPI: " + std::to_string(commQueues_.receiveCount())
            + "\\nBW:"
            + "\\nMin: " + min.substr(0, min.find('.', 0)+4) + suffix
            + "\\nAvg: " + avg.substr(0, avg.find('.', 0)+4) + suffix
            + "\\nMax: " + max.substr(0, max.find('.', 0)+4) + suffix
        ;
    }

private:
    int32_t postMpiReceive() {
        auto [data, sourceNodeId, tagId, blocking] = this->mpiReceiveProtocol();
        if(data == nullptr) return 0;

        if(blocking) {
            MPI_Status mpiStatus;
            const auto bufferSizeInBytes = getBufferSizeInBytes(*data);

            std::chrono::time_point<std::chrono::system_clock> start, end;
            if(true) {
                std::lock_guard mpiLg(mpiMutex);
                start = std::chrono::system_clock::now();
                checkMpiErrors(MPI_Recv(getBuffer(*data), bufferSizeInBytes, MPI_BYTE, sourceNodeId, tagId, mpiComm_, &mpiStatus));
                end  = std::chrono::system_clock::now();
            }

            preProcessMpiReceivedData(data, sourceNodeId, tagId, bufferSizeInBytes);
            this->execute(data);

            std::lock_guard lg(commQueues_);
            commQueues_.logBandwidth(start, end, bufferSizeInBytes);

            return 1;
        }

        std::lock_guard commLg(commQueues_);
        auto &recvQueue = commQueues_.recvQueue;

        std::lock_guard mpiLg(mpiMutex);
        recvQueue.emplace_back(data, MPI_Request{}, std::chrono::system_clock::now());
        auto &mpiRequest = std::get<1>(recvQueue.back());
        checkMpiErrors(MPI_Irecv(getBuffer(*data), getBufferSizeInBytes(*data), MPI_BYTE, sourceNodeId, tagId, mpiComm_, &mpiRequest));

        return recvQueue.size();
    }

    int32_t processRecvQueue(bool flush = false) {
        std::lock_guard commLg(commQueues_);
        auto &recvQueue = commQueues_.recvQueue;

        for(auto it = recvQueue.begin(); it != recvQueue.end();) {
            auto &[data, mpiRequest, start] = *it;
            int32_t flag = false;
            MPI_Status mpiStatus;
            std::lock_guard mpiLg(mpiMutex);
            checkMpiErrors(MPI_Test(&mpiRequest, &flag, &mpiStatus));
            if(flag) {
                auto end = std::chrono::system_clock::now();
                int32_t bufferSizeInBytes;
                checkMpiErrors(MPI_Get_count(&mpiStatus, MPI_CHAR, &bufferSizeInBytes));
                preProcessMpiReceivedData(data, mpiStatus.MPI_SOURCE, mpiStatus.MPI_TAG, bufferSizeInBytes);
                this->execute(data);
                it = recvQueue.erase(it);
                commQueues_.logBandwidth(start, end, bufferSizeInBytes);
            }
            else {
                it++;
            }
        }

        if(flush) {
            std::vector<MPI_Request> enqueuedMpiReceiveRequests;
            enqueuedMpiReceiveRequests.reserve(recvQueue.size()+8);
            for(auto &[data, mpiRequest, start]: recvQueue) {
                std::lock_guard mpiLg(mpiMutex);
                checkMpiErrors(MPI_Cancel(&mpiRequest));
                enqueuedMpiReceiveRequests.emplace_back(mpiRequest);
            }

            int processed = false;
            do {
                std::this_thread::sleep_for(sleepTime_);
                std::lock_guard mpiLg(mpiMutex);
                checkMpiErrors(MPI_Testall(enqueuedMpiReceiveRequests.size(), enqueuedMpiReceiveRequests.data(), &processed, MPI_STATUSES_IGNORE));
            } while(!processed);

            recvQueue.clear();
        }

        return recvQueue.size();
    }

    int32_t postMpiSend() {
        auto [data, destinationNodeId, tagId] = this->mpiSendProtocol();
        if(data == nullptr) return 0;

        this->sendToViaMpiAsync(data, destinationNodeId, tagId);
        return 1;
    }

    int32_t processSendQueue(bool flush = false) {
        std::lock_guard commLg(commQueues_);
        auto &sendQueue = commQueues_.sendQueue;

        do {
            for(auto it = sendQueue.begin(); it != sendQueue.end();) {
                auto &[data, mpiRequest] = *it;
                int flag = false;

                std::lock_guard mpiLg(mpiMutex);
                MPI_Status mpiStatus;
                checkMpiErrors(MPI_Test(&mpiRequest, &flag, &mpiStatus));
                if(flag) {
                    postProcessMpiSentData(data, mpiStatus.MPI_SOURCE, mpiStatus.MPI_TAG, getBufferSizeInBytes(*data));
                    it = sendQueue.erase(it);
                }
                else {
                    it++;
                }
            }
        } while(flush and !sendQueue.empty());

        return sendQueue.size();
    }

    void consumerDaemon() {
        while(!canTerminate()) {
            for(int32_t i = 0; 0 < postMpiSend() and i < 32; ++i);
            processRecvQueue();
            processSendQueue();

            std::this_thread::sleep_for(sleepTime_);
        }

        // process recv and send queues if any
        processSendQueue(true);
        processRecvQueue(true);
    }

    void receiverDaemon() {
        postMpiReceive();
        while(!canTerminate()) {
            if(postMpiReceive() == 0 and processRecvQueue() == 0) {
                std::this_thread::sleep_for(sleepTime_);
            }
            processSendQueue();
        }
    }

protected:

private:
    CommQueues                commQueues_       {};
    std::thread               consumerDaemon_ = {};
    MPI_Comm                  mpiComm_        = MPI_COMM_WORLD;
    int32_t                   mpiNodeId_      = 0;
    int32_t                   mpiNumNodes_    = 0;
    std::thread               receiverDaemon_ = {};
    std::chrono::milliseconds sleepTime_      = 4ms;
};

#endif //HH3_MATMUL_COMMON_TASKS_H

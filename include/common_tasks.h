#ifndef HH3_MATMUL_COMMON_TASKS_H
#define HH3_MATMUL_COMMON_TASKS_H

#include <array>

#include "common_data.h"

template<typename MatrixType, char Id>
class MatrixWarehouseTask: public hh::AbstractTask<2, MatrixContainer<MatrixType, Id>, DbRequest<Id>, MatrixTile<MatrixType, Id>> {
private:
    using Matrix     = MatrixContainer<MatrixType, Id>;
    using DB_Request = DbRequest<Id>;
    using Tile       = MatrixTile<MatrixType, Id>;

    class InterNodeRequest {
    private:
        using MetaDataBuffer = std::array<int64_t, 2048>;
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
        explicit InterNodeRequest() = default;

        explicit InterNodeRequest(const int64_t otherNode, const std::vector<std::tuple<int64_t, int64_t>> &&indices, bool quit = false):
            otherNode_(otherNode) {
            assert((indices.size()*STRIDE + BEGIN) < (sizeof(MetaDataBuffer)/sizeof(int64_t)));

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
    explicit MatrixWarehouseTask(): hh::AbstractTask<2, Matrix, DB_Request, Tile>("Matrix DB Task", 1, false) {}

    ~MatrixWarehouseTask() override = default;

    void execute(std::shared_ptr<Matrix> matrix) override {
        assert(matrix_ == nullptr);

        matrix_ = matrix;
        daemon_ = std::thread(&MatrixWarehouseTask::daemon, this);
        liveNodeCounter_.store(matrix_->numNodes());

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
        return hh::AbstractTask<2, Matrix, DB_Request, Tile>::canTerminate() and canTerminate_;
    }

private:
    void handleDbRequest(std::shared_ptr<DB_Request> dbRequest) {
        /** send meta data asynchronously and enqueue the future **/{
            auto request   = InterNodeRequest(dbRequest->srcNode, std::move(dbRequest->indices), dbRequest->quit);
            auto &mdBuffer = request.metaDataBuffer();
            if(dbRequest->srcNode != matrix_->nodeId()) {
                // metadata is pretty small (16KB) therefore eager protocol will be used by MPI while sending this
                // buffer, hence a blocking send is good enough here.
                auto lg = std::lock_guard(mpiMutex);
                checkMpiErrors(MPI_Isend(&mdBuffer[0], sizeof(mdBuffer), MPI_BYTE, request.otherNode(), Id, matrix_->mpiComm(), request.mpiRequestHandle()));
            }
            auto lg = std::lock_guard(mutex_);
            outgoingRequests_.emplace_back(std::move(request));
        }

        if(!dbRequest->quit) return;

        /** send quit requests to the other nodes **/{
            auto sl = std::scoped_lock(mutex_, mpiMutex);
            for(int64_t node = 0; node < matrix_->numNodes(); ++node) {
                if(node == matrix_->nodeId() or node == dbRequest->srcNode) continue;

                auto request   = InterNodeRequest();
                auto &mdBuffer = request.metaDataBuffer();
                request.quit(node);
                checkMpiErrors(MPI_Isend(&mdBuffer[0], sizeof(mdBuffer), MPI_BYTE, request.otherNode(), Id, matrix_->mpiComm(), request.mpiRequestHandle()));
                outgoingRequests_.emplace_back(std::move(request));
            }
            liveNodeCounter_.fetch_sub(1);
        }
        daemon_.join();
    }

    void processOutgoingRequests() {
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

            int32_t done;
            MPI_Status mpiStatus;
            mpiMutex.lock();
            checkMpiErrors(MPI_Test(request.mpiRequestHandle(), &done, &mpiStatus));
            mpiMutex.unlock();

            if(done) {
                incomingResponse_.pInterNodeRequest = &request;
                incomingResponse_.currentIndex      = 0;
            }
        }
    }

    void processCurrentIncomingResponse() {
        if(incomingResponse_.pInterNodeRequest == nullptr or this->memoryManager()->currentSize() == 0) return;

        MPI_Status mpiStatus = {};
        auto       &request  = *incomingResponse_.pInterNodeRequest;
        for(auto &index = incomingResponse_.currentIndex; this->memoryManager()->currentSize() != 0 and index < request.batchSize(); index++) {
            auto tile = std::static_pointer_cast<Tile>(this->getManagedMemory());
            auto [rowIdx, colIdx, tagId] = request[index];
            tile->init(rowIdx, colIdx, matrix_->tileHeight(rowIdx, colIdx), matrix_->tileWidth(rowIdx, colIdx));
            tile->memoryState(MemoryState::ON_HOLD);

            std::lock_guard mpiLg(mpiMutex);
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
        }

        if(incomingResponse_.currentIndex == request.batchSize()) {
            auto lg = std::lock_guard(mutex_);
            outgoingRequests_.pop_front();
            incomingResponse_.pInterNodeRequest = nullptr;
            incomingResponse_.currentIndex      = -1;
        }
    }

    void processIncomingRequests() {
        int32_t          flag           = false;
        InterNodeRequest request          {};
        auto&            metaDataBuffer = request.metaDataBuffer();
        MPI_Status       mpiStatus      = {};

        mpiMutex.lock();
        checkMpiErrors(MPI_Iprobe(MPI_ANY_SOURCE, Id, matrix_->mpiComm(), &flag, &mpiStatus));
        mpiMutex.unlock();

        while(flag) {
            // this can be a long loop, we shouldn't hog the mpi mutex
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(4ms);
            auto lg = std::lock_guard(mpiMutex);
            MPI_Status mpiRecvStatus;
            checkMpiErrors(MPI_Recv(&metaDataBuffer[0], sizeof(metaDataBuffer), MPI_BYTE, mpiStatus.MPI_SOURCE, Id, matrix_->mpiComm(), &mpiRecvStatus));

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
            checkMpiErrors(MPI_Iprobe(MPI_ANY_SOURCE, Id, matrix_->mpiComm(), &flag, &mpiStatus));
        }
    }

    void processOutgoingResponses() {
        auto lg = std::lock_guard(mpiMutex);
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
        auto lg = std::lock_guard(mutex_);
        canTerminate_ = dbRequests_.empty()
                and outgoingRequests_.empty()
                and outgoingResponses_.empty()
                and (liveNodeCounter_.load() == 0)
                and (incomingResponse_.pInterNodeRequest == nullptr);
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
    bool                                   canTerminate_       = false;
    std::shared_ptr<Matrix>                matrix_             = nullptr;
    std::thread                            daemon_             = {};
    std::mutex                             mutex_              = {};
    std::list<std::shared_ptr<DB_Request>> dbRequests_         = {};// Need to use mutex_
    std::list<InterNodeRequest>            outgoingRequests_   = {};// Need to use mutex_
    std::list<InterNodeResponse>           outgoingResponses_  = {};
    std::atomic_int64_t                    liveNodeCounter_    = {};
    IncomingResponse                       incomingResponse_   = {};
};


#endif //HH3_MATMUL_COMMON_TASKS_H

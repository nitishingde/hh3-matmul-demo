#ifndef HH3_MATMUL_COMMON_TASKS_H
#define HH3_MATMUL_COMMON_TASKS_H

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

        Iterator begin() { return Iterator(&metaDataBuffer_[BEGIN]);                     }
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

public:
    explicit MatrixWarehouseTask(): hh::AbstractTask<2, Matrix, DB_Request, Tile>("Matrix DB Task", 1, false) {}

    ~MatrixWarehouseTask() override = default;

    void execute(std::shared_ptr<Matrix> matrix) override {
        assert(matrix_ == nullptr);

        matrix_       = matrix;
        daemon_       = std::thread(&MatrixWarehouseTask::daemon, this);
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
        return hh::AbstractTask<2, Matrix, DB_Request, Tile>::canTerminate() and liveNodeCounter_.load() == 0 and outgoingResponses_.empty() and outgoingRequests_.empty();
    }

private:
    void handleDbRequest(std::shared_ptr<DB_Request> dbRequest) {
        if(dbRequest->srcNode == matrix_->nodeId()) {
            for(auto [rowIdx, colIdx]: dbRequest->indices) {
                this->addResult(matrix_->tile(rowIdx, colIdx));
            }
        }
        else {
            // metadata is pretty small (16KB) therefore eager protocol will be used by MPI while sending this
            // buffer, hence a blocking send is good enough here.
            auto request   = InterNodeRequest(dbRequest->srcNode, std::move(dbRequest->indices), dbRequest->quit);
            auto &mdBuffer = request.metaDataBuffer();

            mpiMutex.lock();
            checkMpiErrors(MPI_Isend(&mdBuffer[0], sizeof(mdBuffer), MPI_BYTE, request.otherNode(), Id, matrix_->mpiComm(), request.mpiRequestHandle()));
            mpiMutex.unlock();

            mutex_.lock();
            outgoingRequests_.emplace_back(std::move(request));
            mutex_.unlock();
        }

        if(!dbRequest->quit) return;

        {
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
        auto lg = std::lock_guard(mutex_);
        mpiMutex.lock();
        for(auto it = outgoingRequests_.begin(); it != outgoingRequests_.end();) {
            auto &request = *it;
            int32_t done;
            MPI_Status mpiStatus;
            checkMpiErrors(MPI_Test(request.mpiRequestHandle(), &done, &mpiStatus));
            if(done) {
                mpiMutex.unlock();
                for(auto [rowIdx, colIdx, tagId]: request) {
                    auto tile = std::static_pointer_cast<Tile>(this->getManagedMemory());
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
                it = outgoingRequests_.erase(it);
                mpiMutex.lock();
            }
            else {
                it++;
            }
        }
        mpiMutex.unlock();
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
            MPI_Status mpiRecvStatus;
            mpiMutex.lock();
            checkMpiErrors(MPI_Recv(&metaDataBuffer[0], sizeof(metaDataBuffer), MPI_BYTE, mpiStatus.MPI_SOURCE, Id, matrix_->mpiComm(), &mpiRecvStatus));
            mpiMutex.unlock();

            auto sl = std::scoped_lock(mutex_, mpiMutex);
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
        std::scoped_lock sl(mutex_, mpiMutex);
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

    void daemon() {
        using namespace std::chrono_literals;
        while(!canTerminate()) {
            processIncomingRequests();
            processOutgoingResponses();
            processOutgoingRequests();
            std::this_thread::sleep_for(4ms);
        }
    }

private:
    std::shared_ptr<Matrix>                matrix_             = nullptr;
    std::thread                            daemon_             = {};
    std::mutex                             mutex_              = {};
    std::list<std::shared_ptr<DB_Request>> dbRequests_         = {};
    std::list<InterNodeRequest>            outgoingRequests_   = {};
    std::list<InterNodeResponse>           outgoingResponses_  = {};
    std::atomic_int64_t                    liveNodeCounter_    = {};
    std::shared_ptr<uint8_t[]>             mpiBuffer_          = {};
};


#endif //HH3_MATMUL_COMMON_TASKS_H

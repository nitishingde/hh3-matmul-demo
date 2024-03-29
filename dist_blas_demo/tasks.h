#ifndef HH3_MATMUL_TASKS
#define HH3_MATMUL_TASKS

#include <atomic>
#include <bitset>
#include <hedgehog/hedgehog.h>
#include <list>
#include <cblas.h>
#include "data.h"

//#define LOG() printf("[%d] %s:%d\n", getNodeId(), __FILE__, __LINE__)

template<typename MatrixType, char Id>
class MatrixDbTask: public hh::AbstractTask<2, MatrixContainer<MatrixType, Id>, DbRequest<Id>, MatrixTile<MatrixType, Id>> {
private:
    using Matrix     = MatrixContainer<MatrixType, Id>;
    using DB_Request = DbRequest<Id>;
    using Tile       = MatrixTile<MatrixType, Id>;

    class InterNodeRequest {
    private:
        using MetaDataBuffer = std::array<int64_t, 4>;
    public:
        explicit InterNodeRequest() = default;

        explicit InterNodeRequest(const std::shared_ptr<Tile> data, const int64_t otherNode, const int64_t tagId, const bool quit = false) {
            assert((data == nullptr and quit == true) or (data != nullptr and quit == false));
            metaDataBuffer_[Offset::ROW] = data? data->rowIdx(): -1;
            metaDataBuffer_[Offset::COL] = data? data->colIdx(): -1;
            metaDataBuffer_[Offset::TAG] = tagId;
            metaDataBuffer_[Offset::FIN] = quit;
            otherNode_ = otherNode;
            data_ = data;
        }

        // Getters
        [[nodiscard]] int64_t                 tagId()          const { return metaDataBuffer_[Offset::TAG]; }
        [[nodiscard]] int64_t                 rowIdx()         const { return metaDataBuffer_[Offset::ROW]; }
        [[nodiscard]] int64_t                 colIdx()         const { return metaDataBuffer_[Offset::COL]; }
        [[nodiscard]] bool                    quit()           const { return metaDataBuffer_[Offset::FIN]; }
        [[nodiscard]] void*                   dataBuffer()           { return data_->data();                }
        [[nodiscard]] int64_t                 dataByteSize()   const { return data_->byteSize();            }
        [[nodiscard]] std::array<int64_t, 4>& metaDataBuffer()       { return metaDataBuffer_;              }
        [[nodiscard]] int64_t                 otherNode()      const { return otherNode_;                   }
        [[nodiscard]] std::shared_ptr<Tile>   data()                 { return data_;                        }

    private:
        enum Offset {
            ROW = 0,
            COL = 1,
            TAG = 2,
            FIN = 3,
        };
        MetaDataBuffer        metaDataBuffer_ = {0};
        int64_t               otherNode_      = -1;
        std::shared_ptr<Tile> data_           = nullptr;
    };

    struct InterNodeResponse {
        MPI_Request           mpiRequest = {};
        std::shared_ptr<Tile> pData      = nullptr;
    };

public:
    explicit MatrixDbTask(): hh::AbstractTask<2, Matrix, DB_Request, Tile>("Matrix DB Task", 1, false) {}

    ~MatrixDbTask() override = default;

    void execute(std::shared_ptr<Matrix> matrix) override {
        assert(matrix_ == nullptr);

        matrix_ = matrix;
        liveNodeCounter_.store(matrix_->numNodes());
        liveNodeList_ = std::vector<bool>(matrix_->numNodes(), true);
        daemon_ = std::thread(&MatrixDbTask::daemon, this);

        for(; !dbRequests_.empty(); dbRequests_.pop_front()) {
            handleDbRequest(dbRequests_.front());
        }
    }

    void execute(std::shared_ptr<DB_Request> dbRequest) override {
        if(matrix_ == nullptr) {
            dbRequests_.emplace_back(dbRequest);
            return;
        }

        if(!this->liveNodeList_[matrix_->nodeId()]) {
            throw std::runtime_error("MatrixDbTask shouldn't receive db requests!");
        }

        handleDbRequest(dbRequest);
    }

    [[nodiscard]] bool canTerminate() const override {
        return isStarted_ and liveNodeCounter_ == 0 and outgoingRequests_.empty() and incomingResponses_.empty() and outgoingResponses_.empty();
    }

private:
    void handleDbRequest(std::shared_ptr<DB_Request> dbRequest) {
        if(dbRequest->quit and liveNodeList_[matrix_->nodeId()]) {
            liveNodeList_[matrix_->nodeId()] = false;
            liveNodeCounter_.store(std::accumulate(liveNodeList_.begin(), liveNodeList_.end(), 0));
            for(int64_t node = 0; node < matrix_->numNodes(); ++node) {
                if(node == matrix_->nodeId()) continue;
                outgoingRequests_.emplace_back(InterNodeRequest(nullptr, node, -1, true));
            }
            daemon_.join();
            return;
        }

        if(auto tile = matrix_->tile(dbRequest->rowIdx, dbRequest->colIdx); tile != nullptr) {
            this->addResult(tile);
            return;
        }

        // wait
        auto matrixTile = std::static_pointer_cast<MatrixTile<MatrixType, Id>>(this->getManagedMemory());
        const auto rowIdx = dbRequest->rowIdx, colIdx = dbRequest->colIdx;
        matrixTile->init(rowIdx, colIdx, this->matrix_->tileHeight(rowIdx, colIdx), this->matrix_->tileWidth(rowIdx, colIdx));

        std::lock_guard lc(mutex_);
        outgoingRequests_.emplace_back(InterNodeRequest(
            matrixTile,
            matrix_->owner(rowIdx, colIdx),
            tagGenerator()
        ));
    }

    void processOutgoingRequests() {
        std::lock_guard lc(mutex_);
        for(auto &request: outgoingRequests_) {
            if(request.otherNode() == matrix_->nodeId()) continue;

            // metadata is pretty small, 4*8B = 32B, therefore eager protocol will be used by MPI while sending this
            // buffer, hence a blocking send is good enough here.
            auto mdBuffer = request.metaDataBuffer();
            checkMpiErrors(MPI_Send(&mdBuffer[0], sizeof(mdBuffer), MPI_BYTE, request.otherNode(), Id, matrix_->mpiComm()));
            if(request.quit()) continue;

            auto&& response = InterNodeResponse();
            response.pData  = request.data();
            checkMpiErrors(MPI_Irecv(
                request.dataBuffer(),
                (int)request.dataByteSize(),
                MPI_BYTE,
                request.otherNode(),
                request.tagId(),
                matrix_->mpiComm(),
                &response.mpiRequest
            ));
            incomingResponses_.emplace_back(response);
        }
        outgoingRequests_.resize(0);
    }

    void processIncomingRequests() {
        auto             lockGuard      = std::lock_guard(mutex_);
        int32_t          flag           = false;
        InterNodeRequest request          {};
        auto&            metaDataBuffer = request.metaDataBuffer();
        MPI_Status       mpiStatus      = {};
        for(checkMpiErrors(MPI_Iprobe(MPI_ANY_SOURCE, Id, matrix_->mpiComm(), &flag, &mpiStatus)); flag; checkMpiErrors(MPI_Iprobe(MPI_ANY_SOURCE, Id, matrix_->mpiComm(), &flag, &mpiStatus))) {
            MPI_Status mpiRecvStatus;
            checkMpiErrors(MPI_Recv(&metaDataBuffer[0], sizeof(metaDataBuffer), MPI_BYTE, mpiStatus.MPI_SOURCE, Id, matrix_->mpiComm(), &mpiRecvStatus));
            if(request.quit()) {
                liveNodeList_[mpiStatus.MPI_SOURCE] = false;
                liveNodeCounter_.store(std::accumulate(liveNodeList_.begin(), liveNodeList_.end(), 0));
                continue;
            }
            InterNodeResponse response = {};
            auto tile = matrix_->tile(request.rowIdx(), request.colIdx());
            response.pData = tile;
            checkMpiErrors(MPI_Isend(tile->data(), tile->byteSize(), MPI_BYTE, mpiStatus.MPI_SOURCE, request.tagId(), matrix_->mpiComm(), &response.mpiRequest));
            outgoingResponses_.emplace_back(response);
        }
    }

    void processIncomingResponses() {
        std::lock_guard lc(mutex_);
        for(auto it = incomingResponses_.begin(); it != incomingResponses_.end(); ) {
            auto &response = *it;
            int32_t flag = 0;
            MPI_Status mpiStatus;
            if(checkMpiErrors(MPI_Test(&response.mpiRequest, &flag, &mpiStatus)); flag) {
                response.pData->memoryState(MemoryState::SHARED);
                this->addResult(response.pData);
                it = incomingResponses_.erase(it);
            }
            else {
                it++;
            }
        }
    }

    void processOutgoingResponses() {
        std::lock_guard lc(mutex_);
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
        isStarted_ = true;
        while(!canTerminate()) {
            processOutgoingRequests();
            processIncomingRequests();
            processIncomingResponses();
            processOutgoingResponses();
            std::this_thread::sleep_for(4ms);
        }
    }

private:
    std::shared_ptr<Matrix>                matrix_             = nullptr;
    std::thread                            daemon_             = {};
    bool                                   isStarted_          = false;
    std::mutex                             mutex_              = {};
    std::list<std::shared_ptr<DB_Request>> dbRequests_         = {};
    std::list<InterNodeRequest>            outgoingRequests_   = {};
    std::list<InterNodeResponse>           incomingResponses_  = {};
    std::list<InterNodeResponse>           outgoingResponses_  = {};
    std::vector<bool>                      liveNodeList_       = {};
    std::atomic_int64_t                    liveNodeCounter_    = {};
};

template<typename MatrixType, char IdA, char IdB, char IdC>
class ProductTask: public hh::AbstractTask<
        1,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>, std::shared_ptr<MatrixTile<MatrixType, IdC>>>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>, std::shared_ptr<MatrixTile<MatrixType, IdC>>>
    >{
private:
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using TileC   = MatrixTile<MatrixType, IdC>;
    using Triplet = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>, std::shared_ptr<TileC>>;

public:
    explicit ProductTask(const int numThreads): hh::AbstractTask<1, Triplet , Triplet>("Product Task", numThreads, false) {}

    void execute(std::shared_ptr<Triplet> triplet) override {
        auto tileA = std::get<std::shared_ptr<TileA>>(*triplet);
        auto tileB = std::get<std::shared_ptr<TileB>>(*triplet);
        auto tileC = std::get<std::shared_ptr<TileC>>(*triplet);
        auto major = tileC->major() == Major::ROW? CBLAS_ORDER::CblasRowMajor: CBLAS_ORDER::CblasColMajor;

        const MatrixType alpha = 1., beta = 1.;

        if constexpr(std::is_same_v<MatrixType, float>) {
            cblas_sgemm(
                major, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
                tileA->height(), tileB->width(), tileA->width(), alpha,
                (float *) tileA->data(), tileA->leadingDimension(),
                (float *) tileB->data(), tileB->leadingDimension(), beta,
                (float *) tileC->data(), tileC->leadingDimension()
            );
        }
        else if constexpr(std::is_same_v<MatrixType, double>) {
            cblas_dgemm(
                major, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
                tileA->height(), tileB->width(), tileA->width(), alpha,
                (double *) tileA->data(), tileA->leadingDimension(),
                (double *) tileB->data(), tileB->leadingDimension(), beta,
                (double *) tileC->data(), tileC->leadingDimension()
            );
        }
        else {
            //TODO: add complex number support
            throw std::runtime_error("Datatype not supported for cuda product task.");
        }

        this->addResult(triplet);
    }

    std::shared_ptr<hh::AbstractTask<1, Triplet, Triplet>>
    copy() override {
        return std::make_shared<ProductTask>(this->numberThreads());
    }
};

#endif //HH3_MATMUL_TASKS

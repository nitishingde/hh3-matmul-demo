#ifndef HH3_MATMUL_TASKS
#define HH3_MATMUL_TASKS

#include "data.h"

template<typename MatrixType, char IdA, char IdB, char IdC>
class GpuJobGeneratorTask: public hh::AbstractTask<
        3,
        std::tuple<std::shared_ptr<MatrixContainer<MatrixType, IdA>>, std::shared_ptr<MatrixContainer<MatrixType, IdB>>, std::shared_ptr<MatrixContainer<MatrixType, IdC>>>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        DbRequest<IdA>,
        DbRequest<IdB>,
        GpuJob<MatrixType, IdA, IdB, IdC>
> {
private:
    template<class GridT>
    using Grid    = std::vector<std::vector<GridT>>;
    using MatrixA = MatrixContainer<MatrixType, IdA>;
    using MatrixB = MatrixContainer<MatrixType, IdB>;
    using MatrixC = MatrixContainer<MatrixType, IdC>;
    using Triplet = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>;
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using Job     = GpuJob<MatrixType, IdA, IdB, IdC>;

public:
    explicit GpuJobGeneratorTask(const int64_t devMemSizeInBytes, const int64_t prodTilesPerDev):
        hh::AbstractTask<3, Triplet, TileA, TileB, DbRequest<IdA>, DbRequest<IdB>, Job>("GpuJobGeneratorTask", 1, false),
        devMemSizeInBytes_(devMemSizeInBytes), prodTilesPerDev_(prodTilesPerDev) {}

    void execute(std::shared_ptr<Triplet> triplet) override {
        auto matrixA = std::get<std::shared_ptr<MatrixA>>(*triplet);
        auto matrixB = std::get<std::shared_ptr<MatrixB>>(*triplet);
        auto matrixC = std::get<std::shared_ptr<MatrixC>>(*triplet);

        std::set<int64_t> rowIndices = {};
        std::set<int64_t> colIndices = {};
        for(int64_t rowIdx = 0; rowIdx < matrixC->matrixNumRowTiles(); ++rowIdx) {
            for(int64_t colIdx = 0; colIdx < matrixC->matrixNumColTiles(); ++colIdx) {
                if(matrixC->tile(rowIdx, colIdx)) {
                    rowIndices.emplace(rowIdx);
                    colIndices.emplace(colIdx);
                }
            }
        }
        totalColTilesA_ = rowIndices.size();
        totalRowTilesB_ = colIndices.size();

        int64_t tileSize = std::max(std::max(matrixA->tileDim(), matrixB->tileDim()), matrixC->tileDim());
        int64_t tilesPerDev = devMemSizeInBytes_/(tileSize*tileSize*sizeof(MatrixType));
        // wh + ww + tilesPerDev = tilesPerDev, let wh = ww
        windowHeight_ = windowWidth_ = (tilesPerDev-prodTilesPerDev_)/2;

        struct JobK {
            int64_t index    = 0;
            int64_t priority = 0;
            bool operator()(const JobK &a, const JobK &b) {
                return a.priority < b.priority;
            }
        };
        std::priority_queue<JobK, std::vector<JobK>, JobK> priorityQueue;
        auto KT = matrixA->matrixNumColTiles();
        ttl_ = KT;

        // prioritize jobs on the basis of locality of data
        for(int64_t k = 0; k < KT; ++k) {
            auto jobK = JobK{
                    .index    = k,
                    .priority = 0,
            };

            // evaluate priority
            for(auto rowIdx: rowIndices) {
                if(matrixA->tile(rowIdx, k) != nullptr) {
                    jobK.priority++;
                }
            }
            for(auto colIdx: colIndices) {
                if(matrixB->tile(k, colIdx) != nullptr) {
                    jobK.priority++;
                }
            }

            priorityQueue.emplace(jobK);
        }

        colTilesFromA_.resize(KT);
        rowTilesFromB_.resize(KT);
        for(; !priorityQueue.empty(); priorityQueue.pop()) {
            auto jobK = priorityQueue.top();
            auto k = jobK.index;
            for(auto rowIdx: rowIndices) {
                if(auto tileA = matrixA->tile(rowIdx, k); tileA != nullptr) {
                    colTilesFromA_[k].emplace_back(tileA);
                }
                else {
                    this->addResult(std::make_shared<DbRequest<IdA>>(rowIdx, k));
                }
            }
            for(auto colIdx: colIndices) {
                if(auto tileB = matrixB->tile(k, colIdx); tileB != nullptr) {
                    rowTilesFromB_[k].emplace_back(tileB);
                }
                else {
                    this->addResult(std::make_shared<DbRequest<IdB>>(k, colIdx));
                }
            }
        }

        this->addResult(std::make_shared<DbRequest<IdA>>(0, 0, true));
        this->addResult(std::make_shared<DbRequest<IdB>>(0, 0, true));
        for(int64_t kt = 0; kt < KT; ++kt) {
            genJobsIfReady(kt);
        }
    }

    void execute(std::shared_ptr<TileA> tileA) override {
        auto k = tileA->colIdx();
        colTilesFromA_[k].emplace_back(tileA);
        genJobsIfReady(k);
    }

    void execute(std::shared_ptr<TileB> tileB) override {
        auto k = tileB->rowIdx();
        rowTilesFromB_[k].emplace_back(tileB);
        genJobsIfReady(k);
    }

    [[nodiscard]] bool canTerminate() const override {
        return ttl_ == 0;
    }

private:
    void genJobsIfReady(int64_t k) {
        if(colTilesFromA_[k].size() != totalColTilesA_ or rowTilesFromB_[k].size() != totalRowTilesB_) return;

        auto& vecA = colTilesFromA_[k];
        auto& vecB = rowTilesFromB_[k];

        for(size_t i = 0; i < vecB.size(); i+=windowWidth_) {
            for(size_t j = 0; j < vecA.size(); j+=windowHeight_) {
                auto job = std::make_shared<Job>();
                for(size_t ii = i; (ii < i+windowWidth_) and (ii < vecB.size()); ++ii) {
                    job->addTileA(vecA[ii]);
                }
                for(size_t jj = j; (jj < j+windowHeight_) and (jj < vecA.size()); ++jj) {
                    job->addTileB(vecB[jj]);
                }

                auto gpuToken = std::static_pointer_cast<GpuToken>(this->getManagedMemory());
                job->token(gpuToken);
                this->addResult(job);
            }
        }

        vecA.resize(0);
        vecB.resize(0);
        ttl_--;
        if(ttl_ == 0) {
            this->addResult(std::make_shared<Job>(true));
        }
    }

private:
    int64_t                      devMemSizeInBytes_ = -1;
    int64_t                      prodTilesPerDev_   = -1;
    int64_t                      windowHeight_      = -1;
    int64_t                      windowWidth_       = -1;
    size_t                       totalColTilesA_    = -1;
    size_t                       totalRowTilesB_    = -1;
    int64_t                      ttl_               = -1;
    Grid<std::shared_ptr<TileA>> colTilesFromA_     = {};
    Grid<std::shared_ptr<TileB>> rowTilesFromB_     = {};
};

template<typename MatrixType, char Id>
class MatrixWarehouseTask: public hh::AbstractTask<2, MatrixContainer<MatrixType, Id>, DbRequest<Id>, MatrixTile<MatrixType, Id>> {
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
    explicit MatrixWarehouseTask(): hh::AbstractTask<2, Matrix, DB_Request, Tile>("Matrix DB Task", 1, false) {}

    ~MatrixWarehouseTask() override = default;

    void execute(std::shared_ptr<Matrix> matrix) override {
        assert(matrix_ == nullptr);

        matrix_ = matrix;
        liveNodeCounter_.store(matrix_->numNodes());
        liveNodeList_ = std::vector<bool>(matrix_->numNodes(), true);
        daemon_ = std::thread(&MatrixWarehouseTask::daemon, this);

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

template<typename MatrixType, char IdA, char IdB, char IdC, char IdP>
class GpuJobSchedulerTask: public hh::AbstractTask<
        4,
        GpuJob<MatrixType, IdA, IdB, IdC>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        MatrixTile<MatrixType, IdP>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>
    > {
private:
    using Job     = GpuJob<MatrixType, IdA, IdB, IdC>;
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using TileP   = MatrixTile<MatrixType, IdP>;
    using Pair    = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>>;

public:
    explicit GpuJobSchedulerTask():
        hh::AbstractTask<4, Job, TileA, TileB, TileP, Pair, TileA, TileB>("GpuJobSchedulerTask", 1, false) {}

    void execute(std::shared_ptr<Job> job) override {
        if(job->shouldQuit()) {
            isDone_ = true;
            return;
        }

        assert(job_ == nullptr);

        job_ = job;
        ttl_ = job->tilesFromMatrixA().size() * job->tilesFromMatrixB().size();
        for(auto &colA = job->tilesFromMatrixA(); !colA.empty(); colA.pop_front()) {
            this->addResult(colA.front());
        }
        for(auto &colB = job->tilesFromMatrixB(); !colB.empty(); colB.pop_front()) {
            this->addResult(colB.front());
        }
    }

    void execute(std::shared_ptr<TileA> tileA) override {
        for(auto tileB: job_->tilesFromMatrixB()) {
            auto pair = std::make_shared<Pair>(std::make_tuple(tileA, tileB));
            this->addResult(pair);
        }
        job_->addTileA(tileA);
    }

    void execute(std::shared_ptr<TileB> tileB) override {
        for(auto tileA: job_->tilesFromMatrixA()) {
            auto pair = std::make_shared<Pair>(std::make_tuple(tileA, tileB));
            this->addResult(pair);
        }
        job_->addTileB(tileB);
    }

    void execute([[maybe_unused]] std::shared_ptr<TileP> tileP) override {
        ttl_--;
        if(ttl_ == 0) {
            job_->finished();
            job_ = nullptr;
        }
    }

    [[nodiscard]] bool canTerminate() const override {
        return isDone_ and job_ == nullptr;
    }

    std::shared_ptr<hh::AbstractTask<4, Job, TileA, TileB, TileP, Pair, TileA, TileB>>
    copy() override{
        return std::make_shared<GpuJobSchedulerTask<MatrixType, IdA, IdB, IdC, IdP>>();
    }

private:
    bool                                               isDone_ = false;
    int64_t                                            ttl_    = 0;
    std::shared_ptr<GpuJob<MatrixType, IdA, IdB, IdC>> job_    = nullptr;
};

template<typename MatrixType, char Id>
class HostToDeviceCopyTask: public hh::AbstractCUDATask<1, MatrixTile<MatrixType, Id>, MatrixTile<MatrixType, Id>> {
private:
    using Tile = MatrixTile<MatrixType, Id>;

public:
    explicit HostToDeviceCopyTask(int32_t threadCount = 1): hh::AbstractCUDATask<1, Tile, Tile>("H2D", threadCount, false, false) {}

    void execute(std::shared_ptr<Tile> tile) override {
        checkCudaErrors(cudaMemPrefetchAsync(tile->data(), tile->byteSize(), this->deviceId(), this->stream()));
        checkCudaErrors(cudaStreamSynchronize(this->stream()));
        this->addResult(tile);
    }

    std::shared_ptr<hh::AbstractTask<1, Tile, Tile>>
    copy() override {
        return std::make_shared<HostToDeviceCopyTask>(this->numberThreads());
    }
};

template<typename MatrixType, char IdA, char IdB, char IdP>
class ProductTask: public hh::AbstractCUDATask<
        1,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>>,
        MatrixTile<MatrixType, IdP>
    > {
private:
    using TileA = MatrixTile<MatrixType, IdA>;
    using TileB = MatrixTile<MatrixType, IdB>;
    using TileP = MatrixTile<MatrixType, IdP>;
    using Pair  = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>>;

public:
    explicit ProductTask(int32_t threadCount = 4):
        hh::AbstractCUDATask<1, Pair, TileP>("GEMM", threadCount, false) {}

    void initializeCuda() override {
        checkCudaErrors(cublasCreate_v2(&handle_));
        checkCudaErrors(cublasSetStream_v2(handle_, this->stream()));
    }

    void shutdownCuda() override {
        checkCudaErrors(cublasDestroy_v2(handle_));
    }

    void execute(std::shared_ptr<Pair> pair) override {
        auto tileA = std::get<std::shared_ptr<TileA>>(*pair);
        auto tileB = std::get<std::shared_ptr<TileB>>(*pair);
        MatrixType alpha = 1., beta = 0.;

        auto tileP = std::static_pointer_cast<TileP>(this->getManagedMemory());
        tileP->init(tileA->rowIdx(), tileB->colIdx(), tileA->height(), tileB->width());
        checkCudaErrors(cudaMemPrefetchAsync(
            tileP->data(),
            tileP->height()*tileP->width()*sizeof(MatrixType),
            this->deviceId(),
            this->stream()
        ));

        if constexpr(std::is_same_v<MatrixType, float>) {
            checkCudaErrors(cublasSgemm_v2(
                handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                tileA->height(), tileB->width(), tileA->width(), &alpha,
                (float *) tileA->data(), tileA->leadingDimension(),
                (float *) tileB->data(), tileB->leadingDimension(), &beta,
                (float *) tileP->data(), tileP->leadingDimension()
            ));
        }
        else if constexpr(std::is_same_v<MatrixType, double>) {
            checkCudaErrors(cublasDgemm_v2(
                handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                tileA->height(), tileB->width(), tileA->width(), &alpha,
                (double *) tileA->data(), tileA->leadingDimension(),
                (double *) tileB->data(), tileB->leadingDimension(), &beta,
                (double *) tileP->data(), tileP->leadingDimension()
            ));
        }
        else {
            throw std::runtime_error("Datatype not supported for cuda product task.");
        }
        checkCudaErrors(cudaStreamSynchronize(this->stream()));

        checkCudaErrors(cudaMemPrefetchAsync(
            tileP->data(),
            tileP->byteSize(),
            cudaCpuDeviceId,
            this->stream()
        ));
        tileP->recordEvent(this->stream());
        this->addResult(tileP);
    }

    std::shared_ptr<hh::AbstractTask<1, Pair, TileP>>
    copy() override {
        return std::make_shared<ProductTask>(this->numberThreads());
    }
private:
    cublasHandle_t handle_{};
};

template<typename MatrixType, char IdC, char IdP>
class AccumulateTask: public hh::AbstractTask<
        1,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdC>>, std::shared_ptr<MatrixTile<MatrixType, IdP>>>,
        MatrixTile<MatrixType, IdC>
    > {
private:
    using TileC = MatrixTile<MatrixType, IdC>;
    using TileP = MatrixTile<MatrixType, IdP>;
    using Pair  = std::tuple<std::shared_ptr<TileC>, std::shared_ptr<TileP>>;
public:
    explicit AccumulateTask(int32_t threadCount = 4):
        hh::AbstractTask<1, Pair, TileC>("Accumulate Task", threadCount, false) {}

    void execute(std::shared_ptr<Pair> pair) override {
        auto tileC = std::get<std::shared_ptr<TileC>>(*pair);
        auto tileP = std::get<std::shared_ptr<TileP>>(*pair);

        tileP->synchronizeEvent();
        accumulate(
            (MatrixType*)tileC->data(), tileC->leadingDimension(),
            (MatrixType*)tileP->data(), tileP->leadingDimension(),
            tileC->width(), tileC->height()
        );

        this->addResult(tileC);
        if(tileP->isMemoryManagerConnected()) {
            tileP->returnToMemoryManager();
        }
    }

    std::shared_ptr<hh::AbstractTask<1, Pair, TileC>>
    copy() override {
        return std::make_shared<AccumulateTask>();
    }

private:
    void accumulate(MatrixType *dataC, int64_t leadingDimensionC, MatrixType *dataP, int64_t leadingDimensionP, int64_t tileWidth, int64_t tileHeight) {
        if(leadingDimensionC == tileHeight and leadingDimensionP == tileHeight) {
            std::transform(dataC, dataC+tileWidth*tileHeight, dataP, dataC, std::plus<MatrixType>());
            return;
        }
        for(int64_t j = 0; j < tileWidth; ++j) {
            for(int64_t i = 0; i < tileHeight; ++i) {
                dataC[j*leadingDimensionC + i] += dataP[j*leadingDimensionP + i];
            }
        }
    }
};

#endif //HH3_MATMUL_TASKS

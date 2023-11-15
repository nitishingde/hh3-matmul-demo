#ifndef HH3_MATMUL_TASKS
#define HH3_MATMUL_TASKS

#include "common_tasks.h"
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
    explicit GpuJobGeneratorTask(const int64_t windowSize):
        hh::AbstractTask<3, Triplet, TileA, TileB, DbRequest<IdA>, DbRequest<IdB>, Job>("GpuJobGeneratorTask", 1, false),
        windowHeight_(windowSize), windowWidth_(windowSize)  {}

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
        for(int64_t kt = 0; kt < KT; ++kt) {
            auto jobK = JobK{
                .index    = kt,
                .priority = 0,
            };

            // evaluate priority
            for(auto rowIdx: rowIndices) {
                if(matrixA->tile(rowIdx, kt) != nullptr) {
                    jobK.priority++;
                }
            }
            for(auto colIdx: colIndices) {
                if(matrixB->tile(kt, colIdx) != nullptr) {
                    jobK.priority++;
                }
            }

            priorityQueue.emplace(jobK);
        }

        colTilesFromA_.resize(KT);
        rowTilesFromB_.resize(KT);
        for(; !priorityQueue.empty(); priorityQueue.pop()) {
            auto jobK = priorityQueue.top();
            auto kt   = jobK.index;

            // FIXME: works only for TwoDBlockCyclicMatrix and TwoDBlockCyclicContiguousSubMatrix
            auto reqA = std::make_shared<DbRequest<IdA>>(matrixA->owner(*rowIndices.begin(), kt));
            for(auto rowIdx: rowIndices) {
                if(auto tileA = matrixA->tile(rowIdx, kt); tileA != nullptr) {
                    colTilesFromA_[kt].emplace_back(tileA);
                }
                else {
                    reqA->addIndex(rowIdx, kt);
                }
            }
            this->addResult(reqA);

            // FIXME: works only for TwoDBlockCyclicMatrix and TwoDBlockCyclicContiguousSubMatrix
            auto reqB = std::make_shared<DbRequest<IdB>>(matrixB->owner(kt, *colIndices.begin()));
            for(auto colIdx: colIndices) {
                if(auto tileB = matrixB->tile(kt, colIdx); tileB != nullptr) {
                    rowTilesFromB_[kt].emplace_back(tileB);
                }
                else {
                    reqB->addIndex(kt, colIdx);
                }
            }
            this->addResult(reqB);
        }

        this->addResult(std::make_shared<DbRequest<IdA>>(true));
        this->addResult(std::make_shared<DbRequest<IdB>>(true));

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

        int64_t ttlA = windowWidth_  < int64_t(totalRowTilesB_)? int64_t(totalRowTilesB_+windowWidth_-1)/windowWidth_  : 1;
        int64_t ttlB = windowHeight_ < int64_t(totalColTilesA_)? int64_t(totalColTilesA_+windowHeight_-1)/windowHeight_: 1;
        for(auto &tileA: vecA) tileA->ttl(ttlA);
        for(auto &tileB: vecB) tileB->ttl(ttlB);

        for(size_t i = 0; i < vecB.size(); i+=windowWidth_) {
            for(size_t j = 0; j < vecA.size(); j+=windowHeight_) {
                auto job = std::make_shared<Job>();
                for(size_t ii = i; (ii < i+windowWidth_) and (ii < vecA.size()); ++ii) {
                    job->addTileA(vecA[ii]);
                }
                for(size_t jj = j; (jj < j+windowHeight_) and (jj < vecB.size()); ++jj) {
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
    int64_t                      windowHeight_      = -1;
    int64_t                      windowWidth_       = -1;
    size_t                       totalColTilesA_    = -1;
    size_t                       totalRowTilesB_    = -1;
    int64_t                      ttl_               = -1;
    Grid<std::shared_ptr<TileA>> colTilesFromA_     = {};
    Grid<std::shared_ptr<TileB>> rowTilesFromB_     = {};
};

template<typename MatrixType, char Id, char IdA = 'a', char IdB = 'b', char IdC = 'c'>
class MatrixWarehousePrefetchTask: public hh::AbstractTask<
        2,
        std::tuple<std::shared_ptr<MatrixContainer<MatrixType, Id>>, std::shared_ptr<MatrixContainer<MatrixType, IdC>>>,
        DbRequest<Id>,
        MatrixTile<MatrixType, Id>
    >{
private:
    using Matrix     = MatrixContainer<MatrixType, Id>;
    using MatrixC    = MatrixContainer<MatrixType, IdC>;
    using Pair       = std::tuple<std::shared_ptr<MatrixContainer<MatrixType, Id>>, std::shared_ptr<MatrixContainer<MatrixType, IdC>>>;
    using Tile       = MatrixTile<MatrixType, Id>;
    using DB_Request = DbRequest<Id>;

public:
    explicit MatrixWarehousePrefetchTask(std::atomic_int32_t *pStop):
        hh::AbstractTask<2, Pair, DB_Request, Tile>("Prefetch Task", 1, false), pStop_(pStop) {
        assert(pStop != nullptr);
        tagOffset_ = (Id == IdA? 0: 100000);
    }

    void execute(std::shared_ptr<Pair> pair) override {
        assert(Id == IdA or Id == IdB);
        assert(pStop_ != nullptr);
        matrix_      = std::get<std::shared_ptr<Matrix>>(*pair);
        auto matrixC = std::get<std::shared_ptr<MatrixC>>(*pair);

        // initiate all the mpi isends
        // FIXME: this is hardcoded for prototyping. if successful, need to come up with a generic way to establish dependencies
        if constexpr(Id == IdA) {
            std::lock_guard lockGuard(mpiMutex);
            for(int64_t rowIdx = 0; rowIdx < matrix_->matrixNumRowTiles(); rowIdx++) {
                std::vector<std::shared_ptr<Tile>> tiles = {};
                for(int64_t colIdx = 0; colIdx < matrix_->matrixNumColTiles(); colIdx++) {
                    if(auto tile = matrix_->tile(rowIdx, colIdx); tile != nullptr) {
                        tiles.emplace_back(tile);
                    }
                }
                if(tiles.empty()) continue;

                std::vector<bool> done(getNumNodes(), false);
                done[getNodeId()] = true;
                for(int64_t colIdx = 0; colIdx < matrixC->matrixNumColTiles(); colIdx++) {
                    auto destination = matrixC->owner(rowIdx, colIdx);
                    if(done[destination]) continue;

                    done[destination] = true;
                    for(const std::shared_ptr<Tile> &tile: tiles) {
                        mpiSends_.emplace_back(MPI_Request{});
                        int32_t tag = tagOffset_ + tile->rowIdx()*matrix_->matrixNumColTiles() + tile->colIdx();
                        checkMpiErrors(MPI_Issend(tile->data(), tile->byteSize(), MPI_CHAR, destination, tag, matrix_->mpiComm(), &mpiSends_.back()));
                    }
                }
            }
        }
        else if constexpr(Id == IdB) {
            std::lock_guard lockGuard(mpiMutex);
            for(int64_t colIdx = 0; colIdx < matrix_->matrixNumColTiles(); colIdx++) {
                std::vector<std::shared_ptr<Tile>> tiles = {};
                for(int64_t rowIdx = 0; rowIdx < matrix_->matrixNumColTiles(); rowIdx++) {
                    if(auto tile = matrix_->tile(rowIdx, colIdx); tile != nullptr) {
                        tiles.emplace_back(tile);
                    }
                }

                if(tiles.empty()) continue;

                std::vector<bool> done(getNumNodes(), false);
                done[getNodeId()] = true;
                for(int64_t rowIdx = 0; rowIdx < matrix_->matrixNumColTiles(); rowIdx++) {
                    auto destination = matrixC->owner(rowIdx, colIdx);
                    if(done[destination]) continue;

                    done[destination] = true;
                    for(const std::shared_ptr<Tile> &tile: tiles) {
                        mpiSends_.emplace_back(MPI_Request{});
                        int32_t tag = tagOffset_ + tile->rowIdx()*matrix_->matrixNumColTiles() + tile->colIdx();
                        checkMpiErrors(MPI_Issend(tile->data(), tile->byteSize(), MPI_CHAR, destination, tag, matrix_->mpiComm(), &mpiSends_.back()));
                    }
                }
            }
        }
        else {
            throw std::runtime_error("MatrixWarehousePrefetchTask-Id\n");
        }

        for(pStop_->fetch_sub(1); pStop_->load();) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(4ms);
        }
    }

    void execute(std::shared_ptr<DB_Request> dbRequest) override {
        requests_.emplace_back(dbRequest);
        handleRequests();
    }

    [[nodiscard]] std::string extraPrintingInformation() const override {
        DotTimer dotTimer;
        dotTimer.merge(dotTimer_);

        double tileSize = (matrix_->tileDim()*matrix_->tileDim()*sizeof(MatrixType))/(1024.*1024.);
        auto minBw = std::to_string(tileSize/dotTimer.max());
        auto avgBw = std::to_string(tileSize/dotTimer.avg());
        auto maxBw = std::to_string(tileSize/dotTimer.min());

        auto min   = std::to_string(dotTimer.min());
        auto avg   = std::to_string(dotTimer.avg());
        auto max   = std::to_string(dotTimer.max());

        return "#Requests received: " + std::to_string(dotTimer.count()) + "\\n"
            "Bandwidth per request:\\n"
            "Min: " + minBw.substr(0, minBw.find('.', 0)+4) + "MB/s\\n"
            "Avg: " + avgBw.substr(0, avgBw.find('.', 0)+4) + "MB/s\\n"
            "Max: " + maxBw.substr(0, maxBw.find('.', 0)+4) + "MB/s\\n"
            "Time per request:\\n"
            "Min: " + min.substr(0, min.find('.', 0)+4)  + "s\\n"
            "Avg: " + avg.substr(0, avg.find('.', 0)+4)  + "s\\n"
            "Max: " + max.substr(0, max.find('.', 0)+4)  + "s\\n"
            "Total time spent on requests: " + std::to_string(dotTimer.totalTime()) + "s"
            ;
    }

private:
    void handleRequests() {
        if(matrix_ == nullptr) return;

        for(; !requests_.empty(); requests_.pop_front()) {
            auto dbRequest = requests_.front();

            // MPI receive
            for(auto [rowIdx, colIdx]: dbRequest->indices) {
                if(auto tile = matrix_->tile(rowIdx, colIdx); tile!= nullptr) {
                    this->addResult(tile);
                    continue;
                }

                auto tile   = std::static_pointer_cast<Tile>(this->getManagedMemory());
                tile->init(rowIdx, colIdx,  matrix_->tileHeight(rowIdx, colIdx), matrix_->tileWidth(rowIdx, colIdx));
                int32_t tag = tagOffset_ + rowIdx*matrix_->matrixNumColTiles() + colIdx;

                {
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(4ms);
                    std::lock_guard lockGuard(mpiMutex);
                    dotTimer_.start();
                    checkMpiErrors(MPI_Recv(tile->data(), tile->byteSize(), MPI_CHAR, matrix_->owner(rowIdx, colIdx), tag, matrix_->mpiComm(), MPI_STATUS_IGNORE));
                    dotTimer_.stop();

                    this->addResult(tile);
                }
            }
        }
    }

private:
    std::atomic_int32_t                     *pStop_     = nullptr;
    std::shared_ptr<Matrix>                 matrix_     = nullptr;
    std::deque<std::shared_ptr<DB_Request>> requests_   = {};
    DotTimer                                dotTimer_     {};
    int32_t                                 tagOffset_  = 0;
    std::vector<MPI_Request>                mpiSends_   = {};
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

        dotTimer_.start();
        job_  = job;
        ttl_  = job->tilesFromMatrixA().size() * job->tilesFromMatrixB().size();
        ttlA_ = job->tilesFromMatrixB().size();
        ttlB_ = job->tilesFromMatrixA().size();
        for(auto &colA = job->tilesFromMatrixA(); !colA.empty(); colA.pop_front()) {
            this->addResult(colA.front());
        }
        for(auto &colB = job->tilesFromMatrixB(); !colB.empty(); colB.pop_front()) {
            this->addResult(colB.front());
        }
    }

    void execute(std::shared_ptr<TileA> tileA) override {
        tileA->ttl(ttlA_);
        for(auto tileB: job_->tilesFromMatrixB()) {
            auto pair = std::make_shared<Pair>(std::make_tuple(tileA, tileB));
            this->addResult(pair);
        }
        job_->addTileA(tileA);
    }

    void execute(std::shared_ptr<TileB> tileB) override {
        tileB->ttl(ttlB_);
        for(auto tileA: job_->tilesFromMatrixA()) {
            auto pair = std::make_shared<Pair>(std::make_tuple(tileA, tileB));
            this->addResult(pair);
        }
        job_->addTileB(tileB);
    }

    void execute(std::shared_ptr<TileP> tileP) override {
        ttl_--;
        if(ttl_ == 0) {
            job_->finished();
            job_ = nullptr;
            dotTimer_.stop();
        }

        if(tileP->isMemoryManagerConnected()) {
            tileP->used();
            tileP->returnToMemoryManager();
        }
    }

    [[nodiscard]] bool canTerminate() const override {
        return isDone_ and job_ == nullptr;
    }

    std::shared_ptr<hh::AbstractTask<4, Job, TileA, TileB, TileP, Pair, TileA, TileB>>
    copy() override{
        return std::make_shared<GpuJobSchedulerTask<MatrixType, IdA, IdB, IdC, IdP>>();
    }

    [[nodiscard]] std::string extraPrintingInformation() const override {
        auto dotTimer = this->dotTimer_;
        auto suffix = dotTimer.format();

        auto min = std::to_string(dotTimer.min());
        auto avg = std::to_string(dotTimer.avg());
        auto max = std::to_string(dotTimer.max());
        return "#Jobs received: " + std::to_string(dotTimer.count()) + "\\n"
            "Execution Time Per Job:\\n"
            "Min: " + min.substr(0, min.find('.', 0)+4) + suffix + "\\n"
            "Avg: " + avg.substr(0, avg.find('.', 0)+4) + suffix + "\\n"
            "Max: " + max.substr(0, max.find('.', 0)+4) + suffix + "\\n"
            "Total time spent on jobs: " + std::to_string(dotTimer.totalTime()) + suffix
            ;
    }

private:
    bool                                               isDone_   = false;
    int64_t                                            ttl_      = 0;
    int64_t                                            ttlA_     = 0;
    int64_t                                            ttlB_     = 0;
    std::shared_ptr<GpuJob<MatrixType, IdA, IdB, IdC>> job_      = nullptr;
    DotTimer                                           dotTimer_   {};
};

template<typename MatrixType, char Id>
class HostToDeviceCopyTask: public hh::AbstractCUDATask<1, MatrixTile<MatrixType, Id>, MatrixTile<MatrixType, Id>, GcMatrixTile<MatrixType, Id>> {
private:
    using Tile   = MatrixTile<MatrixType, Id>;
    using GcTile = GcMatrixTile<MatrixType, Id>;

public:
    explicit HostToDeviceCopyTask(int32_t threadCount = 1): hh::AbstractCUDATask<1, Tile, Tile, GcTile>("H2D", threadCount, false, false) {}

    void execute(std::shared_ptr<Tile> tile) override {
        assert(tile->major() == Major::COL);
        assert(tile->leadingDimension() == tile->height());

        auto cudaTile = std::static_pointer_cast<Tile>(this->getManagedMemory());
        cudaTile->init(tile->rowIdx(), tile->colIdx(), tile->height(), tile->width());
        checkCudaErrors(cudaMemcpyAsync(
            cudaTile->data(),
            reinterpret_cast<const void*>(tile->data()),
            tile->byteSize(),
            cudaMemcpyHostToDevice,
            this->stream()
        ));

        cudaTile->recordEvent(this->stream(), this->deviceId());
        this->addResult(cudaTile);

        tile->recordEvent(this->stream(), this->deviceId());
        this->addResult(std::make_shared<GcTile>(tile));
    }

    std::shared_ptr<hh::AbstractTask<1, Tile, Tile, GcTile>>
    copy() override {
        return std::make_shared<HostToDeviceCopyTask>(this->numberThreads());
    }
};

template<typename MatrixType, char IdA, char IdB>
class GarbageCollectorTask: public hh::AbstractCUDATask<2, GcMatrixTile<MatrixType, IdA>, GcMatrixTile<MatrixType, IdB>, void*> {
private:
    using GcTileA = GcMatrixTile<MatrixType, IdA>;
    using GcTileB = GcMatrixTile<MatrixType, IdB>;

public:
    explicit GarbageCollectorTask(int32_t threadCount = 1):
        hh::AbstractCUDATask<2, GcTileA, GcTileB, void*>("GC", threadCount, false) {}

    void execute(std::shared_ptr<GcTileA> data) override {
        auto &tile = data->tile;
        tile->used();
        if(!tile->isMemoryManagerConnected()) return;

        tile->synchronizeEvent(this->deviceId());
        tile->returnToMemoryManager();
    }

    void execute(std::shared_ptr<GcTileB> data) override {
        auto &tile = data->tile;
        tile->used();
        if(!tile->isMemoryManagerConnected()) return;

        tile->synchronizeEvent(this->deviceId());
        tile->returnToMemoryManager();
    }

    std::shared_ptr<hh::AbstractTask<2, GcTileA, GcTileB, void*>>
    copy() override {
        return std::make_shared<GarbageCollectorTask>(this->numberThreads());
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
        tileP->ttl(2);
        checkCudaErrors(cudaMemPrefetchAsync(
            tileP->data(),
            tileP->height()*tileP->width()*sizeof(MatrixType),
            this->deviceId(),
            this->stream()
        ));

        tileA->synchronizeEvent(this->deviceId());
        tileB->synchronizeEvent(this->deviceId());

        dotTimer_.start();
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
        dotTimer_.stop();

        checkCudaErrors(cudaMemPrefetchAsync(
            tileP->data(),
            tileP->byteSize(),
            cudaCpuDeviceId,
            this->stream()
        ));
        tileP->recordEvent(this->stream());
        this->addResult(tileP);

        tileA->used();
        if(tileA->isMemoryManagerConnected()) {
            tileA->returnToMemoryManager();
        }

        tileB->used();
        if(tileB->isMemoryManagerConnected()) {
            tileB->returnToMemoryManager();
        }
    }

    [[nodiscard]] std::string extraPrintingInformation() const override {
        DotTimer dotTimer;
        for(auto pNode: this->group()) {
            auto pTask = dynamic_cast<ProductTask const *>(pNode);
            dotTimer.merge(pTask->dotTimer_);
        }

        auto suffix = dotTimer.format();

        auto min = std::to_string(dotTimer.min());
        auto avg = std::to_string(dotTimer.avg());
        auto max = std::to_string(dotTimer.max());

        return "#Gemms received: " + std::to_string(dotTimer.count()) + "\\n"
            "Execution Time Per Gemm:\\n"
            "Min: " + min.substr(0, min.find('.', 0)+4) + suffix + "\\n"
            "Avg: " + avg.substr(0, avg.find('.', 0)+4) + suffix + "\\n"
            "Max: " + max.substr(0, max.find('.', 0)+4) + suffix + "\\n"
            "Total time spent on gemms: " + std::to_string(dotTimer.totalTime()) + suffix
            ;
    }

    std::shared_ptr<hh::AbstractTask<1, Pair, TileP>>
    copy() override {
        return std::make_shared<ProductTask>(this->numberThreads());
    }
private:
    cublasHandle_t handle_  {};
    DotTimer       dotTimer_{};
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
        dotTimer_.start();
        accumulate(
            (MatrixType*)tileC->data(), tileC->leadingDimension(),
            (MatrixType*)tileP->data(), tileP->leadingDimension(),
            tileC->width(), tileC->height()
        );
        dotTimer_.stop();

        this->addResult(tileC);
        if(tileP->isMemoryManagerConnected()) {
            tileP->used();
            tileP->returnToMemoryManager();
        }
    }

    [[nodiscard]] std::string extraPrintingInformation() const override {
        DotTimer dotTimer;
        for(auto pNode: this->group()) {
            auto pTask = dynamic_cast<AccumulateTask const *>(pNode);
            dotTimer.merge(pTask->dotTimer_);
        }

        auto suffix = dotTimer.format();

        auto min = std::to_string(dotTimer.min());
        auto avg = std::to_string(dotTimer.avg());
        auto max = std::to_string(dotTimer.max());

        return "#Accumaltes received: " + std::to_string(dotTimer.count()) + "\\n"
            "Execution Time Per accumulate:\\n"
            "Min: " + min.substr(0, min.find('.', 0)+4) + suffix + "\\n"
            "Avg: " + avg.substr(0, avg.find('.', 0)+4) + suffix + "\\n"
            "Max: " + max.substr(0, max.find('.', 0)+4) + suffix + "\\n"
            "Total time spent on accumulation: " + std::to_string(dotTimer.totalTime()) + suffix
            ;
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

private:
    DotTimer dotTimer_{};
};

#endif //HH3_MATMUL_TASKS

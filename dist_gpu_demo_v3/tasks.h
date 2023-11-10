#ifndef HH3_MATMUL_TASKS
#define HH3_MATMUL_TASKS

#include "data.h"

template<typename MatrixType, char IdA, char IdB, char IdC>
class GpuJob {
private:
    using TileA      = MatrixTile<MatrixType, IdA>;
    using TileB      = MatrixTile<MatrixType, IdB>;
    using TileC      = MatrixTile<MatrixType, IdC>;

public:
    explicit GpuJob(bool shouldQuit = false): quit_(shouldQuit), gpuToken_(std::make_shared<GpuToken>(-1)) {}

    ~GpuJob() {
        finished();
    }

    void finished() {
        if(gpuToken_ and gpuToken_->isMemoryManagerConnected()) {
            gpuToken_->returnToMemoryManager();
        }
        gpuToken_ = nullptr;
    }

    // Getters
    [[nodiscard]] int32_t gpuId()            const { return gpuToken_->id; }
    [[nodiscard]] bool    shouldQuit()       const { return quit_;         }
    [[nodiscard]] auto&   tilesFromMatrixC()       { return tileCs_;       }

    // Setters
    void token(std::shared_ptr<GpuToken> token) { gpuToken_ = std::move(token); }
    void quit(const bool flag)                  { quit_ = flag;                 }
    void addTileC(std::shared_ptr<TileC> tileC) { tileCs_.emplace_back(tileC);  }

public:
    int32_t                             height   = 0;
    int32_t                             width    = 0;

private:
    std::vector<std::shared_ptr<TileC>> tileCs_   = {};
    std::shared_ptr<GpuToken>           gpuToken_ = nullptr;
    bool                                quit_     = false;
};

template<typename MatrixType, char IdA, char IdB, char IdC>
class GpuJobGeneratorTask: public hh::AbstractTask<
        1,
        std::tuple<std::shared_ptr<MatrixContainer<MatrixType, IdA>>, std::shared_ptr<MatrixContainer<MatrixType, IdB>>, std::shared_ptr<MatrixContainer<MatrixType, IdC>>>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        DbRequest<IdA>,
        DbRequest<IdB>,
        GpuJob<MatrixType, IdA, IdB, IdC>
    > {
private:
    using MatrixA = MatrixContainer<MatrixType, IdA>;
    using MatrixB = MatrixContainer<MatrixType, IdB>;
    using MatrixC = MatrixContainer<MatrixType, IdC>;
    using Triplet = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>;
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using Job     = GpuJob<MatrixType, IdA, IdB, IdC>;

public:
    explicit GpuJobGeneratorTask(const size_t gp, const size_t gq, const int64_t windowHeight, const int64_t windowWidth):
        hh::AbstractTask<1, Triplet, TileA, TileB, DbRequest<IdA>, DbRequest<IdB>, Job>("GpuJobGeneratorTask", 1, false),
        deviceCount_(gp*gq), gp0_(gp), gq0_(gq), windowHeight_(windowHeight), windowWidth_(windowWidth) {}

    void execute(std::shared_ptr<Triplet> triplet) override {
        auto matrixA = std::get<std::shared_ptr<MatrixA>>(*triplet);
        auto matrixB = std::get<std::shared_ptr<MatrixB>>(*triplet);
        auto matrixC = std::get<std::shared_ptr<MatrixC>>(*triplet);
        auto KT = matrixA->matrixNumColTiles();

        std::set<int64_t> rowIndicesSet = {};
        std::set<int64_t> colIndicesSet = {};
        for(int64_t rowIdx = 0; rowIdx < matrixC->matrixNumRowTiles(); ++rowIdx) {
            for(int64_t colIdx = 0; colIdx < matrixC->matrixNumColTiles(); ++colIdx) {
                if(matrixC->tile(rowIdx, colIdx)) {
                    rowIndicesSet.emplace(rowIdx);
                    colIndicesSet.emplace(colIdx);
                }
            }
        }
        std::vector<int64_t> rowIndices(rowIndicesSet.begin(), rowIndicesSet.end());
        std::vector<int64_t> colIndices(colIndicesSet.begin(), colIndicesSet.end());
//        auto priorityQueue = getPrioritySequence(matrixA, matrixB, rowIndices, colIndices);//FIXME: not been used

#ifndef NDEBUG
        auto MT = matrixC->matrixNumRowTiles(), NT = matrixC->matrixNumColTiles();
        auto debug = std::vector<std::vector<int32_t>>(MT, std::vector<int32_t>(NT, -1));
        auto print = [&debug, MT, NT]() {
            char buffer[4096];
            int32_t len = 0;
            memset(buffer, 0, sizeof(buffer));

            // row header
            len += sprintf(&buffer[len], "     ");
            for(int j = 0; j < NT; ++j) len += sprintf(&buffer[len], "%2d, ", j);
            len -= 2;
            len += sprintf(&buffer[len], "\n    ");
            for(int j = 0; j < NT; ++j) len += sprintf(&buffer[len], "----");
            len += sprintf(&buffer[len], "\n");

            for(int i = 0; i < MT; ++i) {
                len += sprintf(&buffer[len], "%2d | ", i);
                for(int j = 0; j < NT; ++j) {
                    len += sprintf(&buffer[len], "%2d, ", debug[i][j]);
                }
                len -= 2;
                len += sprintf(&buffer[len], "\n");
            }
            printf("%s\n", buffer);
            fflush(stdout);
        };
        print();
#endif

        for(size_t i = 0; i < rowIndices.size(); i+= (gp0_*windowHeight_)) {
            for(size_t j = 0; j < colIndices.size(); j+= (gq0_*windowWidth_)) {
                this->taskBarrier();
                rowIndicesSet.clear();
                colIndicesSet.clear();
                for(size_t gp = 0; gp < gp0_; ++gp) {
                    for(size_t gq = 0; gq < gq0_; ++gq) {
                        auto job   = std::make_shared<Job>();
                        auto token = std::static_pointer_cast<GpuToken>(this->getManagedMemory());
                        job->token(token);
                        token->id = gp*gp0_+gq;//FIXME
                        for(size_t ii = i+gp; (job->height < windowHeight_) and (ii < rowIndices.size()); ii+=gp0_) {
                            job->height++;
                            job->width = 0;
                            for(size_t jj = j+gq; (job->width < windowWidth_) and (jj < colIndices.size()); jj+=gq0_) {
                                rowIndicesSet.insert(rowIndices[ii]);
                                colIndicesSet.insert(colIndices[jj]);
#ifndef NDEBUG
                                debug[rowIndices[ii]][colIndices[jj]] = int32_t(token->id);
#endif
                                job->addTileC(matrixC->tile(rowIndices[ii], colIndices[jj]));
                                job->width++;
                            }
                        }

                        if(job->tilesFromMatrixC().empty()) continue;
                        this->addResult(job);
                    }
                }

#ifndef NDEBUG
                print();
#endif

//                for(int64_t kt = 0; kt < KT; ++kt) {
//                    auto reqA = std::make_shared<DbRequest<IdA>>(matrixA->owner(*rowIndicesSet.begin(), kt));
//                    for(auto rowIdx: rowIndicesSet) {
//                        reqA->addIndex(rowIdx, kt);
//                    }
//                    this->addResult(reqA);
//
//                    auto reqB = std::make_shared<DbRequest<IdB>>(matrixB->owner(kt, *colIndicesSet.begin()));
//                    for(auto colIdx: colIndicesSet) {
//                        reqB->addIndex(kt, colIdx);
//                    }
//                    this->addResult(reqB);
//                }
                for(int64_t kt = 0; kt < KT; ++kt) {
                    for(auto rowIdx: rowIndicesSet) {
                        this->addResult(matrixA->tile(rowIdx, kt));
                    }

                    for(auto colIdx: colIndicesSet) {
                        this->addResult(matrixB->tile(kt, colIdx));
                    }
                }
            }
        }

//        this->addResult(std::make_shared<DbRequest<IdA>>(true));
//        this->addResult(std::make_shared<DbRequest<IdB>>(true));

        this->taskBarrier();
        this->addResult(std::make_shared<Job>(true));
    }

private:
    std::vector<int64_t> getPrioritySequence(std::shared_ptr<MatrixA> &matrixA, std::shared_ptr<MatrixB> &matrixB, const std::vector<int64_t> &rowIndices, const std::vector<int64_t> &colIndices) {
        struct JobK {
            int64_t index    = 0;
            int64_t priority = 0;

            bool operator<(const JobK &other) {
                return priority < other.priority;
            }
        };
        std::vector<JobK> priorityQueue;
        auto KT = matrixA->matrixNumColTiles();

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

            priorityQueue.emplace_back(jobK);
        }
        std::sort(priorityQueue.begin(), priorityQueue.end());

        std::vector<int64_t> priority;
        priority.reserve(priorityQueue.size());
        for(const auto &jobK: priorityQueue) priority.emplace_back(jobK.index);//FIXME: higher valued priority should be first

        return priority;
    }

    void taskBarrier() {
        using namespace std::chrono_literals;
        while(this->memoryManager()->currentSize() != this->memoryManager()->capacity()) {
            std::this_thread::sleep_for(4ms);
        }
    }

private:
    size_t  deviceCount_  = 0;
    size_t  gp0_          = 0;
    size_t  gq0_          = 0;
    int64_t windowHeight_ = 0;
    int64_t windowWidth_  = 0;
};

template<typename MatrixType, char IdA, char IdB>
class TileSorterTask: public hh::AbstractTask<
        2,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>
    > {
private:
    using TileA = MatrixTile<MatrixType, IdA>;
    using TileB = MatrixTile<MatrixType, IdB>;

public:
    explicit TileSorterTask(size_t gp, size_t gq):
        hh::AbstractTask<2, TileA, TileB, TileA, TileB>("TileSorter", 1, false), gp_(gp), gq_(gq) {}

    void execute(std::shared_ptr<TileA> tileA) override {
        tileA->ttl(gq_);
        this->addResult(tileA);
    }

    void execute(std::shared_ptr<TileB> tileB) override {
        tileB->ttl(gp_);
        this->addResult(tileB);
    }
private:
    size_t gp_ = 0;
    size_t gq_ = 0;
};

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
};

template<typename MatrixType, char IdA, char IdB, char IdC>
class GpuJobSchedulerTask: public hh::AbstractTask<
        4,
        GpuJob<MatrixType, IdA, IdB, IdC>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        MatrixTile<MatrixType, IdC>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        MatrixTile<MatrixType, IdC>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>, std::shared_ptr<MatrixTile<MatrixType, IdC>>>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdC>>, std::shared_ptr<MatrixTile<MatrixType, IdC>>>
    > {
private:
    template<typename T>
    using Grid    = std::vector<std::vector<T>>;
    using Job     = GpuJob<MatrixType, IdA, IdB, IdC>;
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using TileC   = MatrixTile<MatrixType, IdC>;
    using Triplet = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>, std::shared_ptr<TileC>>;
    using Pair    = std::tuple<std::shared_ptr<TileC>, std::shared_ptr<TileC>>;

    struct JobPair {
        std::shared_ptr<TileA> cudaTileA = nullptr;
        std::shared_ptr<TileB> cudaTileB = nullptr;
    };

public:
    explicit GpuJobSchedulerTask(const int64_t MT, const int64_t KT, const int64_t NT):
        hh::AbstractTask<4, Job, TileA, TileB, TileC, TileA, TileB, TileC, Triplet, Pair>("GpuJobSchedulerTask", 1, false),
        MT_(MT), KT_(KT), NT_(NT) {
        ttlKt_.resize(KT);
        gridCudaTileA_.resize(KT);
        gridCudaTileB_.resize(KT);
        gridCudaTileC_.resize(MT, std::vector<std::shared_ptr<TileC>>(NT, nullptr));
    }

    void execute(std::shared_ptr<Job> job) override {
        if(job->shouldQuit()) {
            isDone_ = true;
            return;
        }

        assert(job_ == nullptr);

        dotTimer_.start();
        job_  = job;
        ttlA_ = job->width;
        ttlB_ = job->height;
        ttl_  = (KT_+1)*job->width*job->height;

        std::fill(ttlKt_.begin(), ttlKt_.end(), job->width*job->height);
        queue_.clear();

        for(int32_t kt = 0; kt < KT_; ++kt) {
            gridCudaTileA_[kt].clear();
            gridCudaTileB_[kt].clear();
        }

        for(int64_t row = 0; row < MT_; ++row) {
            for(int64_t col = 0; col < NT_; ++col) {
                gridCudaTileC_[row][col] = nullptr;
            }
        }

        rowIndices_.clear();
        colIndices_.clear();
        for(auto &tileC: job->tilesFromMatrixC()) {
            rowIndices_.insert(tileC->rowIdx());
            colIndices_.insert(tileC->colIdx());
            tileC->ttl(KT_);
            this->addResult(tileC);
        }

//        std::stringstream rows, cols;
//        for(auto rowIdx: rowIndices_) rows << rowIdx << ", ";
//        rows <<"\b\b";
//        for(auto colIdx: colIndices_) cols << colIdx << ", ";
//        cols <<"\b\b";
//        printf("[GPU %d][GpuJobSchedulerTask][ttlA_ %ld][ttlB_ %ld][ttl_ %ld][width %d][height %d][rows %s][cols %s]\n", this->deviceId(), ttlA_, ttlB_, ttl_, job->width, job->height, rows.rdbuf()->str().c_str(), cols.rdbuf()->str().c_str());
//        fflush(stdout);
    }

    void execute(std::shared_ptr<TileA> tileA) override {
        if(!rowIndices_.contains(tileA->rowIdx())) return;

        if(tileA->memoryType() == MemoryType::HOST) {
            this->addResult(tileA);
            return;
        }

        auto &cudaTileA = tileA;
        cudaTileA->ttl(ttlA_);
        int64_t kt = cudaTileA->colIdx();
        for(auto cudaTileB: gridCudaTileB_[kt]) {
            if(auto cudaTileC = gridCudaTileC_[cudaTileA->rowIdx()][cudaTileB->colIdx()]; cudaTileC != nullptr) {
                gridCudaTileC_[cudaTileA->rowIdx()][cudaTileB->colIdx()] = nullptr;
                auto triplet = std::make_shared<Triplet>(std::make_tuple(cudaTileA, cudaTileB, cudaTileC));
                this->addResult(triplet);
                ttlKt_[kt]--;
            }
            else {
                queue_.emplace_back(JobPair{
                    .cudaTileA = cudaTileA,
                    .cudaTileB = cudaTileB
                });
            }
        }
        gridCudaTileA_[kt].emplace_back(cudaTileA);
        cleanUp(kt);
    }

    void execute(std::shared_ptr<TileB> tileB) override {
        if(!colIndices_.contains(tileB->colIdx())) return;

        if(tileB->memoryType() == MemoryType::HOST) {
            this->addResult(tileB);
            return;
        }

        auto &cudaTileB = tileB;
        cudaTileB->ttl(ttlB_);
        int64_t kt = cudaTileB->rowIdx();
        for(auto cudaTileA: gridCudaTileA_[kt]) {
            if(auto cudaTileC = gridCudaTileC_[cudaTileA->rowIdx()][cudaTileB->colIdx()]; cudaTileC != nullptr) {
                gridCudaTileC_[cudaTileA->rowIdx()][cudaTileB->colIdx()] = nullptr;
                auto triplet = std::make_shared<Triplet>(std::make_tuple(cudaTileA, cudaTileB, cudaTileC));
                this->addResult(triplet);
                ttlKt_[kt]--;
            }
            else {
                queue_.emplace_back(JobPair{
                    .cudaTileA = cudaTileA,
                    .cudaTileB = cudaTileB
                });
            }
        }
        gridCudaTileB_[kt].emplace_back(cudaTileB);
        cleanUp(kt);
    }

    void execute(std::shared_ptr<TileC> cudaTileC) override {
        ttl_--;

        int64_t rowIdx = cudaTileC->rowIdx(), colIdx = cudaTileC->colIdx();
        if(cudaTileC->canBeRecycled()) {
            auto &vecC = job_->tilesFromMatrixC();
            auto it = std::find_if(vecC.begin(), vecC.end(), [rowIdx, colIdx](std::shared_ptr<TileC> tile) {
                return tile->rowIdx() == rowIdx and tile->colIdx() == colIdx;
            });
            this->addResult(std::make_shared<Pair>(std::make_tuple(cudaTileC, *it)));
            vecC.erase(it);
            if(ttl_ == 0) {
                job_->finished();
                job_ = nullptr;
                dotTimer_.stop();
            }
            return;
        }


        for(auto it = queue_.begin(); it != queue_.end(); ++it) {
            if(it->cudaTileA->rowIdx() == rowIdx and it->cudaTileB->colIdx() == colIdx) {
                this->addResult(std::make_shared<Triplet>(std::make_tuple(it->cudaTileA, it->cudaTileB, cudaTileC)));
                ttlKt_[it->cudaTileA->colIdx()]--;
                cleanUp(it->cudaTileA->colIdx());
                queue_.erase(it);
                return;
            }
        }

        gridCudaTileC_[rowIdx][colIdx] = cudaTileC;
    }

    [[nodiscard]] bool canTerminate() const override {
        return isDone_ and job_ == nullptr;
    }

    std::shared_ptr<hh::AbstractTask<4, Job, TileA, TileB, TileC, TileA, TileB, TileC, Triplet, Pair>>
    copy() override {
        return std::make_shared<GpuJobSchedulerTask<MatrixType, IdA, IdB, IdC>>(this->MT_, this->KT_, this->NT_);
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
    void cleanUp(int64_t kt) {
//        std::stringstream ss;
//        ss << "vecA: [";
//        for(auto &vec: gridCudaTileA_) {
//            ss << vec.size() << ' ';
//        }
//        ss << "]\n";
//        ss << "vecB: [";
//        for(auto &vec: gridCudaTileB_) {
//            ss << vec.size() << ' ';
//        }
//        ss << "]\n";
//        ss << "ttlKt:[";
//        for(auto k: ttlKt_) {
//            ss << k << ' ';
//        }
//        ss << "]\n";
//        ss << "queue: " << queue_.size() << "\n";
//        printf("%s", ss.str().c_str());

        if(ttlKt_[kt] != 0) return;

        gridCudaTileA_[kt].clear();
        gridCudaTileB_[kt].clear();
    }

private:
    bool                                                          isDone_             = false;
    int64_t                                                       MT_                 = 0;
    int64_t                                                       KT_                 = 0;
    int64_t                                                       NT_                 = 0;
    int64_t                                                       ttl_                = 0;
    int64_t                                                       ttlA_               = 0;
    int64_t                                                       ttlB_               = 0;
    std::vector<int64_t>                                          ttlKt_              = {};
    Grid<std::shared_ptr<TileA>>                                  gridCudaTileA_      = {};
    Grid<std::shared_ptr<TileB>>                                  gridCudaTileB_      = {};
    Grid<std::shared_ptr<TileC>>                                  gridCudaTileC_      = {};
    std::deque<JobPair>                                           queue_              = {};
    std::shared_ptr<GpuJob<MatrixType, IdA, IdB, IdC>>            job_                = nullptr;
    std::set<int64_t>                                             rowIndices_         = {};
    std::set<int64_t>                                             colIndices_         = {};
    DotTimer                                                      dotTimer_             {};
};

template<typename MatrixType, char Id>
class BlockingHostToDeviceCopyTask: public hh::AbstractCUDATask<1, MatrixTile<MatrixType, Id>, MatrixTile<MatrixType, Id>> {
private:
    using Tile   = MatrixTile<MatrixType, Id>;

public:
    explicit BlockingHostToDeviceCopyTask(int32_t threadCount = 1): hh::AbstractCUDATask<1, Tile, Tile>("H2D", threadCount, false, false) {}

    void execute(std::shared_ptr<Tile> tile) override {
        assert(tile->major() == Major::COL);
        assert(tile->leadingDimension() == tile->height());

        auto cudaTile = std::static_pointer_cast<Tile>(this->getManagedMemory());
        cudaTile->init(tile->rowIdx(), tile->colIdx(), tile->height(), tile->width());
        cudaTile->ttl(tile->ttl());
        checkCudaErrors(cudaMemcpyAsync(
            cudaTile->data(),
            reinterpret_cast<const void*>(tile->data()),
            tile->byteSize(),
            cudaMemcpyHostToDevice,
            this->stream()
        ));

        checkCudaErrors(cudaStreamSynchronize(this->stream()));
        this->addResult(cudaTile);

        tile->used();
        if(tile->isMemoryManagerConnected()) tile->returnToMemoryManager();
    }

    std::shared_ptr<hh::AbstractTask<1, Tile, Tile>>
    copy() override {
        return std::make_shared<BlockingHostToDeviceCopyTask>(this->numberThreads());
    }
};

template<typename MatrixType, char Id>
class BlockingDeviceToHostCopyTask: public hh::AbstractCUDATask<
        1,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, Id>>, std::shared_ptr<MatrixTile<MatrixType, Id>>>,
        MatrixTile<MatrixType, Id>
    > {
private:
    using Tile = MatrixTile<MatrixType, Id>;
    using Pair = std::tuple<std::shared_ptr<Tile>, std::shared_ptr<Tile>>;

public:
    explicit BlockingDeviceToHostCopyTask(int32_t threadCount = 2): hh::AbstractCUDATask<1, Pair, Tile>("D2H", threadCount, false, false) {}

    void execute(std::shared_ptr<Pair> pair) override {
        auto cudaTile = std::get<0>(*pair);
        auto tile     = std::get<1>(*pair);
        checkCudaErrors(cudaMemcpyAsync(
            tile->data(),
            cudaTile->data(),
            tile->byteSize(),
            cudaMemcpyDeviceToHost,
            this->stream()
        ));
        checkCudaErrors(cudaStreamSynchronize(this->stream()));
        if(cudaTile->isMemoryManagerConnected()) cudaTile->returnToMemoryManager();
        this->addResult(tile);
    }

    std::shared_ptr<hh::AbstractTask<1, Pair, Tile>>
    copy() override {
        return std::make_shared<BlockingDeviceToHostCopyTask>(this->numberThreads());
    }
};

template<typename MatrixType, char IdA, char IdB, char IdC>
class ProductTask: public hh::AbstractCUDATask<
        1,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>, std::shared_ptr<MatrixTile<MatrixType, IdC>>>,
        MatrixTile<MatrixType, IdC>
    > {
private:
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using TileC   = MatrixTile<MatrixType, IdC>;
    using Triplet = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>, std::shared_ptr<TileC>>;

public:
    explicit ProductTask(int32_t threadCount = 4):
        hh::AbstractCUDATask<1, Triplet, TileC>("Product", threadCount, false) {}

    void initializeCuda() override {
        checkCudaErrors(cublasCreate_v2(&handle_));
        checkCudaErrors(cublasSetStream_v2(handle_, this->stream()));
    }

    void shutdownCuda() override {
        checkCudaErrors(cublasDestroy_v2(handle_));
    }

    void execute(std::shared_ptr<Triplet> triplet) override {
        auto cudaTileA = std::get<std::shared_ptr<TileA>>(*triplet);
        auto cudaTileB = std::get<std::shared_ptr<TileB>>(*triplet);
        auto cudaTileC = std::get<std::shared_ptr<TileC>>(*triplet);
        auto alpha = MatrixType(1), beta = MatrixType(1);

        assert(cudaTileA->rowIdx() == cudaTileC->rowIdx());
        assert(cudaTileB->colIdx() == cudaTileC->colIdx());
        assert(cudaTileA->colIdx() == cudaTileB->rowIdx());
        dotTimer_.start();
        if constexpr(std::is_same_v<MatrixType, float>) {
            checkCudaErrors(cublasSgemm_v2(
                handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                cudaTileA->height(), cudaTileB->width(), cudaTileA->width(), &alpha,
                (float *) cudaTileA->data(), cudaTileA->leadingDimension(),
                (float *) cudaTileB->data(), cudaTileB->leadingDimension(), &beta,
                (float *) cudaTileC->data(), cudaTileC->leadingDimension()
            ));
        }
        else if constexpr(std::is_same_v<MatrixType, double>) {
            checkCudaErrors(cublasDgemm_v2(
                handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                cudaTileA->height(), cudaTileB->width(), cudaTileA->width(), &alpha,
                (double *) cudaTileA->data(), cudaTileA->leadingDimension(),
                (double *) cudaTileB->data(), cudaTileB->leadingDimension(), &beta,
                (double *) cudaTileC->data(), cudaTileC->leadingDimension()
            ));
        }
        else {
            throw std::runtime_error("Datatype not supported for cuda product task.");
        }

        checkCudaErrors(cudaStreamSynchronize(this->stream()));
        dotTimer_.stop();

        cudaTileA->used();
        if(cudaTileA->isMemoryManagerConnected()) {
            cudaTileA->returnToMemoryManager();
        }

        cudaTileB->used();
        if(cudaTileB->isMemoryManagerConnected()) {
            cudaTileB->returnToMemoryManager();
        }

        cudaTileC->used();
        this->addResult(cudaTileC);
    }

    [[nodiscard]] std::string extraPrintingInformation() const override {
        return "";
        DotTimer dotTimer;
        for(auto pNode: this->group()) {
            auto pTask = dynamic_cast<ProductTask const *>(pNode);
            dotTimer.merge(pTask->dotTimer_);
        }

        auto suffix = dotTimer.format();

        auto min = std::to_string(dotTimer.min());
        auto avg = std::to_string(dotTimer.avg());
        auto max = std::to_string(dotTimer.max());

        return "#Gemm received: " + std::to_string(dotTimer.count()) + "\\n"
            "Execution Time Per Gemm:\\n"
            "Min: " + min.substr(0, min.find('.', 0)+4) + suffix + "\\n"
            "Avg: " + avg.substr(0, avg.find('.', 0)+4) + suffix + "\\n"
            "Max: " + max.substr(0, max.find('.', 0)+4) + suffix + "\\n"
            "Total time spent on gemms: " + std::to_string(dotTimer.totalTime()) + suffix
            ;
    }

    std::shared_ptr<hh::AbstractTask<1, Triplet, TileC>>
    copy() override {
        return std::make_shared<ProductTask>(this->numberThreads());
    }
private:
    cublasHandle_t handle_   {};
    DotTimer       dotTimer_ {};
};

#endif //HH3_MATMUL_TASKS

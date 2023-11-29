#ifndef HH3_MATMUL_TASKS
#define HH3_MATMUL_TASKS

#include "common_tasks.h"
#include "data.h"
#include "utility.h"

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
    explicit GpuJobGeneratorTask(const size_t gp, const size_t gq, const int64_t windowHeight, const int64_t windowWidth, std::shared_ptr<GraphFilterState> &graphFilterState):
        hh::AbstractTask<1, Triplet, TileA, TileB, DbRequest<IdA>, DbRequest<IdB>, Job>("GpuJobGeneratorTask", 1, false),
        gp0_(gp), gq0_(gq), windowHeight_(windowHeight), windowWidth_(windowWidth), graphFilterState_(graphFilterState) {}

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
        auto priorityQueue = getPrioritySequence(matrixA, matrixB, rowIndices, colIndices);

#ifndef NDEBUG
        auto MT = matrixC->matrixNumRowTiles(), NT = matrixC->matrixNumColTiles();
        auto debug = std::vector<std::vector<int32_t>>(MT, std::vector<int32_t>(NT, -1));
        auto print = [&debug, MT, NT]() {
            std::string buffer((MT*NT+8)*8, '\0');
            int32_t len = 0;

            // row header
            len += sprintf(&buffer[len], "[Node %ld] GPU workload distribution (the unit's place represents the GPU-Id, and the rest of the digits represents the Node-Id):\n", getNodeId());
            len += sprintf(&buffer[len], "      ");
            for(int j = 0; j < NT; ++j) len += sprintf(&buffer[len], "%3d, ", j);
            len -= 2;
            len += sprintf(&buffer[len], "\n    ");
            for(int j = 0; j < NT; ++j) len += sprintf(&buffer[len], "----");
            len += sprintf(&buffer[len], "\n");

            for(int i = 0; i < MT; ++i) {
                len += sprintf(&buffer[len], "%3d | ", i);
                for(int j = 0; j < NT; ++j) {
                    len += sprintf(&buffer[len], "%3d, ", debug[i][j]);
                }
                len -= 2;
                len += sprintf(&buffer[len], "\n");
            }
            printf("%s\n", buffer.c_str());
            fflush(stdout);
        };
#endif

        std::vector<std::shared_ptr<Job>> jobs(gp0_*gq0_, nullptr);
        for(size_t i = 0; i < rowIndices.size(); i+= (gp0_*windowHeight_)) {
            for(size_t j = 0; j < colIndices.size(); j+= (gq0_*windowWidth_)) {
                rowIndicesSet.clear();
                colIndicesSet.clear();
                for(size_t gp = 0; gp < gp0_; ++gp) {
                    for(size_t gq = 0; gq < gq0_; ++gq) {
                        auto job   = std::make_shared<Job>();
                        auto token = std::static_pointer_cast<GpuToken>(this->getManagedMemory());
                        job->token(token);
                        token->id       = gp*gq0_ + gq;
                        jobs[token->id] = job;
                        graphFilterState_->rowIndices[token->id].clear();
                        graphFilterState_->colIndices[token->id].clear();
                        for(size_t ii = i+gp; (job->height < windowHeight_) and (ii < rowIndices.size()); ii+=gp0_) {
                            job->height++;
                            job->width = 0;
                            for(size_t jj = j+gq; (job->width < windowWidth_) and (jj < colIndices.size()); jj+=gq0_) {
                                rowIndicesSet.insert(rowIndices[ii]);
                                colIndicesSet.insert(colIndices[jj]);
                                graphFilterState_->rowIndices[token->id].insert(rowIndices[ii]);
                                graphFilterState_->colIndices[token->id].insert(colIndices[jj]);
#ifndef NDEBUG
                                debug[rowIndices[ii]][colIndices[jj]] = getNodeId()*10 + int32_t(token->id);
#endif
                                job->addTileC(matrixC->tile(rowIndices[ii], colIndices[jj]));
                                job->width++;
                            }
                        }

                        if(job->tilesFromMatrixC().empty()) {
                            job->processed();
                            job->finished();
                            continue;
                        };
                        this->addResult(job);
                    }
                }

#ifndef NDEBUG
                if(1 <= args.v) print();
#endif
                // wait for all the gpu jobs to be processed before start sending tiles from matrices A and B
                for(auto &job: jobs) {
                    while(!job->hasBeenProcessed()) continue;
                }

                for(auto kt: priorityQueue) {
                    auto reqA = std::make_shared<DbRequest<IdA>>(matrixA->owner(*rowIndicesSet.begin(), kt));
                    for(auto rowIdx: rowIndicesSet) {
                        reqA->addIndex(rowIdx, kt);
                    }
                    this->addResult(reqA);

                    auto reqB = std::make_shared<DbRequest<IdB>>(matrixB->owner(kt, *colIndicesSet.begin()));
                    for(auto colIdx: colIndicesSet) {
                        reqB->addIndex(kt, colIdx);
                    }
                    this->addResult(reqB);
                }

                if(2 <= args.v) fprintf(stderr, "[Host %s][Started!]\n", getHostName().c_str());
                this->taskBarrier();
                if(2 <= args.v) {
                    fprintf(stderr, "[Host %s][Ended!]\n", getHostName().c_str());
                    auto pCore = this->coreTask()->belongingGraph();
                    hh::DotPrinter printer(
                        std::filesystem::absolute(args.P+"/tmp_"+std::to_string(getNodeId())+".dot"),
                        hh::ColorScheme::EXECUTION,
                        hh::StructureOptions::QUEUE,
                        hh::InputOptions::GATHERED,
                        hh::DebugOptions::ALL,
                        pCore,
                        std::move(std::make_unique<hh::JetColor>()),
                        false
                    );
                    pCore->visit(&printer);
                }
            }
        }

        this->addResult(std::make_shared<DbRequest<IdA>>(true));
        this->addResult(std::make_shared<DbRequest<IdB>>(true));

        this->taskBarrier();
        this->addResult(std::make_shared<Job>(true));
    }

private:
    std::vector<int64_t> getPrioritySequence(std::shared_ptr<MatrixA> &matrixA, std::shared_ptr<MatrixB> &matrixB, const std::vector<int64_t> &rowIndices, const std::vector<int64_t> &colIndices) {
        struct JobK {
            int64_t index    = 0;
            int64_t priority = 0;

            bool operator<(const JobK &other) {
                return priority > other.priority;
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
    size_t                            gp0_              = 0;
    size_t                            gq0_              = 0;
    int64_t                           windowHeight_     = 0;
    int64_t                           windowWidth_      = 0;
    std::shared_ptr<GraphFilterState> graphFilterState_ = nullptr;
};

template<typename MatrixType, char IdA, char IdB, char IdC>
class GpuJobGeneratorTask2: public hh::AbstractTask<
        1,
        std::tuple<std::shared_ptr<MatrixContainer<MatrixType, IdA>>, std::shared_ptr<MatrixContainer<MatrixType, IdB>>, std::shared_ptr<MatrixContainer<MatrixType, IdC>>>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        DwBatchRequest<IdA>,
        DwBatchRequest<IdB>,
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
    explicit GpuJobGeneratorTask2(const size_t gp, const size_t gq, const int64_t windowHeight, const int64_t windowWidth, std::shared_ptr<GraphFilterState> &graphFilterState):
        hh::AbstractTask<1, Triplet, TileA, TileB, DwBatchRequest<IdA>, DwBatchRequest<IdB>, Job>("GpuJobGeneratorTask", 1, false),
        gp0_(gp), gq0_(gq), windowHeight_(windowHeight), windowWidth_(windowWidth), graphFilterState_(graphFilterState) {}

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
        auto priorityQueue = getPrioritySequence(matrixA, matrixB, rowIndices, colIndices);

#ifndef NDEBUG
        auto MT = matrixC->matrixNumRowTiles(), NT = matrixC->matrixNumColTiles();
        auto debug = std::vector<std::vector<int32_t>>(MT, std::vector<int32_t>(NT, -1));
        auto print = [&debug, MT, NT]() {
            std::string buffer((MT*NT+8)*8, '\0');
            int32_t len = 0;

            // row header
            len += sprintf(&buffer[len], "[Node %ld] GPU workload distribution (the unit's place represents the GPU-Id, and the rest of the digits represents the Node-Id):\n", getNodeId());
            len += sprintf(&buffer[len], "      ");
            for(int j = 0; j < NT; ++j) len += sprintf(&buffer[len], "%3d, ", j);
            len -= 2;
            len += sprintf(&buffer[len], "\n    ");
            for(int j = 0; j < NT; ++j) len += sprintf(&buffer[len], "----");
            len += sprintf(&buffer[len], "\n");

            for(int i = 0; i < MT; ++i) {
                len += sprintf(&buffer[len], "%3d | ", i);
                for(int j = 0; j < NT; ++j) {
                    len += sprintf(&buffer[len], "%3d, ", debug[i][j]);
                }
                len -= 2;
                len += sprintf(&buffer[len], "\n");
            }
            printf("%s\n", buffer.c_str());
            fflush(stdout);
        };
#endif
        int32_t batchJobCount = ((rowIndices.size()+gp0_*windowHeight_-1)/(gp0_*windowHeight_))*((colIndices.size()+gq0_*windowWidth_-1)/(gq0_*windowWidth_));
        if(1 <= args.v) printf("[Node %ld][START][%zu x %zu][batchJobCount %d]\n", getNodeId(), rowIndices.size(), colIndices.size(), batchJobCount);
        std::vector<std::shared_ptr<Job>> jobs(gp0_*gq0_, nullptr);
        for(size_t i = 0; i < rowIndices.size(); i+= (gp0_*windowHeight_)) {
            for(size_t j = 0; j < colIndices.size(); j+= (gq0_*windowWidth_)) {
                rowIndicesSet.clear();
                colIndicesSet.clear();
                for(size_t gp = 0; gp < gp0_; ++gp) {
                    for(size_t gq = 0; gq < gq0_; ++gq) {
                        auto job   = std::make_shared<Job>();
                        auto token = std::static_pointer_cast<GpuToken>(this->getManagedMemory());
                        job->token(token);
                        token->id       = gp*gq0_ + gq;
                        jobs[token->id] = job;
                        graphFilterState_->rowIndices[token->id].clear();
                        graphFilterState_->colIndices[token->id].clear();
                        for(size_t ii = i+gp; (job->height < windowHeight_) and (ii < rowIndices.size()); ii+=gp0_) {
                            job->height++;
                            job->width = 0;
                            for(size_t jj = j+gq; (job->width < windowWidth_) and (jj < colIndices.size()); jj+=gq0_) {
                                rowIndicesSet.insert(rowIndices[ii]);
                                colIndicesSet.insert(colIndices[jj]);
                                graphFilterState_->rowIndices[token->id].insert(rowIndices[ii]);
                                graphFilterState_->colIndices[token->id].insert(colIndices[jj]);
#ifndef NDEBUG
                                debug[rowIndices[ii]][colIndices[jj]] = getNodeId()*10 + int32_t(token->id);
#endif
                                job->addTileC(matrixC->tile(rowIndices[ii], colIndices[jj]));
                                job->width++;
                            }
                        }

                        if(job->tilesFromMatrixC().empty()) {
                            job->processed();
                            job->finished();
                            continue;
                        };
                        this->addResult(job);
                    }
                }

#ifndef NDEBUG
                if(1 <= args.v) print();
#endif
                // wait for all the gpu jobs to be processed before start sending tiles from matrices A and B
                for(auto &job: jobs) {
                    while(!job->hasBeenProcessed()) continue;
                }
                batchJobCount--;

                auto reqA = std::make_shared<DwBatchRequest<IdA>>(batchJobCount == 0);
                auto reqB = std::make_shared<DwBatchRequest<IdB>>(batchJobCount == 0);
                for(auto kt: priorityQueue) {
                    for(auto rowIdx: rowIndicesSet) {
                        reqA->addIndex(rowIdx, kt);
                    }

                    for(auto colIdx: colIndicesSet) {
                        reqB->addIndex(kt, colIdx);
                    }
                }
                this->addResult(reqA);
                this->addResult(reqB);

                this->taskBarrier();
            }
        }

        if(1 <= args.v) printf("[Node %ld][START][%zu x %zu][batchJobCount %d]\n", getNodeId(), rowIndices.size(), colIndices.size(), batchJobCount);

        this->taskBarrier();
        this->addResult(std::make_shared<Job>(true));
    }

private:
    std::vector<int64_t> getPrioritySequence(std::shared_ptr<MatrixA> &matrixA, std::shared_ptr<MatrixB> &matrixB, const std::vector<int64_t> &rowIndices, const std::vector<int64_t> &colIndices) {
        struct JobK {
            int64_t index    = 0;
            int64_t priority = 0;

            bool operator<(const JobK &other) {
                return priority > other.priority;
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
    size_t                            gp0_              = 0;
    size_t                            gq0_              = 0;
    int64_t                           windowHeight_     = 0;
    int64_t                           windowWidth_      = 0;
    std::shared_ptr<GraphFilterState> graphFilterState_ = nullptr;
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
        hh::AbstractTask<2, TileA, TileB, TileA, TileB>("TileSorter", 1, false), jobPerBatch_(gp*gq) {}

    void execute(std::shared_ptr<TileA> tileA) override {
        tileA->ttl(jobPerBatch_);
        this->addResult(tileA);
    }

    void execute(std::shared_ptr<TileB> tileB) override {
        tileB->ttl(jobPerBatch_);
        this->addResult(tileB);
    }
private:
    int64_t jobPerBatch_ = 0;
};

template<typename MatrixType, char IdA, char IdB, char IdC>
class GpuJobSchedulerTask: public hh::AbstractTask<
        4,
        GpuJob<MatrixType, IdA, IdB, IdC>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        MatrixTile<MatrixType, IdC>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>, std::shared_ptr<MatrixTile<MatrixType, IdC>>>,
        MatrixTile<MatrixType, IdC>,
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
        hh::AbstractTask<4, Job, TileA, TileB, TileC, Triplet, TileC, Pair>("GpuJobSchedulerTask", 1, false),
        MT_(MT), KT_(KT), NT_(NT) {
        ttlKt_.resize(KT);
        gridCudaTileA_.resize(KT);
        gridCudaTileB_.resize(KT);
        gridCudaTileC_.resize(MT, std::vector<std::shared_ptr<TileC>>(NT, nullptr));
    }

    void execute(std::shared_ptr<Job> job) override {
        job->processed();
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
            gridCudaTileA_[kt].reserve(MT_);
            gridCudaTileB_[kt].clear();
            gridCudaTileB_[kt].reserve(NT_);
        }

        for(int64_t row = 0; row < MT_; ++row) {
            for(int64_t col = 0; col < NT_; ++col) {
                gridCudaTileC_[row][col] = nullptr;
            }
        }

        for(auto &tileC: job->tilesFromMatrixC()) {
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

    void execute(std::shared_ptr<TileA> cudaTileA) override {
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

    void execute(std::shared_ptr<TileB> cudaTileB) override {
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

    std::shared_ptr<hh::AbstractTask<4, Job, TileA, TileB, TileC, Triplet, TileC, Pair>>
    copy() override {
        return std::make_shared<GpuJobSchedulerTask<MatrixType, IdA, IdB, IdC>>(this->MT_, this->KT_, this->NT_);
    }

    [[nodiscard]] std::string extraPrintingInformation() const override {
        auto dotTimer = this->dotTimer_;
        auto suffix = dotTimer.format();

        auto min = std::to_string(dotTimer.min());
        auto avg = std::to_string(dotTimer.avg());
        auto max = std::to_string(dotTimer.max());
        return "GPU " + std::to_string(this->deviceId()) + "\\n"
            "#Jobs received: " + std::to_string(dotTimer.count()) + "\\n"
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
        gridCudaTileA_[kt].reserve(MT_);
        gridCudaTileB_[kt].clear();
        gridCudaTileB_[kt].reserve(NT_);
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
    explicit BlockingDeviceToHostCopyTask(int32_t threadCount = 1): hh::AbstractCUDATask<1, Pair, Tile>("D2H", threadCount, false, false) {}

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

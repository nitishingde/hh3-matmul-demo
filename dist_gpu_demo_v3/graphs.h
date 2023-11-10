#ifndef HH3_MATMUL_GRAPHS_H
#define HH3_MATMUL_GRAPHS_H

#include "data.h"
#include "states.h"
#include "tasks.h"

template<typename MatrixType, char IdA, char IdB, char IdC>
class OuterProductGpuGraph: public hh::Graph<3, GpuJob<MatrixType, IdA, IdB, IdC>, MatrixTile<MatrixType, IdA>, MatrixTile<MatrixType, IdB>, MatrixTile<MatrixType, IdC>> {
private:
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using TileC   = MatrixTile<MatrixType, IdC>;
    using GcTileA = GcMatrixTile<MatrixType, IdA>;
    using GcTileB = GcMatrixTile<MatrixType, IdB>;
    using Job     = GpuJob<MatrixType, IdA, IdB, IdC>;
    using Triplet = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>, std::shared_ptr<TileC>>;
    using Pair    = std::tuple<std::shared_ptr<TileC>, std::shared_ptr<TileC>>;

public:
    explicit OuterProductGpuGraph(const int64_t MT, const int64_t KT, const int64_t NT, const int64_t tileSize, const int64_t windowHeight, const int64_t windowWidth, const int32_t d = 2, const int32_t threadCount = 4):
        hh::Graph<3, Job, TileA, TileB, TileC>() {

        auto h2dTaskA  = std::make_shared<BlockingHostToDeviceCopyTask<MatrixType, IdA>>(std::max(threadCount/2, 1));
        h2dTaskA->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileA, int64_t, MemoryType>>(windowHeight*d, tileSize, MemoryType::CUDA)
        );
        auto h2dTaskB  = std::make_shared<BlockingHostToDeviceCopyTask<MatrixType, IdB>>(std::max(threadCount/2, 1));
        h2dTaskB->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileB, int64_t, MemoryType>>(windowWidth*d, tileSize, MemoryType::CUDA)
        );
        auto h2dTaskC  = std::make_shared<BlockingHostToDeviceCopyTask<MatrixType, IdC>>(std::max(threadCount/2, 1));
        h2dTaskC->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileA, int64_t, MemoryType>>(windowHeight*windowWidth, tileSize, MemoryType::CUDA)
        );
        auto schedTask = std::make_shared<GpuJobSchedulerTask<MatrixType, IdA, IdB, IdC>>(MT, KT, NT);
        auto prodTask  = std::make_shared<ProductTask<MatrixType, IdA, IdB, IdC>>(threadCount);
        auto d2hTaskC  = std::make_shared<BlockingDeviceToHostCopyTask<MatrixType, IdC>>(2);

        this->template input<Job>(schedTask);
        this->template input<TileA>(schedTask);
        this->template input<TileB>(schedTask);

        this->template edge<TileA>(schedTask, h2dTaskA);
        this->template edge<TileA>(h2dTaskA, schedTask);

        this->template edge<TileB>(schedTask, h2dTaskB);
        this->template edge<TileB>(h2dTaskB, schedTask);

        this->template edge<TileC>(schedTask, h2dTaskC);
        this->template edge<TileC>(h2dTaskC, schedTask);

        this->template edge<Triplet>(schedTask, prodTask);
        this->template edge<TileC>(prodTask, schedTask);
        this->template edge<Pair>(schedTask, d2hTaskC);
        this->template output<TileC>(d2hTaskC);
    }
};

template<typename MatrixType, char IdA, char IdB, char IdC>
class OuterProductExecutionPipeline: public hh::AbstractExecutionPipeline<3, GpuJob<MatrixType, IdA, IdB, IdC>, MatrixTile<MatrixType, IdA>, MatrixTile<MatrixType, IdB>, MatrixTile<MatrixType, IdC>> {
private:
    using TileA = MatrixTile<MatrixType, IdA>;
    using TileB = MatrixTile<MatrixType, IdB>;
    using TileC = MatrixTile<MatrixType, IdC>;
    using Job   = GpuJob<MatrixType, IdA, IdB, IdC>;

public:
    explicit OuterProductExecutionPipeline(const std::shared_ptr<hh::Graph<3, Job, TileA, TileB, TileC>> &graph, const std::vector<int32_t> &deviceIds):
        hh::AbstractExecutionPipeline<3, Job, TileA, TileB, TileC>(graph, deviceIds, "GPU Pipeline"), deviceIds_{deviceIds} {}

    bool sendToGraph(std::shared_ptr<Job> &job, size_t const &graphId) override {
        return job->shouldQuit() or job->gpuId() == deviceIds_[graphId];
    }

    bool sendToGraph([[maybe_unused]] std::shared_ptr<TileA> &tileA, [[maybe_unused]] size_t const &graphId) override {
        return true;
    }

    bool sendToGraph([[maybe_unused]] std::shared_ptr<TileB> &tileB, [[maybe_unused]] size_t const &graphId) override {
        return true;
    }

private:
    std::vector<int32_t> deviceIds_ = {};
};

#endif //HH3_MATMUL_GRAPHS_H

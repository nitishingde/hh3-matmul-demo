#ifndef HH3_MATMUL_GRAPHS_H
#define HH3_MATMUL_GRAPHS_H

#include "data.h"
#include "states.h"
#include "tasks.h"

template<typename MatrixType, char IdA, char IdB, char IdC, char IdP>
class OuterProductGpuGraph: public hh::Graph<1, GpuJob<MatrixType, IdA, IdB, IdC>, MatrixTile<MatrixType, IdP>> {
private:
    using TileA = MatrixTile<MatrixType, IdA>;
    using TileB = MatrixTile<MatrixType, IdB>;
    using TileP = MatrixTile<MatrixType, IdP>;
    using Job   = GpuJob<MatrixType, IdA, IdB, IdC>;
    using Pair  = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>>;

public:
    explicit OuterProductGpuGraph(int64_t tileSize, int32_t threadCount = 4):
        hh::Graph<1, GpuJob<MatrixType, IdA, IdB, IdC>, MatrixTile<MatrixType, IdP>>() {

        auto h2dTaskA  = std::make_shared<HostToDeviceCopyTask<MatrixType, IdA>>();
        auto h2dTaskB  = std::make_shared<HostToDeviceCopyTask<MatrixType, IdB>>();
        auto schedTask = std::make_shared<GpuJobSchedulerTask<MatrixType, IdA, IdB, IdC, IdP>>();
        auto prodTask  = std::make_shared<ProductTask<MatrixType, IdA, IdB, IdP>>(threadCount);
        prodTask->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileP, int64_t, MemoryType>>(threadCount, tileSize, MemoryType::CUDA_UNIFIED_MEMORY)
        );

        this->template input<Job>(schedTask);
        this->template edge<TileA>(schedTask, h2dTaskA);
        this->template edge<TileB>(schedTask, h2dTaskB);
        this->template edge<TileA>(h2dTaskA, schedTask);
        this->template edge<TileB>(h2dTaskB, schedTask);
        this->template edge<Pair>(schedTask, prodTask);
        this->template edge<TileP>(prodTask, schedTask);
        this->template output<TileP>(prodTask);
    }
};

template<typename MatrixType, char IdA, char IdB, char IdC, char IdP>
class OuterProductExecutionPipeline: public hh::AbstractExecutionPipeline<1, GpuJob<MatrixType, IdA, IdB, IdC>, MatrixTile<MatrixType, IdP>> {
private:
    using Job   = GpuJob<MatrixType, IdA, IdB, IdC>;
    using TileP = MatrixTile<MatrixType, IdP>;

public:
    explicit OuterProductExecutionPipeline(const std::shared_ptr<hh::Graph<1, Job, TileP>> &graph, const std::vector<int32_t> &deviceIds):
        hh::AbstractExecutionPipeline<1, Job, TileP>(graph, deviceIds, "GPU Pipeline"), deviceIds_{deviceIds} {}

    bool sendToGraph(std::shared_ptr<Job> &job, size_t const &graphId) override {
        return job->shouldQuit() or job->gpuId() == deviceIds_[graphId];
    }

private:
    std::vector<int32_t> deviceIds_ = {};
};

#endif //HH3_MATMUL_GRAPHS_H

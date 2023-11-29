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
            std::make_shared<hh::StaticMemoryManager<TileC, int64_t, MemoryType>>(windowHeight*windowWidth, tileSize, MemoryType::CUDA)
        );
        auto schedTask = std::make_shared<GpuJobSchedulerTask<MatrixType, IdA, IdB, IdC>>(MT, KT, NT);
        auto prodTask  = std::make_shared<ProductTask<MatrixType, IdA, IdB, IdC>>(threadCount);
        auto d2hTaskC  = std::make_shared<BlockingDeviceToHostCopyTask<MatrixType, IdC>>(2);

        this->template input<Job>(schedTask);
        this->template input<TileA>(h2dTaskA);
        this->template input<TileB>(h2dTaskB);

        this->template edge<TileA>(h2dTaskA, schedTask);
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
    explicit OuterProductExecutionPipeline(const std::shared_ptr<hh::Graph<3, Job, TileA, TileB, TileC>> &graph, const std::vector<int32_t> &deviceIds, std::shared_ptr<GraphFilterState> graphFilterState):
        hh::AbstractExecutionPipeline<3, Job, TileA, TileB, TileC>(graph, deviceIds, "GPU Pipeline"),
        deviceIds_{deviceIds}, graphFilterState_(graphFilterState) {}

    bool sendToGraph(std::shared_ptr<Job> &job, size_t const &graphId) override {
        return job->shouldQuit() or job->gpuId() == deviceIds_[graphId];
    }

    bool sendToGraph(std::shared_ptr<TileA> &tileA, size_t const &graphId) override {
        auto isNeeded = graphFilterState_->rowIndices[graphId].contains(tileA->rowIdx());
        if(!isNeeded) tileA->used();
        return isNeeded;
    }

    bool sendToGraph(std::shared_ptr<TileB> &tileB, size_t const &graphId) override {
        auto isNeeded =  graphFilterState_->colIndices[graphId].contains(tileB->colIdx());
        if(!isNeeded) tileB->used();
        return isNeeded;
    }

private:
    std::shared_ptr<GraphFilterState> graphFilterState_ = nullptr;
    std::vector<int32_t>              deviceIds_        = {};
};

#endif //HH3_MATMUL_GRAPHS_H

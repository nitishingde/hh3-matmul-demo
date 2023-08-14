#ifndef HH3_MATMUL_MMD_H
#define HH3_MATMUL_MMD_H

#include "graphs.h"

template<class MatrixType, char IdA, char IdB, char IdC>
class MMD_Strategy {
protected:
    using MatrixA = MatrixContainer<MatrixType, IdA>;
    using MatrixB = MatrixContainer<MatrixType, IdB>;
    using MatrixC = MatrixContainer<MatrixType, IdC>;

public:
    explicit MMD_Strategy() {
        checkCudaErrors(cudaGetDeviceProperties(&cudaDeviceProp_, 0));
    }

    virtual double executeImpl(
        std::shared_ptr<MatrixA> matrixA,
        std::shared_ptr<MatrixB> matrixB,
        std::shared_ptr<MatrixC> matrixC,
        const std::vector<int32_t> &deviceIds,
        MPI_Comm mpiComm,
        std::string dotFile
    ) = 0;

protected:
#ifdef HH_USE_CUDA
    cudaDeviceProp cudaDeviceProp_ = {};
#endif
};

template<class MatrixType, char IdA, char IdB, char IdC>
class MMD_WindowStrategy: public MMD_Strategy<MatrixType, IdA, IdB, IdC> {
private:
    using MatrixA = MMD_Strategy<MatrixType, IdA, IdB, IdC>::MatrixA;
    using MatrixB = MMD_Strategy<MatrixType, IdA, IdB, IdC>::MatrixB;
    using MatrixC = MMD_Strategy<MatrixType, IdA, IdB, IdC>::MatrixC;

public:
    explicit MMD_WindowStrategy() = default;

    MMD_Strategy<MatrixType, IdA, IdB, IdC>& builder(const int64_t windowSize, const int64_t productThreads) {
        windowSize_     = windowSize;
        productThreads_ = productThreads;
        return *this;
    }

    double executeImpl(
        std::shared_ptr<MatrixA> matrixA,
        std::shared_ptr<MatrixB> matrixB,
        std::shared_ptr<MatrixC> matrixC,
        const std::vector<int32_t> &deviceIds,
        MPI_Comm mpiComm,
        std::string dotFile
    ) override {
        constexpr char       IdP        = 'p';
        constexpr MemoryType memoryType = MemoryType::CUDA_UNIFIED_MEMORY;

        using Triplet    = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>;
        using TileA      = MatrixTile<MatrixType, IdA>;
        using TileB      = MatrixTile<MatrixType, IdB>;
        using TileC      = MatrixTile<MatrixType, IdC>;
        using TileP      = MatrixTile<MatrixType, IdP>;
        using Pair       = std::tuple<std::shared_ptr<TileC>, std::shared_ptr<TileP>>;
        using Job        = GpuJob<MatrixType, IdA, IdB, IdC>;

        auto MT = matrixC->matrixNumRowTiles(), KT = matrixA->matrixNumColTiles(), NT = matrixC->matrixNumColTiles();
        auto T  = std::max(std::max(matrixA->tileDim(), matrixB->tileDim()), matrixC->tileDim());

        // Generate graph
        auto graph = hh::Graph<1, Triplet, TileC>("MM");

        auto inputStateManager  = std::make_shared<hh::StateManager<1, Triplet, MatrixA, MatrixB, MatrixC, Triplet>>(
            std::make_shared<InputState<MatrixType, IdA, IdB, IdC>>(),
            "InputStateManager",
            false
        );
        auto jobGenTask         = std::make_shared<GpuJobGeneratorTask<MatrixType, IdA, IdB, IdC>>(windowSize_);
        jobGenTask->connectMemoryManager(std::make_shared<GpuTokenMemoryManager>(deviceIds));
        auto execPipeline       = std::make_shared<OuterProductExecutionPipeline<MatrixType, IdA, IdB, IdC, IdP>>(
            std::make_shared<OuterProductGpuGraph<MatrixType, IdA, IdB, IdC, IdP>>(T, productThreads_), deviceIds
        );
        auto compStateManager   = std::make_shared<OuterProductComputationStateManager<MatrixType, IdA, IdB, IdC, IdP>>(
            std::make_shared<OuterProductComputationState<MatrixType, IdA, IdB, IdC, IdP>>(MT, KT, NT)
        );
        auto accTask            = std::make_shared<AccumulateTask<MatrixType, IdC, IdP>>(productThreads_);
        auto dwTaskA            = std::make_shared<MatrixWarehouseTask<MatrixType, IdA>>();
        dwTaskA->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileA, int64_t, MemoryType>>(MT, T, memoryType)
        );
        auto dwTaskB            = std::make_shared<MatrixWarehouseTask<MatrixType, IdB>>();
        dwTaskB->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileB, int64_t, MemoryType>>(NT, T, memoryType)
        );

        graph.template input<Triplet>(inputStateManager);
        graph.template edge<MatrixA>(inputStateManager, dwTaskA);
        graph.template edge<MatrixB>(inputStateManager, dwTaskB);
        graph.template edge<MatrixC>(inputStateManager, compStateManager);
        graph.template edge<Triplet>(inputStateManager, jobGenTask);
        graph.template edge<DbRequest<IdA>>(jobGenTask, dwTaskA);
        graph.template edge<DbRequest<IdB>>(jobGenTask, dwTaskB);
        graph.template edge<TileA>(dwTaskA, jobGenTask);
        graph.template edge<TileB>(dwTaskB, jobGenTask);
        graph.template edge<Job>(jobGenTask, execPipeline);
        graph.template edge<TileP>(execPipeline, compStateManager);
        graph.template edge<Pair>(compStateManager, accTask);
        graph.template edge<TileC>(accTask, compStateManager);
        graph.template output<TileC>(compStateManager);
        graph.executeGraph();

        MPI_Barrier(mpiComm);
        graph.pushData(std::make_shared<Triplet>(std::make_tuple(matrixA, matrixB, matrixC)));
        graph.finishPushingData();

        graph.waitForTermination();
        graph.createDotFile(
            dotFile,
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::ALL,
            hh::DebugOptions::ALL,
            std::make_unique<hh::JetColor>(),
            false
        );

        double time = double((
                graph.core()->executionDuration() == std::chrono::nanoseconds::zero()?
                    std::chrono::system_clock::now() - graph.core()->startExecutionTimeStamp():
                    graph.core()->executionDuration()
            ).count())/1.e9;
        double maxTime = 0;
        checkMpiErrors(MPI_Reduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiComm));
        return maxTime;
    }

private:
    int64_t windowSize_     = 0;
    int64_t productThreads_ = 0;
};

#endif //HH3_MATMUL_MMD_H

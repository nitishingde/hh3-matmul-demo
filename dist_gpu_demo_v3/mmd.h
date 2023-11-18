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

    MMD_Strategy<MatrixType, IdA, IdB, IdC>& builder(const int64_t gp, const int64_t gq, const int64_t windowHeight, const int64_t windowWidth, const int64_t depth, const int64_t productThreads) {
        gp_             = gp;
        gq_             = gq;
        windowHeight_   = windowHeight;
        windowWidth_    = windowWidth;
        depth_          = depth;
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
        constexpr MemoryType memoryType = MemoryType::HOST;

        using Triplet    = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>;
        using TileA      = MatrixTile<MatrixType, IdA>;
        using TileB      = MatrixTile<MatrixType, IdB>;
        using TileC      = MatrixTile<MatrixType, IdC>;
        using Job        = GpuJob<MatrixType, IdA, IdB, IdC>;

        auto MT     = matrixC->matrixNumRowTiles();
        auto KT     = matrixA->matrixNumColTiles();
        auto NT     = matrixC->matrixNumColTiles();
        auto T      = std::max(std::max(matrixA->tileDim(), matrixB->tileDim()), matrixC->tileDim());
        auto [P, Q] = getGridDim();
        auto G      = deviceIds.size();

        auto graphFilterState = std::make_shared<GraphFilterState>(deviceIds);

        // Generate graph
        auto graph = hh::Graph<1, Triplet, TileC>("MM");

        auto inputStateManager  = std::make_shared<hh::StateManager<1, Triplet, MatrixA, MatrixB, MatrixC, Triplet>>(
            std::make_shared<InputState<MatrixType, IdA, IdB, IdC>>(),
            "InputStateManager",
            false
        );
        auto jobGenTask         = std::make_shared<GpuJobGeneratorTask<MatrixType, IdA, IdB, IdC>>(gp_, gq_, windowHeight_, windowWidth_, graphFilterState);
        jobGenTask->connectMemoryManager(std::make_shared<GpuTokenMemoryManager>(deviceIds));
        auto tileSorterTask     = std::make_shared<TileSorterTask<MatrixType, IdA, IdB>>(gp_, gq_);
        auto execPipeline       = std::make_shared<OuterProductExecutionPipeline<MatrixType, IdA, IdB, IdC>>(
            std::make_shared<OuterProductGpuGraph<MatrixType, IdA, IdB, IdC>>(MT, KT, NT, T, windowHeight_, windowWidth_, depth_, productThreads_),
            deviceIds,
            graphFilterState
        );
        auto dwTaskA            = std::make_shared<MatrixWarehouseTask<MatrixType, IdA>>();
        dwTaskA->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileA, int64_t, MemoryType>>(((MT+P-1)/P)*(G*4), T, memoryType)
        );
        auto dwTaskB            = std::make_shared<MatrixWarehouseTask<MatrixType, IdB>>();
        dwTaskB->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileB, int64_t, MemoryType>>(((NT+Q-1)/Q)*(G*4), T, memoryType)
        );

        graph.template input<Triplet>(inputStateManager);
        graph.template edge<MatrixA>(inputStateManager, dwTaskA);
        graph.template edge<MatrixB>(inputStateManager, dwTaskB);
        graph.template edge<Triplet>(inputStateManager, jobGenTask);
        graph.template edge<DbRequest<IdA>>(jobGenTask, dwTaskA);
        graph.template edge<DbRequest<IdB>>(jobGenTask, dwTaskB);
        graph.template edge<TileA>(dwTaskA, tileSorterTask);
        graph.template edge<TileB>(dwTaskB, tileSorterTask);
        graph.template edge<TileA>(tileSorterTask, execPipeline);
        graph.template edge<TileB>(tileSorterTask, execPipeline);
        graph.template edge<Job>(jobGenTask, execPipeline);
        graph.template output<TileC>(execPipeline);
        graph.executeGraph();

        MPI_Barrier(mpiComm);
        graph.pushData(std::make_shared<Triplet>(std::make_tuple(matrixA, matrixB, matrixC)));
        graph.finishPushingData();

#ifndef NDEBUG
        std::atomic_bool quit = false;
        auto dotGraphDaemon = std::thread([&graph, &dotFile, &quit]() {
            using namespace std::chrono_literals;
            while(!quit.load()) {
                graph.createDotFile(
                    dotFile,
                    hh::ColorScheme::EXECUTION,
                    hh::StructureOptions::QUEUE,
                    hh::InputOptions::SEPARATED,
                    hh::DebugOptions::ALL,
                    std::make_unique<hh::JetColor>(),
                    false
                );
                std::this_thread::sleep_for(4ms);
            }
        });
#endif

        graph.waitForTermination();

#if NDEBUG
        graph.createDotFile(
            dotFile,
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::QUEUE,
            hh::InputOptions::GATHERED,
            hh::DebugOptions::NONE,
            std::make_unique<hh::JetColor>(),
            false
        );
#else
        quit.store(true);
        dotGraphDaemon.join();
        graph.createDotFile(
            dotFile,
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::QUEUE,
            hh::InputOptions::GATHERED,
            hh::DebugOptions::NONE,
            std::make_unique<hh::JetColor>(),
            false
        );
#endif

        double time = double((
                graph.core()->dequeueExecDuration() == std::chrono::nanoseconds::zero()?
                    std::chrono::system_clock::now() - graph.core()->startExecutionTimeStamp():
                    graph.core()->dequeueExecDuration()
            ).count())/1.e9;
        double maxTime = 0;
        checkMpiErrors(MPI_Reduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiComm));
        return maxTime;
    }

private:
    int64_t windowHeight_   = 0;
    int64_t windowWidth_    = 0;
    int64_t gp_             = 0;
    int64_t gq_             = 0;
    int64_t depth_          = 0;
    int64_t productThreads_ = 0;
};

#endif //HH3_MATMUL_MMD_H

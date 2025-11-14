#ifndef HH3_MATMUL_MMD_H
#define HH3_MATMUL_MMD_H

#include <openblas/cblas.h>

#include "common_data.h"
#include "tasks.h"

template<class MatrixType, char IdA, char IdB, char IdC>
class MMD_Strategy {
protected:
    using MatrixA = MatrixContainer<MatrixType, IdA>;
    using MatrixB = MatrixContainer<MatrixType, IdB>;
    using MatrixC = MatrixContainer<MatrixType, IdC>;

public:
    explicit MMD_Strategy() = default;
    virtual ~MMD_Strategy() = default;

    virtual double executeImpl(
        std::shared_ptr<MatrixA> matrixA,
        std::shared_ptr<MatrixB> matrixB,
        std::shared_ptr<MatrixC> matrixC,
        MPI_Comm mpiComm,
        const std::string &dotFile
    ) = 0;
};

template<class MatrixType, char IdA, char IdB, char IdC>
class MMD_Simd1 final: public MMD_Strategy<MatrixType, IdA, IdB, IdC> {
public:
    using base = MMD_Strategy<MatrixType, IdA, IdB, IdC>;
    using MatrixA = base::MatrixA;
    using MatrixB = base::MatrixB;
    using MatrixC = base::MatrixC;

    explicit MMD_Simd1() = default;

    MMD_Strategy<MatrixType, IdA, IdB, IdC>& builder(const int64_t productThreads, const int64_t lookAhead = 1) {
        lookAhead_      = lookAhead;
        productThreads_ = productThreads;

        return *this;
    }

    double executeImpl(
        std::shared_ptr<MatrixA> matrixA,
        std::shared_ptr<MatrixB> matrixB,
        std::shared_ptr<MatrixC> matrixC,
        const MPI_Comm mpiComm,
        const std::string &dotFile
    ) override {
        using TileA         = MatrixTile<MatrixType, IdA>;
        using TileB         = MatrixTile<MatrixType, IdB>;
        using TileC         = MatrixTile<MatrixType, IdC>;
        using TileTriplet   = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>, std::shared_ptr<TileC>>;
        using MatrixTriplet = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>;

        constexpr MemoryType memoryType = MemoryType::HOST;

        const auto MT                   = matrixC->matrixNumRowTiles();
        const auto KT                   = matrixA->matrixNumColTiles();
        const auto NT                   = matrixC->matrixNumColTiles();
        const auto T                    = std::max(std::max(matrixA->tileDim(), matrixB->tileDim()), matrixC->tileDim());
        const auto [pNodeId, qNodeId]   = getGridNodeId();
        const auto [pNodeDim, qNodeDim] = getGridDim();
        const auto jobWidth             = (NT-qNodeId + qNodeDim-1)/qNodeDim;
        const auto jobHeight            = (MT-pNodeId + pNodeDim-1)/pNodeDim;

        auto jobGenerator = std::make_shared<JobGenerator<MatrixType, IdA, IdB, IdC>>();

        auto commTaskA    = std::make_shared<MatrixCommTask<MatrixType, IdA>>("CommA", matrixA, jobHeight);
        commTaskA->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileA, int64_t, MemoryType>>(jobHeight*lookAhead_, T, memoryType)
        );
        auto commTaskB    = std::make_shared<MatrixCommTask<MatrixType, IdB>>("CommB", matrixB, jobWidth);
        commTaskB->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileB, int64_t, MemoryType>>(jobWidth*lookAhead_, T, memoryType)
        );
        auto jobScheduler = std::make_shared<JobScheduler<MatrixType, IdA, IdB, IdC>>("JobScheduler");
        auto productTask  = std::make_shared<hh::LambdaTask<1, TileTriplet, TileTriplet>>("Product", productThreads_, false);
        productTask->template setLambda<TileTriplet>([](const std::shared_ptr<TileTriplet> &triplet, auto self) {
            auto &[tileA, tileB, tileC] = *triplet;
            constexpr MatrixType alpha = 1;
            constexpr MatrixType beta  = 1;
            if constexpr(std::is_same_v<MatrixType, float>) {
                cblas_sgemm(
                    CblasColMajor, CblasNoTrans, CblasNoTrans,
                    tileC->height(), tileC->width(), tileA->width(),
                    alpha,
                    static_cast<float*>(tileA->data()), tileA->leadingDimension(),
                    static_cast<float*>(tileB->data()), tileB->leadingDimension(),
                    beta,
                    static_cast<float*>(tileC->data()), tileC->leadingDimension()
                );
            }
            else if constexpr(std::is_same_v<MatrixType, double>) {
                cblas_dgemm(
                    CblasColMajor, CblasNoTrans, CblasNoTrans,
                    tileC->height(), tileC->width(), tileA->width(),
                    alpha,
                    static_cast<double*>(tileA->data()), tileA->leadingDimension(),
                    static_cast<double*>(tileB->data()), tileB->leadingDimension(),
                    beta,
                    static_cast<double*>(tileC->data()), tileC->leadingDimension()
                );
            }

            self.addResult(triplet);
        });

        auto graph = hh::Graph<4, MatrixA, MatrixB, MatrixC, MatrixTriplet, TileC>("SIMD");

        // job generator
        graph.template input<MatrixTriplet>(jobGenerator);
        graph.template edge<CommRequestList<IdA>>(jobGenerator, commTaskA);
        graph.template edge<CommRequestList<IdB>>(jobGenerator, commTaskB);
        graph.template edge<CommSendList<IdA>>(jobGenerator, commTaskA);
        graph.template edge<CommSendList<IdB>>(jobGenerator, commTaskB);

        // job scheduler
        graph.template input<MatrixTriplet>(jobScheduler);
        graph.template edge<TileA>(commTaskA, jobScheduler);
        graph.template edge<TileB>(commTaskB, jobScheduler);
        graph.template edge<TileTriplet>(jobScheduler, productTask);
        graph.template edge<TileTriplet>(productTask, jobScheduler);
        graph.template output<TileC>(jobScheduler);

        graph.executeGraph();

        graph.pushData(std::make_shared<MatrixTriplet>(std::make_tuple(matrixA, matrixB, matrixC)));
        graph.finishPushingData();

        graph.waitForTermination();

        graph.createDotFile(
            dotFile,
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::QUEUE,
            hh::InputOptions::GATHERED,
            hh::DebugOptions::NONE,
            std::make_unique<hh::JetColor>(),
            false
        );

        const double time = static_cast<double>((graph.core()->dequeueExecDuration() == std::chrono::nanoseconds::zero()?
            std::chrono::system_clock::now() - graph.core()->startExecutionTimeStamp():
            graph.core()->dequeueExecDuration()
        ).count())/1.e9;
        double maxTime = 0;
        checkMpiErrors(MPI_Reduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, mpiComm));

        return maxTime;
    }

private:
    int64_t lookAhead_      = 1;
    int64_t productThreads_ = 4;
};

template<class MatrixType, char IdA, char IdB, char IdC>
class MMD_Simd2 final: public MMD_Strategy<MatrixType, IdA, IdB, IdC> {
public:
    using base    = MMD_Strategy<MatrixType, IdA, IdB, IdC>;
    using MatrixA = base::MatrixA;
    using MatrixB = base::MatrixB;
    using MatrixC = base::MatrixC;

    explicit MMD_Simd2(const MPI_Comm groupCommA, const MPI_Comm groupCommB):
        groupCommA_(groupCommA), groupCommB_(groupCommB) {}

    auto& builder(const int64_t productThreads, const int64_t lookAhead = 1) {
        productThreads_ = productThreads;
        lookAhead_      = lookAhead;

        return *this;
    }

    double executeImpl(
        std::shared_ptr<MatrixA> matrixA,
        std::shared_ptr<MatrixB> matrixB,
        std::shared_ptr<MatrixC> matrixC,
        const MPI_Comm gridComm,
        const std::string &dotFile
    ) override {
        using TileA         = MatrixTile<MatrixType, IdA>;
        using TileB         = MatrixTile<MatrixType, IdB>;
        using TileC         = MatrixTile<MatrixType, IdC>;
        using TileTriplet   = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>, std::shared_ptr<TileC>>;
        using MatrixTriplet = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>;
        using MatrixDuplet  = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>>;

        auto MT = matrixC->matrixNumRowTiles();
        auto KT = matrixA->matrixNumColTiles();
        auto NT = matrixC->matrixNumColTiles();
        auto T = std::max(std::max(matrixA->tileDim(), matrixB->tileDim()), matrixC->tileDim());

        auto jobGenerator = std::make_shared<JobGenerator<MatrixType, IdA, IdB, IdC>>();

        auto [p0, q0]     = getGridNodeId();
        auto [pDim, qDim] = getGridDim();

        const auto limitA = ((MT - p0 + pDim - 1)/pDim)*lookAhead_;
        const auto limitB = ((NT - q0 + qDim - 1)/qDim)*lookAhead_;

        auto commTask     = std::make_shared<BroadcastTask<MatrixType, IdA, IdB>>("Comm", gridComm, groupCommA_, groupCommB_, limitA, limitB, T);
        auto jobScheduler = std::make_shared<JobScheduler<MatrixType, IdA, IdB, IdC>>("JobScheduler");
        auto productTask  = std::make_shared<hh::LambdaTask<1, TileTriplet, TileTriplet>>("Product", 4, false);
        productTask->template setLambda<TileTriplet>([](const std::shared_ptr<TileTriplet> &triplet, auto self) {
            auto &[tileA, tileB, tileC] = *triplet;
            constexpr MatrixType alpha = 1;
            constexpr MatrixType beta  = 1;
            if constexpr(std::is_same_v<MatrixType, float>) {
                cblas_sgemm(
                    CblasColMajor, CblasNoTrans, CblasNoTrans,
                    tileC->height(), tileC->width(), tileA->width(),
                    alpha,
                    static_cast<float*>(tileA->data()), tileA->leadingDimension(),
                    static_cast<float*>(tileB->data()), tileB->leadingDimension(),
                    beta,
                    static_cast<float*>(tileC->data()), tileC->leadingDimension()
                );
            }
            else if constexpr(std::is_same_v<MatrixType, double>) {
                cblas_dgemm(
                    CblasColMajor, CblasNoTrans, CblasNoTrans,
                    tileC->height(), tileC->width(), tileA->width(),
                    alpha,
                    static_cast<double*>(tileA->data()), tileA->leadingDimension(),
                    static_cast<double*>(tileB->data()), tileB->leadingDimension(),
                    beta,
                    static_cast<double*>(tileC->data()), tileC->leadingDimension()
                );
            }

            self.addResult(triplet);
        });

        auto graph = hh::Graph<2, MatrixTriplet, MatrixDuplet, TileC>("SIMD");

        // job scheduler
        graph.template input<MatrixTriplet>(jobScheduler);
        graph.template input<MatrixDuplet>(commTask);
        graph.template edge<TileA>(commTask, jobScheduler);
        graph.template edge<TileB>(commTask, jobScheduler);
        graph.template edge<TileTriplet>(jobScheduler, productTask);
        graph.template edge<TileTriplet>(productTask, jobScheduler);
        graph.template output<TileC>(jobScheduler);

        graph.executeGraph();

        graph.pushData(std::make_shared<MatrixTriplet>(std::make_tuple(matrixA, matrixB, matrixC)));
        graph.pushData(std::make_shared<MatrixDuplet>(std::make_tuple(matrixA, matrixB)));
        graph.finishPushingData();

        graph.waitForTermination();

        graph.createDotFile(
            dotFile,
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::QUEUE,
            hh::InputOptions::GATHERED,
            hh::DebugOptions::NONE,
            std::make_unique<hh::JetColor>(),
            false
        );

        const double time = static_cast<double>((graph.core()->dequeueExecDuration() == std::chrono::nanoseconds::zero()?
            std::chrono::system_clock::now() - graph.core()->startExecutionTimeStamp():
            graph.core()->dequeueExecDuration()
        ).count())/1.e9;
        double maxTime = 0;
        checkMpiErrors(MPI_Reduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, gridComm));

        return maxTime;
    }

private:
    MPI_Comm groupCommA_     = MPI_COMM_NULL;
    MPI_Comm groupCommB_     = MPI_COMM_NULL;
    int64_t  productThreads_ = 12;
    int64_t  lookAhead_      = 1;
};

#endif //HH3_MATMUL_MMD_H
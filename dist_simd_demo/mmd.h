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
    using MatrixA = typename base::MatrixA;
    using MatrixB = typename base::MatrixB;
    using MatrixC = typename base::MatrixC;

    explicit MMD_Simd1() = default;

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

        auto MT     = matrixC->matrixNumRowTiles();
        auto KT     = matrixA->matrixNumColTiles();
        auto NT     = matrixC->matrixNumColTiles();
        auto T      = std::max(std::max(matrixA->tileDim(), matrixB->tileDim()), matrixC->tileDim());

        auto jobGenerator = std::make_shared<JobGenerator<MatrixType, IdA, IdB, IdC>>();

        auto commTaskA    = std::make_shared<MatrixCommTask<MatrixType, IdA>>("CommA", matrixA);
        commTaskA->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileA, int64_t, MemoryType>>(MT, T, memoryType)
        );
        auto commTaskB    = std::make_shared<MatrixCommTask<MatrixType, IdB>>("CommB", matrixB);
        commTaskB->connectMemoryManager(
            std::make_shared<hh::StaticMemoryManager<TileB, int64_t, MemoryType>>(NT, T, memoryType)
        );
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
};

#endif //HH3_MATMUL_MMD_H
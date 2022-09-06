#ifndef HH3_MATMUL_MMD_H
#define HH3_MATMUL_MMD_H

#include "utility.h"
#ifdef HH_USE_CUDA
#include <cublasXt.h>
#else
#include <openblas/cblas.h>
#endif
#include "data/contiguous_sub_matrix_container.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"
#include "execution_pipeline/outer_product_exec_pipeline.h"
#include "graph/outer_product_cuda_graph.h"
#include "state/outer_product_computation_state.h"
#include "state/outer_product_computation_state_manager.h"
#include "state/outer_product_output_state.h"
#include "task/accumulate_task.h"
#include "task/comm_tasks.h"
#include "task/matrix_col_traversal_task.h"
#include "task/matrix_row_traversal_task.h"

template<class MatrixType, char IdA, char IdB, char IdC, Order Ord>
class MMD_Strategy {
private:
    virtual void executeImpl(
            std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
            std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
            std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
            const std::vector<int32_t> &deviceIds,
            std::string dotFile
    ) = 0;
    [[nodiscard]] virtual std::string strategy() const = 0;
    [[nodiscard]] virtual std::string matTypeA() const = 0;
    [[nodiscard]] virtual std::string matTypeB() const = 0;
    [[nodiscard]] virtual std::string matTypeC() const = 0;
    [[nodiscard]] virtual std::string className() const = 0;

public:
    void execute(
            std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
            std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
            std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
            const std::vector<int32_t> &deviceIds
    ) {
#if not NDEBUG
        int32_t isMpiInitialized = false;
        if(auto status = MPI_Initialized(&isMpiInitialized); not isMpiInitialized) {
            std::runtime_error("[MPI] not initialized!\n");
        }
#endif

        auto start = std::chrono::high_resolution_clock::now();
        this->executeImpl(matrixA, matrixB, matrixC, deviceIds, className());
        auto end = std::chrono::high_resolution_clock::now();

        if(getNodeId() != 0) return;

        size_t M = matrixC->matrixHeight(), N = matrixC->matrixWidth(), K = matrixA->matrixWidth();
        double time = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1.e9;
        //https://forums.developer.nvidia.com/t/how-to-compute-gflops-for-gemm-blas/20218/6
        double gflops = (M*N*(2*K+2))/(1.e9*time);
        printf(
            "[ " GREEN("MMD") " ][ Strategy = " GREEN("%-12s") " ][ A = " GREEN("%-9s") " ][ B = " GREEN("%-9s") " ][ C = " GREEN("%-9s") " ][ " CYAN("%9.3f") " gflops ][ " RED("%8.3f") " secs ]\n",
            strategy().c_str(),
            matTypeA().c_str(),
            matTypeB().c_str(),
            matTypeC().c_str(),
            gflops,
            time
        );
        fflush(stdout);
    }
};

template<class MatrixType, char IdA, char IdB, char IdC, Order Ord>
class MMD_VerifyCublas: public MMD_Strategy<MatrixType, IdA, IdB, IdC, Ord> {
private:
    void executeImpl(
        std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
        std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
        std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile
    ) override {

        auto subA = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Col, MatrixType, IdA, Ord>>(matrixA);
        auto subB = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Row, MatrixType, IdB, Ord>>(matrixB);
        auto matC = std::static_pointer_cast<RedundantMatrixContainer<MatrixType, IdC, Ord>>(matrixC);
        size_t m = matC->matrixHeight(), k = subA->subMatrixWidth(), n = matC->matrixWidth();
        size_t tileSize = matC->matrixTileSize();

        cublasXtHandle_t handle;
        checkCudaErrors(cublasXtCreate(&handle));

        checkCudaErrors(cublasXtDeviceSelect(handle, deviceIds.size(), (int*)deviceIds.data()));
        checkCudaErrors(cublasXtSetBlockDim(handle, tileSize));

        MatrixType alpha = 1.0, beta = 1.0;

        if constexpr(std::is_same<MatrixType, double>::value) {
            checkCudaErrors(cublasXtDgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, (double *) (&alpha),
                (double *) subA->data(), subA->leadingDimension(),
                (double *) subB->data(), subB->leadingDimension(),
                (double *) (&beta), (double *) matC->data(), matC->leadingDimension()
            ));
        } else {
            checkCudaErrors(cublasXtSgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, (float *) (&alpha),
                (float *) subA->data(), subA->leadingDimension(),
                (float *) subB->data(), subB->leadingDimension(),
                (float *) (&beta), (float *) matC->data(), matC->leadingDimension()
            ));
        }

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cublasXtDestroy(handle));

        std::vector<MatrixType> tempC((isRootNodeId()? m*n: 0));
        if constexpr(std::is_same_v<MatrixType, double>) {
            MPI_Reduce(matC->data(), tempC.data(), m*n, MPI_DOUBLE, MPI_SUM, 0, matC->mpiComm());
        }
        else if constexpr(std::is_same_v<MatrixType, float>) {
            MPI_Reduce(matC->data(), tempC.data(), m*n, MPI_FLOAT, MPI_SUM, 0, matC->mpiComm());
        }
        else {
            throw std::runtime_error("Type not supported\n");
        }
        std::memcpy(matC->data(), tempC.data(), sizeof(MatrixType)*tempC.size());
    }

    [[nodiscard]] std::string strategy() const override {
        return "CublasXt";
    }

    [[nodiscard]] std::string matTypeA() const override {
        return "Sub";
    }

    [[nodiscard]] std::string matTypeB() const override {
        return "Sub";
    }

    [[nodiscard]] std::string matTypeC() const override {
        return "Redundant";
    }

    [[nodiscard]] std::string className() const override {
        return NAME(MMD_VerifyCublas);
    }
};

template<class MatrixType, char IdA, char IdB, char IdC, Order Ord>
class MMD_MpiOuterProductCyclic2d: public MMD_Strategy<MatrixType, IdA, IdB, IdC, Ord> {
private:
    void executeImpl(
        std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
        std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
        std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile
    ) override {
        constexpr char ProdId = 'p';
        auto subA = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Col, MatrixType, IdA, Ord>>(matrixA);
        auto subB = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Row, MatrixType, IdB, Ord>>(matrixB);
        auto matC = std::static_pointer_cast<Cyclic2dMatrixContainer<MatrixType, IdC, Ord>>(matrixC);

        const uint32_t mTiles   = matrixC->matrixNumRowTiles();
        const uint32_t kTiles   = matrixA->matrixNumColTiles();
        const uint32_t nTiles   = matrixC->matrixNumColTiles();
        const uint32_t tileSize = matrixC->matrixTileSize();

        uint32_t myTiles = mTiles*nTiles;
        myTiles = (myTiles/getNumNodes()) + ((matrixC->nodeId() < (myTiles%matrixC->numNodes()))? 1: 0);//TODO: verify calculations

        // create nodes
        auto localGraph = hh::Graph<3,
            MatrixContainer<MatrixType, IdA, Ord>,
            MatrixContainer<MatrixType, IdB, Ord>,
            MatrixContainer<MatrixType, IdC, Ord>,
            void*
        >("Local Graph "+std::to_string(getNodeId()));

        auto matrixATraversalTask = std::make_shared<MatrixColTraversalTask<MatrixType, IdA, Ord>>();
        auto matrixBTraversalTask = std::make_shared<MatrixRowTraversalTask<MatrixType, IdB, Ord>>();
        auto matrixCTraversalTask = std::make_shared<MatrixRowTraversalTask<MatrixType, IdC, Ord>>();
        auto accumulateTask       = std::make_shared<AccumulateTask<MatrixType, IdC, ProdId, Ord>>(4);
        auto recyclerTask         = std::make_shared<TtlManagedMemoryRecyclerTask>();
        auto senderTask           = std::make_shared<Cyclic2dSenderTask<MatrixType, IdC, Ord>>(8, mTiles*nTiles-myTiles);
        auto receiverTask         = std::make_shared<Cyclic2dReceiverTask<MatrixType, ProdId, Ord>>(myTiles*(getNumNodes()-1));

        auto memoryManager = std::make_shared<hh::StaticMemoryManager<MatrixTile<MatrixType, ProdId, Ord>, uint32_t>>(8, tileSize);
        receiverTask->connectMemoryManager(memoryManager);

        auto cudaGraph = std::make_shared<OuterProductCudaGraph<MatrixType, IdA, IdB, ProdId, Ord>>(mTiles, kTiles, nTiles, tileSize);
        auto execPipeline = std::make_shared<OuterProductExecPipeline<MatrixType, IdA, IdB, ProdId, Ord>>(cudaGraph, deviceIds);

        auto computationState        = std::make_shared<OuterProductComputationState<MatrixType, IdC, ProdId, Ord>>(
            mTiles,
            nTiles,
            mTiles*nTiles*subA->subMatrixNumColTiles() + myTiles*(getNumNodes()-1)
        );
        auto computationStateManager = std::make_shared<OuterProductComputationStateManager<MatrixType, IdC, ProdId, Ord>>(computationState);
        auto outputState             = std::make_shared<OuterProductOutputState<MatrixType, IdC, Ord>>(mTiles, nTiles, subA->subMatrixNumColTiles());//FIXME
        auto outputStateManager      = std::make_shared<hh::StateManager<1,
            MatrixTile<MatrixType, IdC, Ord>,
            MatrixTile<MatrixType, IdC, Ord>
        >>(outputState, "Output State Manager", false);

        // add edges
        localGraph.template input<MatrixContainer<MatrixType, IdA, Ord>>(matrixATraversalTask);
        localGraph.template input<MatrixContainer<MatrixType, IdB, Ord>>(matrixBTraversalTask);
        localGraph.template input<MatrixContainer<MatrixType, IdC, Ord>>(matrixCTraversalTask);
        localGraph.template edge<MatrixTile<MatrixType, IdA, Ord>>(matrixATraversalTask, execPipeline);
        localGraph.template edge<MatrixTile<MatrixType, IdB, Ord>>(matrixBTraversalTask, execPipeline);
        localGraph.template edge<MatrixTile<MatrixType, IdC, Ord>>(matrixCTraversalTask, computationStateManager);
        localGraph.template edge<MatrixTile<MatrixType, ProdId, Ord>>(execPipeline, computationStateManager);
        localGraph.template edge<MatrixTile<MatrixType, ProdId, Ord>>(receiverTask, computationStateManager);
        localGraph.template edge<std::pair<std::shared_ptr<MatrixTile<MatrixType, IdC, Ord>>, std::shared_ptr<MatrixTile<MatrixType, ProdId, Ord>>>>(
            computationStateManager,
            accumulateTask
        );
        localGraph.template edge<MatrixTile<MatrixType, IdC, Ord>>(accumulateTask, computationStateManager);
        localGraph.template edge<TtlManagedMemory>(accumulateTask, recyclerTask);
        localGraph.template edge<MatrixTile<MatrixType, IdC, Ord>>(accumulateTask, outputStateManager);
        localGraph.template edge<MatrixTile<MatrixType, IdC, Ord>>(outputStateManager, senderTask);

        // execute graph
        localGraph.executeGraph();

        // push data
        localGraph.pushData(matrixA);
        localGraph.pushData(matrixB);
        localGraph.pushData(matrixC);
        localGraph.finishPushingData();

        // wait
        localGraph.waitForTermination();

        // create dot files for analysis
        localGraph.createDotFile(
            dotFile + std::to_string(getNodeId()) + ".dot",
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::ALL
        );
    }

    [[nodiscard]] std::string strategy() const override {
        return "OuterProduct";
    }

    [[nodiscard]] std::string matTypeA() const override {
        return "Sub";
    }

    [[nodiscard]] std::string matTypeB() const override {
        return "Sub";
    }

    [[nodiscard]] std::string matTypeC() const override {
        return "Cyclic2d";
    }

    [[nodiscard]] std::string className() const override {
        return NAME(MMD_MpiOuterProductCyclic2d);
    }
};

#endif //HH3_MATMUL_MMD_H

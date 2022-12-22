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
#include "data/tiled_sub_matrix_container.h"
#include "execution_pipeline/outer_product_exec_pipeline.h"
#include "graph/outer_product_cuda_graph.h"
#include "graph/outer_product_cuda_graph_unified_memory.h"
#include "state/outer_product_computation_state.h"
#include "state/outer_product_computation_state_manager.h"
#include "state/outer_product_output_state.h"
#include "task/accumulate_task.h"
#include "task/comm_tasks.h"
#include "task/matrix_col_traversal_task.h"
#include "task/matrix_row_traversal_task.h"
#include <cblas.h>

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
        double gflops = (double(M) * double(K) * double(N) * double(2)) / (1.e9 * time);
        printf(
            "[ " GREEN("MMD") " ][ Strategy = " GREEN("%-20s") " ][ A = " GREEN("%-13s") " ][ B = " GREEN("%-13s") " ][ C = " GREEN("%-13s") " ][ " CYAN("%9.3f") " gflops ][ " RED("%8.3f") " secs ]\n",
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
        assert((std::is_same_v<MatrixType, double> or std::is_same_v<MatrixType, float>));

        auto subA = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Col, MatrixType, IdA, Ord>>(matrixA);
        auto subB = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Row, MatrixType, IdB, Ord>>(matrixB);
        auto matC = std::static_pointer_cast<RedundantMatrixContainer<MatrixType, IdC, Ord>>(matrixC);
        uint64_t m = matC->matrixHeight(), k = subA->subMatrixWidth(), n = matC->matrixWidth();
        uint64_t tileSize = matC->matrixTileSize();

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
        }
        else {
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

        // free up space
        matrixA = nullptr;
        matrixB = nullptr;
        subA = nullptr;
        subB = nullptr;

        uint64_t len = std::min(m*n, uint64_t(INT32_MAX));
        std::vector<MatrixType> tempC((isRootNodeId()? len: 0));
        for(uint64_t idx = 0, limit = m*n; idx < limit; idx += len) {
            len = (((idx+len) <= limit)? len: limit-idx);
            checkMpiErrors(MPI_Reduce(
                &matC->data()[idx],
                tempC.data(),
                len,
                std::is_same_v<MatrixType, double>? MPI_DOUBLE: MPI_FLOAT,
                MPI_SUM,
                0,
                matC->mpiComm()
            ));
            if(isRootNodeId()) {
                std::memcpy(&matC->data()[idx], tempC.data(), sizeof(MatrixType)*len);
            }
        }
    }

    [[nodiscard]] std::string strategy() const override {
        return "CublasXt+MPI";
    }

    [[nodiscard]] std::string matTypeA() const override {
        return "ContiguousSub";
    }

    [[nodiscard]] std::string matTypeB() const override {
        return "ContiguousSub";
    }

    [[nodiscard]] std::string matTypeC() const override {
        return "Redundant";
    }

    [[nodiscard]] std::string className() const override {
        return NAME(MMD_VerifyCublas);
    }
};

template<class MatrixType, char IdA, char IdB, char IdC, Order Ord>
class MMD_VerifyOpenblas: public MMD_Strategy<MatrixType, IdA, IdB, IdC, Ord> {
private:
    void executeImpl(
        std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
        std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
        std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile
    ) override {
        assert((std::is_same_v<MatrixType, double> or std::is_same_v<MatrixType, float>));

        auto subA = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Col, MatrixType, IdA, Ord>>(matrixA);
        auto subB = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Row, MatrixType, IdB, Ord>>(matrixB);
        auto matC = std::static_pointer_cast<RedundantMatrixContainer<MatrixType, IdC, Ord>>(matrixC);
        size_t m = matC->matrixHeight(), k = subA->subMatrixWidth(), n = matC->matrixWidth();
        size_t tileSize = matC->matrixTileSize();

        if constexpr(std::is_same<MatrixType, double>::value) {
            cblas_dgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1,
                (double *) subA->data(), subA->leadingDimension(),
                (double *) subB->data(), subB->leadingDimension(), 1,
                (double *) matC->data(), matC->leadingDimension()
            );
        }
        else {
            cblas_sgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1,
                (float *) subA->data(), subA->leadingDimension(),
                (float *) subB->data(), subB->leadingDimension(), 1,
                (float *) matC->data(), matC->leadingDimension()
            );
        }

        // free up space
        matrixA = nullptr;
        matrixB = nullptr;
        subA = nullptr;
        subB = nullptr;

        uint64_t len = std::min(m*n, uint64_t(INT32_MAX));
        std::vector<MatrixType> tempC((isRootNodeId()? len: 0));
        for(uint64_t idx = 0, limit = m*n; idx < limit; idx += len) {
            len = (((idx+len) <= limit)? len: limit-idx);
            checkMpiErrors(MPI_Reduce(
                &matC->data()[idx],
                tempC.data(),
                len,
                std::is_same_v<MatrixType, double>? MPI_DOUBLE: MPI_FLOAT,
                MPI_SUM,
                0,
                matC->mpiComm()
            ));
            if(isRootNodeId()) {
                std::memcpy(&matC->data()[idx], tempC.data(), sizeof(MatrixType)*len);
            }
        }
    }

    [[nodiscard]] std::string strategy() const override {
        return "OpenBlas+MPI";
    }

    [[nodiscard]] std::string matTypeA() const override {
        return "ContiguousSub";
    }

    [[nodiscard]] std::string matTypeB() const override {
        return "ContiguousSub";
    }

    [[nodiscard]] std::string matTypeC() const override {
        return "Redundant";
    }

    [[nodiscard]] std::string className() const override {
        return NAME(MMD_VerifyCublas);
    }
};

template<class MatrixType, char IdA, char IdB, char IdC, Order Ord>
class MMD_MpiOuterProduct1: public MMD_Strategy<MatrixType, IdA, IdB, IdC, Ord> {
public:
    explicit MMD_MpiOuterProduct1(size_t productThreads = 4, size_t commThreads = 8)
        : productThreads_(productThreads), commThreads_(commThreads) {}

private:
    void executeImpl(
        std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
        std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
        std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile
    ) override {
        constexpr char ProdId = 'p';
        constexpr char NetId  = 'n';
        auto subA = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Col, MatrixType, IdA, Ord>>(matrixA);
        auto subB = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Row, MatrixType, IdB, Ord>>(matrixB);
        auto matC = std::static_pointer_cast<Cyclic2dMatrixContainer<MatrixType, IdC, Ord>>(matrixC);

        const uint64_t mTiles   = matrixC->matrixNumRowTiles();
        const uint64_t kTiles   = matrixA->matrixNumColTiles();
        const uint64_t nTiles   = matrixC->matrixNumColTiles();
        const uint64_t tileSize = matrixC->matrixTileSize();

        const uint64_t kTilesOnNode = subA->subMatrixNumColTiles();
        std::vector<int32_t> devices;
        devices.reserve(kTilesOnNode);
        for(int32_t i = 0; i < kTilesOnNode and i < deviceIds.size(); ++i) {
            devices.template emplace_back(deviceIds[i]);
        }

        uint64_t myTiles = mTiles*nTiles;
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
        auto accumulateTask       = std::make_shared<AccumulateTask<MatrixType, IdC, ProdId, NetId, Ord>>(devices.size()*productThreads_);
        auto senderTask           = std::make_shared<Cyclic2dSenderTask<MatrixType, IdC, Ord>>(commThreads_, mTiles*nTiles-myTiles);
        auto receiverTask         = std::make_shared<Cyclic2dReceiverTask<MatrixType, ProdId, Ord>>(myTiles*(getNumNodes()-1));

        auto memoryManager = std::make_shared<hh::StaticMemoryManager<MatrixTile<MatrixType, ProdId, Ord>, uint64_t>>(commThreads_, tileSize);
        receiverTask->connectMemoryManager(memoryManager);

        auto cudaGraph    = std::make_shared<OuterProductCudaGraph<MatrixType, IdA, IdB, ProdId, Ord>>(mTiles, kTiles, nTiles, tileSize, productThreads_);
        auto execPipeline = std::make_shared<OuterProductExecPipeline<MatrixType, IdA, IdB, ProdId, Ord>>(cudaGraph, devices);

        auto computationState        = std::make_shared<OuterProductComputationState<MatrixType, IdC, ProdId, NetId, Ord>>(
            mTiles,
            nTiles,
            mTiles*nTiles*subA->subMatrixNumColTiles() + myTiles*(getNumNodes()-1)
        );
        auto computationStateManager = std::make_shared<OuterProductComputationStateManager<MatrixType, IdC, ProdId, NetId, Ord>>(computationState);
        auto outputState             = std::make_shared<OuterProductOutputState<MatrixType, IdC, Ord>>(mTiles, nTiles, subA->subMatrixNumColTiles());
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
            hh::StructureOptions::QUEUE,
            hh::DebugOptions::NONE,
            std::make_unique<hh::JetColor>(),
            false
        );
    }

    [[nodiscard]] std::string strategy() const override {
        return "OuterProduct";
    }

    [[nodiscard]] std::string matTypeA() const override {
        return "ContiguousSub";
    }

    [[nodiscard]] std::string matTypeB() const override {
        return "ContiguousSub";
    }

    [[nodiscard]] std::string matTypeC() const override {
        return "Cyclic2d";
    }

    [[nodiscard]] std::string className() const override {
        return NAME(MMD_MpiOuterProduct1);
    }

private:
    size_t productThreads_ = 0;
    size_t commThreads_    = 0;
};

template<class MatrixType, char IdA, char IdB, char IdC, Order Ord>
class MMD_MpiOuterProduct2: public MMD_Strategy<MatrixType, IdA, IdB, IdC, Ord> {
public:
    explicit MMD_MpiOuterProduct2(size_t productThreads = 4, size_t commThreads = 8)
        : productThreads_(productThreads), commThreads_(commThreads) {}

//private:
    void executeImpl(
        std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
        std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
        std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile
    ) override {
        constexpr char ProdId = 'p';
        constexpr char NetId  = 'n';
        auto subA = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Col, MatrixType, IdA, Ord>>(matrixA);
        auto subB = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Row, MatrixType, IdB, Ord>>(matrixB);
        auto matC = std::static_pointer_cast<Cyclic2dMatrixContainer<MatrixType, IdC, Ord>>(matrixC);

        const uint64_t mTiles   = matrixC->matrixNumRowTiles();
        const uint64_t kTiles   = matrixA->matrixNumColTiles();
        const uint64_t nTiles   = matrixC->matrixNumColTiles();
        const uint64_t tileSize = matrixC->matrixTileSize();

        const uint64_t kTilesOnNode = subA->subMatrixNumColTiles();
        std::vector<int32_t> devices;
        devices.reserve(kTilesOnNode);
        for(int32_t i = 0; i < kTilesOnNode and i < deviceIds.size(); ++i) {
            devices.template emplace_back(deviceIds[i]);
        }

        uint64_t myTiles = mTiles*nTiles;
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
        auto accumulateTask       = std::make_shared<AccumulateTask<
            MatrixType, IdC, ProdId, NetId, Ord,
            MatrixTile<MatrixType, IdC, Ord>,
            UnifiedMatrixTile<MatrixType, ProdId, Ord>
        >>(devices.size()*productThreads_);
        auto senderTask           = std::make_shared<Cyclic2dSenderTask<MatrixType, IdC, Ord>>(commThreads_, mTiles*nTiles-myTiles);
        auto receiverTask         = std::make_shared<Cyclic2dReceiverTask<MatrixType, NetId, Ord>>(myTiles*(getNumNodes()-1));

        auto memoryManager = std::make_shared<hh::StaticMemoryManager<MatrixTile<MatrixType, NetId, Ord>, uint64_t>>(commThreads_, tileSize);
        receiverTask->connectMemoryManager(memoryManager);

        auto cudaGraph    = std::make_shared<OuterProductCudaGraphUnifiedMemory<MatrixType, IdA, IdB, ProdId, Ord>>(mTiles, kTiles, nTiles, tileSize, productThreads_);
        auto execPipeline = std::make_shared<OuterProductExecPipeline<
            MatrixType, IdA, IdB, ProdId, Ord,
            MatrixTile<MatrixType, IdA, Ord>,
            MatrixTile<MatrixType, IdB, Ord>,
            UnifiedMatrixTile<MatrixType, ProdId, Ord>
        >>(cudaGraph, devices);

        auto computationState        = std::make_shared<OuterProductComputationState<
            MatrixType, IdC, ProdId, NetId, Ord,
            MatrixTile<MatrixType, IdC, Ord>,
            UnifiedMatrixTile<MatrixType, ProdId, Ord>,
            MatrixTile<MatrixType, NetId, Ord>
        >>(
            mTiles,
            nTiles,
            mTiles*nTiles*subA->subMatrixNumColTiles() + myTiles*(getNumNodes()-1)
        );
        auto computationStateManager = std::make_shared<OuterProductComputationStateManager<
            MatrixType, IdC, ProdId, NetId, Ord,
            MatrixTile<MatrixType, IdC, Ord>,
            UnifiedMatrixTile<MatrixType, ProdId, Ord>,
            MatrixTile<MatrixType, NetId, Ord>
        >>(computationState);
        auto outputState             = std::make_shared<OuterProductOutputState<MatrixType, IdC, Ord>>(mTiles, nTiles, subA->subMatrixNumColTiles());
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
        localGraph.template edge<UnifiedMatrixTile<MatrixType, ProdId, Ord>>(execPipeline, computationStateManager);
        localGraph.template edge<MatrixTile<MatrixType, NetId, Ord>>(receiverTask, computationStateManager);
        localGraph.template edge<std::pair<std::shared_ptr<MatrixTile<MatrixType, IdC, Ord>>, std::shared_ptr<UnifiedMatrixTile<MatrixType, ProdId, Ord>>>>(
            computationStateManager,
            accumulateTask
        );
        localGraph.template edge<std::pair<std::shared_ptr<MatrixTile<MatrixType, IdC, Ord>>, std::shared_ptr<MatrixTile<MatrixType, NetId, Ord>>>>(
            computationStateManager,
            accumulateTask
        );
        localGraph.template edge<MatrixTile<MatrixType, IdC, Ord>>(accumulateTask, computationStateManager);
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
            hh::StructureOptions::QUEUE,
            hh::DebugOptions::NONE,
            std::make_unique<hh::JetColor>(),
            false
        );
    }

    [[nodiscard]] std::string strategy() const override {
        return "OuterProduct+Unified";
    }

    [[nodiscard]] std::string matTypeA() const override {
        return "ContiguousSub";
    }

    [[nodiscard]] std::string matTypeB() const override {
        return "ContiguousSub";
    }

    [[nodiscard]] std::string matTypeC() const override {
        return "Cyclic2d";
    }

    [[nodiscard]] std::string className() const override {
        return NAME(MMD_MpiOuterProduct2);
    }

private:
    size_t productThreads_ = 0;
    size_t commThreads_    = 0;
};

template<class MatrixType, char IdA, char IdB, char IdC, Order Ord>
class MMD_MpiOuterProduct3: public MMD_Strategy<MatrixType, IdA, IdB, IdC, Ord> {
public:
    explicit MMD_MpiOuterProduct3(size_t productThreads = 4, size_t commThreads = 8)
        : productThreads_(productThreads), commThreads_(commThreads) {}

//private:
    void executeImpl(
        std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
        std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
        std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile
    ) override {
        constexpr char ProdId = 'p';
        constexpr char NetId  = 'n';
        auto subA = std::static_pointer_cast<TiledSubMatrixContainer<Order::Col, MatrixType, IdA, Ord>>(matrixA);
        auto subB = std::static_pointer_cast<TiledSubMatrixContainer<Order::Row, MatrixType, IdB, Ord>>(matrixB);
        auto matC = std::static_pointer_cast<Cyclic2dMatrixContainer<MatrixType, IdC, Ord>>(matrixC);

        const uint64_t mTiles   = matrixC->matrixNumRowTiles();
        const uint64_t kTiles   = matrixA->matrixNumColTiles();
        const uint64_t nTiles   = matrixC->matrixNumColTiles();
        const uint64_t tileSize = matrixC->matrixTileSize();

        const uint64_t kTilesOnNode = subA->subMatrixNumColTiles();
        std::vector<int32_t> devices;
        devices.reserve(kTilesOnNode);
        for(int32_t i = 0; i < kTilesOnNode and i < deviceIds.size(); ++i) {
            devices.template emplace_back(deviceIds[i]);
        }

        uint64_t myTiles = mTiles*nTiles;
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
        auto accumulateTask       = std::make_shared<AccumulateTask<
            MatrixType, IdC, ProdId, NetId, Ord,
            MatrixTile<MatrixType, IdC, Ord>,
            UnifiedMatrixTile<MatrixType, ProdId, Ord>
        >>(devices.size()*productThreads_);
        auto senderTask           = std::make_shared<Cyclic2dSenderTask<MatrixType, IdC, Ord>>(commThreads_, mTiles*nTiles-myTiles);
        auto receiverTask         = std::make_shared<Cyclic2dReceiverTask<MatrixType, NetId, Ord>>(myTiles*(getNumNodes()-1));

        auto memoryManager = std::make_shared<hh::StaticMemoryManager<MatrixTile<MatrixType, NetId, Ord>, uint64_t>>(commThreads_, tileSize);
        receiverTask->connectMemoryManager(memoryManager);

        auto cudaGraph    = std::make_shared<OuterProductCudaGraphUnifiedMemory<MatrixType, IdA, IdB, ProdId, Ord>>(mTiles, kTiles, nTiles, tileSize, productThreads_);
        auto execPipeline = std::make_shared<OuterProductExecPipeline<
            MatrixType, IdA, IdB, ProdId, Ord,
            MatrixTile<MatrixType, IdA, Ord>,
            MatrixTile<MatrixType, IdB, Ord>,
            UnifiedMatrixTile<MatrixType, ProdId, Ord>
        >>(cudaGraph, devices);

        auto computationState        = std::make_shared<OuterProductComputationState<
            MatrixType, IdC, ProdId, NetId, Ord,
            MatrixTile<MatrixType, IdC, Ord>,
            UnifiedMatrixTile<MatrixType, ProdId, Ord>,
            MatrixTile<MatrixType, NetId, Ord>
        >>(
            mTiles,
            nTiles,
            mTiles*nTiles*subA->subMatrixNumColTiles() + myTiles*(getNumNodes()-1)
        );
        auto computationStateManager = std::make_shared<OuterProductComputationStateManager<
            MatrixType, IdC, ProdId, NetId, Ord,
            MatrixTile<MatrixType, IdC, Ord>,
            UnifiedMatrixTile<MatrixType, ProdId, Ord>,
            MatrixTile<MatrixType, NetId, Ord>
        >>(computationState);
        auto outputState             = std::make_shared<OuterProductOutputState<MatrixType, IdC, Ord>>(mTiles, nTiles, subA->subMatrixNumColTiles());
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
        localGraph.template edge<UnifiedMatrixTile<MatrixType, ProdId, Ord>>(execPipeline, computationStateManager);
        localGraph.template edge<MatrixTile<MatrixType, NetId, Ord>>(receiverTask, computationStateManager);
        localGraph.template edge<std::pair<std::shared_ptr<MatrixTile<MatrixType, IdC, Ord>>, std::shared_ptr<UnifiedMatrixTile<MatrixType, ProdId, Ord>>>>(
            computationStateManager,
            accumulateTask
        );
        localGraph.template edge<std::pair<std::shared_ptr<MatrixTile<MatrixType, IdC, Ord>>, std::shared_ptr<MatrixTile<MatrixType, NetId, Ord>>>>(
            computationStateManager,
            accumulateTask
        );
        localGraph.template edge<MatrixTile<MatrixType, IdC, Ord>>(accumulateTask, computationStateManager);
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
            hh::StructureOptions::QUEUE,
            hh::DebugOptions::NONE,
            std::make_unique<hh::JetColor>(),
            false
        );
    }

    [[nodiscard]] std::string strategy() const override {
        return "OuterProduct+Unified";
    }

    [[nodiscard]] std::string matTypeA() const override {
        return "TiledSub";
    }

    [[nodiscard]] std::string matTypeB() const override {
        return "TiledSub";
    }

    [[nodiscard]] std::string matTypeC() const override {
        return "Cyclic2d";
    }

    [[nodiscard]] std::string className() const override {
        return NAME(MMD_MpiOuterProduct3);
    }

private:
    size_t productThreads_ = 0;
    size_t commThreads_    = 0;
};

#endif //HH3_MATMUL_MMD_H

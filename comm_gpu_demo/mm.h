#ifndef HEDGEHOG_TUTORIALS_MM_H
#define HEDGEHOG_TUTORIALS_MM_H

#include "utility.h"
#ifdef HH_USE_CUDA
#include <cublasXt.h>
#else
#include <openblas/cblas.h>
#endif
#include <comm/comm.h>
#include <mpi.h>
#include "data/matrix_data.h"
#include "execution_pipeline/inner_product_exec_pipeline.h"
#include "execution_pipeline/multi_gpu_exec_pipeline.h"
#include "graph/gpu_computation_graph.h"
#include "graph/inner_product_cuda_graph.h"
#include "state/output_state.h"
#include "state/partial_computation_state_manager.h"
#include "task/addition_task.h"
#include "task/comm_task.h"
#include "task/matrix_block_transformer_task.h"
#include "task/matrix_column_traversal_task.h"
#include "task/matrix_row_traversal_task.h"
#include "task/contiguous_matrix_block_generator_task.h"

#define NAME(x) #x

template<class MatrixType, Order Ord>
class MM_Strategy {
private:
    virtual void executeImpl(
        std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
        std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB,
        std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> &matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile = ""
    ) = 0;
    virtual std::string toString() const = 0;
    virtual std::string className() const = 0;

public:
    void execute(
        std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
        std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB,
        std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> &matrixC,
        const std::vector<int32_t> &deviceIds
    ) {
        auto start = std::chrono::high_resolution_clock::now();
        this->executeImpl(matrixA, matrixB, matrixC, deviceIds, className());
        auto end = std::chrono::high_resolution_clock::now();

        int32_t flag = false, mpiNodeId = 0, mpiNumNodes = 1;
        if(auto status = MPI_Initialized(&flag); status == MPI_SUCCESS and flag) {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpiNodeId);
            MPI_Comm_size(MPI_COMM_WORLD, &mpiNumNodes);
        }
        if(mpiNodeId != 0) return;

        size_t M = matrixC->matrixHeight(), N = matrixC->matrixWidth(), K = matrixA->matrixWidth()*mpiNumNodes;//FIXME: K
        double time = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1.e9;
        //https://forums.developer.nvidia.com/t/how-to-compute-gflops-for-gemm-blas/20218/6
        double gflops = (M*N*(2*K+2))/(1.e9*time);
        printf(
            "[ " GREEN("%-64s") " ][ " CYAN("%9.3f") " gflops ][ " RED("%8.3f") " secs ]\n",
            this->toString().c_str(),
            gflops,
            time
        );
        fflush(stdout);
    }
};

/**
 * @brief  Use this strategy to calculate matrix multiplication results for verification
 * @tparam MatrixType float or double
 * @tparam order Column or Row
 */
template<class MatrixType, Order order>
class MM_Verification: public MM_Strategy<MatrixType, order> {
private:
    void executeImpl(
        std::shared_ptr<MatrixData<MatrixType, 'a', order>> &matrixA,
        std::shared_ptr<MatrixData<MatrixType, 'b', order>> &matrixB,
        std::shared_ptr<MatrixData<MatrixType, 'c', order>> &matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile = ""
    ) override {
        size_t m = matrixA->matrixHeight(), k = matrixA->matrixWidth(), n = matrixB->matrixWidth();
        size_t blockSize = matrixC->blockSize();
#if HH_USE_CUDA
        cublasXtHandle_t handle;
        checkCudaErrors(cublasXtCreate(&handle));

        checkCudaErrors(cublasXtDeviceSelect(handle, deviceIds.size(), (int*)deviceIds.data()));
        checkCudaErrors(cublasXtSetBlockDim(handle, blockSize));

        MatrixType alpha = 1.0, beta = 1.0;

        if constexpr (std::is_same<MatrixType, double>::value) {
            checkCudaErrors(cublasXtDgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, (double *) (&alpha),
                (double *) matrixA->data(), matrixA->leadingDimension(),
                (double *) matrixB->data(), matrixB->leadingDimension(),
                (double *) (&beta), (double *) matrixC->data(), matrixC->leadingDimension()
            ));
        } else {
            checkCudaErrors(cublasXtSgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, (float *) (&alpha),
                (float *) matrixA->data(), matrixA->leadingDimension(),
                (float *) matrixB->data(), matrixB->leadingDimension(),
                (float *) (&beta), (float *) matrixC->data(), matrixC->leadingDimension()
            ));
        }

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cublasXtDestroy(handle));
#else
        if(std::is_same<MatrixType, float>::value) {
            cblas_sgemm(
                order == Order::Column ? CblasColMajor : CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1,
                (float *) matrixA->data(), matrixA->leadingDimension(),
                (float *) matrixB->data(), matrixB->leadingDimension(), 1,
                (float *) matrixC->data(), matrixC->leadingDimension()
            );
        } else if (std::is_same<MatrixType, double>::value) {
            cblas_dgemm(
                order == Order::Column ? CblasColMajor : CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1,
                (double *) matrixA->data(), matrixA->leadingDimension(),
                (double *) matrixB->data(), matrixB->leadingDimension(), 1,
                (double *) matrixC->data(), matrixC->leadingDimension()
            );
        } else {
            std::cerr << "The matrix can't be multiplied" << std::endl;
            exit(43);
        }
#endif
    }

    std::string toString() const override {
        return "MM cublasXt verification";
    }

    std::string className() const override {
        return NAME(MM_Verification);
    }
};

template<class MatrixType, Order Ord>
class MM_MpiVerification: public MM_Strategy<MatrixType, Ord> {
private:
    void executeImpl(
            std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
            std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB,
            std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> &matrixC,
            const std::vector<int32_t> &deviceIds,
            std::string dotFile = ""
    ) override {
        int32_t mpiNodeId = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiNodeId);
        MM_Verification<MatrixType, Ord>().executeImpl(matrixA, matrixB, matrixC, deviceIds);
        std::vector<MatrixType> tempC((mpiNodeId == 0? matrixC->matrixWidth()*matrixC->matrixHeight(): 0), 0);
        if constexpr(std::is_same_v<MatrixType, double>) {
            MPI_Reduce(matrixC->data(), tempC.data(), matrixC->matrixWidth()*matrixC->matrixHeight(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else if constexpr(std::is_same_v<MatrixType, float>) {
            MPI_Reduce(matrixC->data(), tempC.data(), matrixC->matrixWidth()*matrixC->matrixHeight(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            throw std::runtime_error("Type not supported\n");
        }
        std::copy(tempC.begin(), tempC.end(), matrixC->data());
    }

    std::string toString() const override {
        return "MM multi node cublasXt verification";
    }

    std::string className() const override {
        return NAME(MM_MpiVerification);
    }
};

/**
 * @brief Use this strategy to compute matrix multiplication <br>
 * 1. uses outer product strategy <br>
 * 2. supports multiple GPUs <br>
 * @tparam MatrixType float or double
 * @tparam order Column or Row
 */
template<class MatrixType, Order Ord>
class MM_OuterProduct: public MM_Strategy<MatrixType, Ord> {
private:
    void executeImpl(
            std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
            std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB,
            std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> &matrixC,
            const std::vector<int32_t> &deviceIds,
            std::string dotFile = ""
    ) override {
        // initial values
        size_t M = matrixA->matrixHeight(), K = matrixA->matrixWidth(), N = matrixB->matrixWidth(), blockSize = matrixC->blockSize();
        size_t mBlocks = std::ceil(M / blockSize) + (M % blockSize == 0 ? 0 : 1);
        size_t kBlocks = std::ceil(K / blockSize) + (K % blockSize == 0 ? 0 : 1);
        size_t nBlocks = std::ceil(N / blockSize) + (N % blockSize == 0 ? 0 : 1);

        // create nodes
        auto mainGraph = hh::Graph<3,
                MatrixData<MatrixType, 'a', Ord>,       //inp1
                MatrixData<MatrixType, 'b', Ord>,       //inp2
                MatrixData<MatrixType, 'c', Ord>,       //inp3
                MatrixBlockData<MatrixType, 'c', Ord>   //out1
        >("Main Graph");

        auto matrixATraversalTask = std::make_shared<MatrixColumnTraversalTask<MatrixType, 'a', Ord>>();
        auto matrixBTraversalTask = std::make_shared<MatrixRowTraversalTask<MatrixType, 'b', Ord>>();
        auto matrixCTraversalTask = std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Ord>>();
        auto additionTask = std::make_shared<AdditionTask<MatrixType, Ord>>(4);

        auto partialComputationState = std::make_shared<PartialComputationState<MatrixType, Ord>>(mBlocks, nBlocks, mBlocks*nBlocks*kBlocks);
        auto partialComputationStateManager = std::make_shared<PartialComputationStateManager<MatrixType, Ord>>(partialComputationState);

        auto gpuComputationGraph = std::make_shared<GPUComputationGraph<MatrixType, Ord>>(M, K, N, blockSize);
        auto multiGpuExecutionPipeline = std::make_shared<MultiGPUExecPipeline<MatrixType, Ord>>(gpuComputationGraph, deviceIds);

        auto outputState = std::make_shared<OutputState<MatrixType, Ord>>(mBlocks, nBlocks, kBlocks);

        // StateManager
        auto outputBlockStateManager = std::make_shared<hh::StateManager<1,
                MatrixBlockData<MatrixType, 'c', Ord>,
                MatrixBlockData<MatrixType, 'c', Ord>>
        >(outputState, "Output State Manager");

        // add edges
        mainGraph.template input<MatrixData<MatrixType, 'a', Ord>>(matrixATraversalTask);
        mainGraph.template input<MatrixData<MatrixType, 'b', Ord>>(matrixBTraversalTask);
        mainGraph.template input<MatrixData<MatrixType, 'c', Ord>>(matrixCTraversalTask);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(matrixCTraversalTask, partialComputationStateManager);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'a', Ord>>(matrixATraversalTask, multiGpuExecutionPipeline);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'b', Ord>>(matrixBTraversalTask, multiGpuExecutionPipeline);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'p', Ord>>(multiGpuExecutionPipeline, partialComputationStateManager);
        mainGraph.template edge<std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>>(partialComputationStateManager, additionTask);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(additionTask, partialComputationStateManager);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(additionTask, outputBlockStateManager);
        mainGraph.template output<MatrixBlockData<MatrixType, 'c', Ord>>(outputBlockStateManager);

        // execute graph
        mainGraph.executeGraph();

        // push data
        mainGraph.pushData(matrixA);
        mainGraph.pushData(matrixB);
        mainGraph.pushData(matrixC);
        mainGraph.finishPushingData();

        // wait
        mainGraph.waitForTermination();

        // create dot files for analysis
        mainGraph.createDotFile(
            dotFile + ".dot",
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::ALL
        );
    }

    std::string toString() const override {
        return "MM outer product";
    }

    std::string className() const override {
        return NAME(MM_OuterProduct);
    }
};

template<class MatrixType, Order Ord>
class MM_OuterProduct2: public MM_Strategy<MatrixType, Ord> {
private:
    void executeImpl(
        std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
        std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB,
        std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> &matrixC, const std::vector<int32_t> &deviceIds,
        std::string dotFile
    ) override {
        // initial values
        size_t M = matrixA->matrixHeight(), K = matrixA->matrixWidth(), N = matrixB->matrixWidth(), blockSize = matrixC->blockSize();
        size_t mBlocks = std::ceil(M / blockSize) + (M % blockSize == 0 ? 0 : 1);
        size_t kBlocks = std::ceil(K / blockSize) + (K % blockSize == 0 ? 0 : 1);
        size_t nBlocks = std::ceil(N / blockSize) + (N % blockSize == 0 ? 0 : 1);

        // create nodes
        auto mainGraph = hh::Graph<3,
                MatrixData<MatrixType, 'a', Ord>,       //inp1
                MatrixData<MatrixType, 'b', Ord>,       //inp2
                MatrixMetaData,                         //inp3
                MatrixBlockData<MatrixType, 'c', Ord>   //out1
        >("Main Graph");

        auto matrixATraversalTask = std::make_shared<MatrixColumnTraversalTask<MatrixType, 'a', Ord>>();
        auto matrixBTraversalTask = std::make_shared<MatrixRowTraversalTask<MatrixType, 'b', Ord>>();
        auto matrixCContiguousBlockGenTask = std::make_shared<ContiguousMatrixBlockGeneratorTask<MatrixType, 'c', Ord>>();
        auto additionTask = std::make_shared<AdditionTask<MatrixType, Ord>>(4);

        // TODO: Should use memory manager?
        auto blockCMemoryManager = std::make_shared<hh::StaticMemoryManager<MatrixBlockData<MatrixType, 'c', Ord>, size_t>>(mBlocks*nBlocks, blockSize);
        matrixCContiguousBlockGenTask->connectMemoryManager(blockCMemoryManager);

        auto partialComputationState = std::make_shared<PartialComputationState<MatrixType, Ord>>(mBlocks, nBlocks, mBlocks*nBlocks*kBlocks);
        auto partialComputationStateManager = std::make_shared<PartialComputationStateManager<MatrixType, Ord>>(partialComputationState);

        auto gpuComputationGraph = std::make_shared<GPUComputationGraph<MatrixType, Ord>>(M, K, N, blockSize);
        auto multiGpuExecutionPipeline = std::make_shared<MultiGPUExecPipeline<MatrixType, Ord>>(gpuComputationGraph, deviceIds);

        auto outputState = std::make_shared<OutputState<MatrixType, Ord>>(mBlocks, nBlocks, kBlocks);

        // StateManager
        auto outputBlockStateManager = std::make_shared<hh::StateManager<1,
                MatrixBlockData<MatrixType, 'c', Ord>,
                MatrixBlockData<MatrixType, 'c', Ord>>
        >(outputState, "Output State Manager");

        // add edges
        mainGraph.template input<MatrixData<MatrixType, 'a', Ord>>(matrixATraversalTask);
        mainGraph.template input<MatrixData<MatrixType, 'b', Ord>>(matrixBTraversalTask);
        mainGraph.template input<MatrixMetaData>(matrixCContiguousBlockGenTask);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(matrixCContiguousBlockGenTask, partialComputationStateManager);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'a', Ord>>(matrixATraversalTask, multiGpuExecutionPipeline);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'b', Ord>>(matrixBTraversalTask, multiGpuExecutionPipeline);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'p', Ord>>(multiGpuExecutionPipeline, partialComputationStateManager);
        mainGraph.template edge<std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>>(partialComputationStateManager, additionTask);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(additionTask, partialComputationStateManager);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(additionTask, outputBlockStateManager);
        mainGraph.template output<MatrixBlockData<MatrixType, 'c', Ord>>(outputBlockStateManager);

        // execute graph
        mainGraph.executeGraph();

        // push data
        mainGraph.pushData(matrixA);
        mainGraph.pushData(matrixB);
        mainGraph.pushData(std::make_shared<MatrixMetaData>(matrixA->matrixHeight(), matrixB->matrixWidth(), matrixA->blockSize()));
        mainGraph.finishPushingData();

        // wait
        // TODO: create a separate task to accumulate
        while(auto res = mainGraph.getBlockingResult()) {
            // FIXME: instead of 0, use MatrixBlockData<MatrixType, 'c', Ord> as template parameter for std::get
            auto blockC = std::get<0>(*res);
            MatrixType *outputLocation = &matrixC->data()[
                blockC->colIdx()*blockSize*matrixC->leadingDimension() +
                blockC->rowIdx()*blockSize
            ];
            for(size_t i = 0; i < blockC->blockSizeWidth(); ++i) {
                for(size_t j = 0; j < blockC->blockSizeHeight(); ++j) {
                    outputLocation[i*matrixC->leadingDimension() + j] += blockC->blockData()[i*blockC->leadingDimension() + j];
                }
            }
        }
        mainGraph.waitForTermination();

        // create dot files for analysis
        mainGraph.createDotFile(
            dotFile + ".dot",
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::ALL
        );
    }

    std::string toString() const override {
        return "MM Outer product, contiguous blocks for matrix C";
    }

    std::string className() const override {
        return NAME(MM_OuterProduct2);
    }
};

template<class MatrixType, Order Ord>
class MM_CommOuterProduct: public MM_Strategy<MatrixType, Ord> {
private:
    void executeImpl(
            std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
            std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB,
            std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> &matrixC,
            const std::vector<int32_t> &deviceIds,
            std::string dotFile = ""
    ) override {
        // initial values
        size_t M = matrixA->matrixHeight(), K = matrixA->matrixWidth(), N = matrixB->matrixWidth(), blockSize = matrixC->blockSize();
        size_t mBlocks = std::ceil(M / blockSize) + (M % blockSize == 0 ? 0 : 1);
        size_t kBlocks = std::ceil(K / blockSize) + (K % blockSize == 0 ? 0 : 1);
        size_t nBlocks = std::ceil(N / blockSize) + (N % blockSize == 0 ? 0 : 1);

        // create nodes
        auto mainGraph = hh::Graph<3,
            MatrixData<MatrixType, 'a', Ord>,       //inp1
            MatrixData<MatrixType, 'b', Ord>,       //inp2
            MatrixData<MatrixType, 'c', Ord>,       //inp3
            MatrixBlockData<MatrixType, 'c', Ord>,  //out1
            void*                                   //out2
        >("Main Graph");

        auto matrixATraversalTask = std::make_shared<MatrixColumnTraversalTask<MatrixType, 'a', Ord>>();
        auto matrixBTraversalTask = std::make_shared<MatrixRowTraversalTask<MatrixType, 'b', Ord>>();
        auto matrixCTraversalTask = std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Ord>>();
        auto additionTask = std::make_shared<AdditionTask<MatrixType, Ord>>(4);
        auto matrixBlockTransformerTask = std::make_shared<MatrixBlockTransformerTask<MatrixType, 'c', 'p', Ord>>();

        auto partialComputationState = std::make_shared<PartialComputationState<MatrixType, Ord>>(
            mBlocks,
            nBlocks,
            mBlocks*nBlocks*(kBlocks + (comm::isMpiRootPid()? (comm::getMpiNumNodes()-1): 0))
        );
        auto partialComputationStateManager = std::make_shared<PartialComputationStateManager<MatrixType, Ord>>(partialComputationState);

        auto gpuComputationGraph = std::make_shared<GPUComputationGraph<MatrixType, Ord>>(M, K, N, blockSize);
        auto multiGpuExecutionPipeline = std::make_shared<MultiGPUExecPipeline<MatrixType, Ord>>(gpuComputationGraph, deviceIds);

        auto outputState = std::make_shared<OutputState<MatrixType, Ord>>(
            mBlocks,
            nBlocks,
            kBlocks + (comm::isMpiRootPid()? (comm::getMpiNumNodes()-1): 0)
        );

        // StateManager
        auto outputBlockStateManager = std::make_shared<hh::StateManager<1,
                MatrixBlockData<MatrixType, 'c', Ord>,
                MatrixBlockData<MatrixType, 'c', Ord>
            >>(outputState, "Output State Manager");

        // add edges
        mainGraph.template input<MatrixData<MatrixType, 'a', Ord>>(matrixATraversalTask);
        mainGraph.template input<MatrixData<MatrixType, 'b', Ord>>(matrixBTraversalTask);
        mainGraph.template input<MatrixData<MatrixType, 'c', Ord>>(matrixCTraversalTask);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(matrixCTraversalTask, partialComputationStateManager);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'a', Ord>>(matrixATraversalTask, multiGpuExecutionPipeline);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'b', Ord>>(matrixBTraversalTask, multiGpuExecutionPipeline);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'p', Ord>>(multiGpuExecutionPipeline, partialComputationStateManager);
        mainGraph.template edge<std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>>(partialComputationStateManager, additionTask);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(additionTask, partialComputationStateManager);
        mainGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(additionTask, outputBlockStateManager);
        if(comm::isMpiRootPid()) {
            auto receiverTask = std::make_shared<ReceiverTask<MatrixBlockData<MatrixType, 'p', Ord>>>(mBlocks*nBlocks*(comm::getMpiNumNodes()-1));
            mainGraph.template edge<MatrixBlockData<MatrixType, 'p', Ord>>(receiverTask, partialComputationStateManager);
            mainGraph.template output<MatrixBlockData<MatrixType, 'c', Ord>>(outputBlockStateManager);
            //FIXME: end?
        }
        else {
            auto senderTask = std::make_shared<SenderTask<MatrixBlockData<MatrixType, 'p', Ord>>>(0, mBlocks*nBlocks);
            mainGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(outputBlockStateManager, matrixBlockTransformerTask);
            mainGraph.template edge<MatrixBlockData<MatrixType, 'p', Ord>>(matrixBlockTransformerTask, senderTask);
            mainGraph.template output<void*>(senderTask);
        }

        // execute graph
        mainGraph.executeGraph();

        // push data
        mainGraph.pushData(matrixA);
        mainGraph.pushData(matrixB);
        mainGraph.pushData(matrixC);
        mainGraph.finishPushingData();

        // wait
        mainGraph.waitForTermination();

        // create dot files for analysis
        mainGraph.createDotFile(
            dotFile + std::to_string(comm::getMpiNodeId()) + ".dot",
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::ALL
        );
    }

    std::string toString() const override {
        return "MM multi node outer product";
    }

    std::string className() const override {
        return NAME(MM_CommOuterProduct);
    }
};

template<class MatrixType, Order Ord>
class MM_MpiOuterProduct: public MM_Strategy<MatrixType, Ord> {
private:
    void executeImpl(
        std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
        std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB,
        std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> &matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile = ""
    ) override {
        int32_t mpiNodeId = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiNodeId);
        MM_OuterProduct<MatrixType, Ord>().executeImpl(matrixA, matrixB, matrixC, deviceIds, dotFile+std::to_string(mpiNodeId));
        std::vector<MatrixType> tempC((mpiNodeId == 0? matrixC->matrixWidth()*matrixC->matrixHeight(): 0), 0);
        if constexpr(std::is_same_v<MatrixType, double>) {
            MPI_Reduce(matrixC->data(), tempC.data(), matrixC->matrixWidth()*matrixC->matrixHeight(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else if constexpr(std::is_same_v<MatrixType, float>) {
            MPI_Reduce(matrixC->data(), tempC.data(), matrixC->matrixWidth()*matrixC->matrixHeight(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            throw std::runtime_error("Type not supported\n");
        }
        std::copy(tempC.begin(), tempC.end(), matrixC->data());
    }

    std::string toString() const override {
        return "MM multi node outer product using MPI, allocate and copy after";
    }

    std::string className() const override {
        return NAME(MM_MpiOuterProduct);
    }
};

template<class MatrixType, Order Ord>
class MM_MpiOuterProduct2: public MM_Strategy<MatrixType, Ord> {
private:
    void executeImpl(
        std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
        std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB,
        std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> &matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile = ""
    ) override {
        int32_t mpiNodeId = -1;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiNodeId);
        std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> tempMatrixC = matrixC;
        if(mpiNodeId == 0) {
            tempMatrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(
                matrixC->matrixHeight(),
                matrixC->matrixWidth(),
                matrixC->blockSize(),
                *(new MatrixType[matrixC->matrixWidth()*matrixC->matrixHeight()]),
                true
            );
            std::copy(matrixC->data(), matrixC->data()+matrixC->matrixWidth()*matrixC->matrixHeight(), tempMatrixC->data());
        }
        MM_OuterProduct<MatrixType, Ord>().executeImpl(matrixA, matrixB, tempMatrixC, deviceIds, dotFile+std::to_string(mpiNodeId));

        if constexpr(std::is_same_v<MatrixType, double>) {
            MPI_Reduce(tempMatrixC->data(), matrixC->data(), matrixC->matrixWidth()*matrixC->matrixHeight(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else if constexpr(std::is_same_v<MatrixType, float>) {
            MPI_Reduce(tempMatrixC->data(), matrixC->data(), matrixC->matrixWidth()*matrixC->matrixHeight(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        else {
            throw std::runtime_error("Type not supported\n");
        }
    }

    std::string toString() const override {
        return "MM multi node outer product using MPI, allocate and copy before";
    }

    std::string className() const override {
        return NAME(MM_MpiOuterProduct2);
    }
};

/**
 *
 *
 */
template<class MatrixType, Order Ord>
class MM_InnerProduct: public MM_Strategy<MatrixType, Ord> {
private:
    void executeImpl(
        [[maybe_unused]]std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
        [[maybe_unused]]std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB,
        [[maybe_unused]]std::shared_ptr<MatrixData<MatrixType, 'c', Ord>> &matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile = ""
    ) override {
        size_t M = matrixA->matrixHeight(), K = matrixA->matrixWidth(), N = matrixB->matrixWidth(), blockSize = matrixC->blockSize();
        size_t mBlocks = std::ceil(double(M) / double(blockSize));
        size_t kBlocks = std::ceil(double(K) / double(blockSize));
        size_t nBlocks = std::ceil(double(N) / double(blockSize));

        auto taskGraph = hh::Graph<1,
                MatrixData<MatrixType, 'c', Ord>,       //inp1
                MatrixBlockData<MatrixType, 'c', Ord>   //out1
            >("Inner Product Task Graph");
        auto cudaGraph = std::make_shared<InnerProductCudaGraph<MatrixType, Ord>>(M, K, N, blockSize, matrixA, matrixB);
        auto executionPipeline = std::make_shared<InnerProductExecPipeline<MatrixType, Ord>>(cudaGraph, deviceIds, nBlocks);

        auto matrixTraversal = std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Ord>>();

        // add edges
        taskGraph.template input<MatrixData<MatrixType, 'c', Ord>>(matrixTraversal);
        taskGraph.template edge<MatrixBlockData<MatrixType, 'c', Ord>>(matrixTraversal, executionPipeline);
        taskGraph.template output<MatrixBlockData<MatrixType, 'c', Ord>>(executionPipeline);

        // execute graph
        taskGraph.executeGraph();

        // push data
        taskGraph.pushData(matrixC);
        taskGraph.finishPushingData();

        // wait
        taskGraph.waitForTermination();

        // create dot files for analysis
        taskGraph.createDotFile(
            dotFile + ".dot",
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::ALL
        );
    }

    std::string toString() const override {
        return "MM inner product";
    }
};

#endif //HEDGEHOG_TUTORIALS_MM_H

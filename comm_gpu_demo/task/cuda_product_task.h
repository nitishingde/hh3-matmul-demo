// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.


#ifndef HH3_MATMUL_CUDA_PRODUCT_TASK_H
#define HH3_MATMUL_CUDA_PRODUCT_TASK_H

#include <hedgehog/api/task/abstract_cuda_task.h>
#include <cublas.h>
#include "../data/cuda_matrix_block_data.h"

template<class MatrixType, Order Ord>
class CudaProductTask: public hh::AbstractCUDATask<1,
        std::pair<std::shared_ptr<CudaMatrixBlockData<MatrixType, 'a', Ord>>, std::shared_ptr<CudaMatrixBlockData<MatrixType, 'b', Ord>>>,  //inp1
        CudaMatrixBlockData<MatrixType, 'p', Ord>                                                                                           //out1
    > {
private:
    using InputBlockPair = std::pair<std::shared_ptr<CudaMatrixBlockData<MatrixType, 'a', Ord>>, std::shared_ptr<CudaMatrixBlockData<MatrixType, 'b', Ord>>>;
    cublasHandle_t handle_{};

public:
    explicit CudaProductTask(size_t threadCount):
        hh::AbstractCUDATask<1,
            std::pair<std::shared_ptr<CudaMatrixBlockData<MatrixType, 'a', Ord>>, std::shared_ptr<CudaMatrixBlockData<MatrixType, 'b', Ord>>>,
            CudaMatrixBlockData<MatrixType, 'p', Ord>
        >("Cuda Product Task", threadCount, false, false) {}

    void initializeCuda() override {
        checkCudaErrors(cublasCreate_v2(&handle_));
        checkCudaErrors(cublasSetStream_v2(handle_, this->stream()));
    }

    void shutdownCuda() override {
        checkCudaErrors(cublasDestroy_v2(handle_));
    }

    void execute(std::shared_ptr<InputBlockPair> blockPair) override {
        auto cudaBlockA = blockPair->first;
        auto cudaBlockB = blockPair->second;
        MatrixType  alpha = 1., beta = 0.;

//        std::shared_ptr<CudaMatrixBlockData<MatrixType, 'p', Ord>>
        auto cudaBlockP = std::static_pointer_cast<CudaMatrixBlockData<MatrixType, 'p', Ord>>(this->getManagedMemory());
        cudaBlockP->rowIdx(cudaBlockA->rowIdx());
        cudaBlockP->colIdx(cudaBlockB->colIdx());
        cudaBlockP->blockSizeWidth(cudaBlockB->blockSizeWidth());
        cudaBlockP->blockSizeHeight(cudaBlockA->blockSizeHeight());
        cudaBlockP->leadingDimension(cudaBlockP->blockSizeHeight());
        cudaBlockP->ttl(1);

        if constexpr(std::is_same_v<MatrixType, float>) {
            checkCudaErrors(cublasSgemm_v2(
                handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                cudaBlockA->blockSizeHeight(), cudaBlockB->blockSizeWidth(), cudaBlockA->blockSizeWidth(), &alpha,
                (float *) cudaBlockA->blockData(), cudaBlockA->leadingDimension(),
                (float *) cudaBlockB->blockData(), cudaBlockB->leadingDimension(), &beta,
                (float *) cudaBlockP->blockData(), cudaBlockP->leadingDimension()
            ));
        }
        else if constexpr(std::is_same_v<MatrixType, double>){
            checkCudaErrors(cublasDgemm_v2(
                handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                cudaBlockA->blockSizeHeight(), cudaBlockB->blockSizeWidth(), cudaBlockA->blockSizeWidth(), &alpha,
                (double *) cudaBlockA->blockData(), cudaBlockA->leadingDimension(),
                (double *) cudaBlockB->blockData(), cudaBlockB->leadingDimension(), &beta,
                (double *) cudaBlockP->blockData(), cudaBlockP->leadingDimension()
            ));
        }
        else {
            std::cerr << "The matrix can't be multiplied" << std::endl;
            exit(43);
        }
        checkCudaErrors(cudaStreamSynchronize(this->stream()));

        cudaBlockA->used();
        cudaBlockA->returnToMemoryManager();
        cudaBlockB->used();
        cudaBlockB->returnToMemoryManager();

        this->addResult(cudaBlockP);
    }

    std::shared_ptr<hh::AbstractTask<1,
        std::pair<std::shared_ptr<CudaMatrixBlockData<MatrixType, 'a', Ord>>, std::shared_ptr<CudaMatrixBlockData<MatrixType, 'b', Ord>>>,
        CudaMatrixBlockData<MatrixType, 'p', Ord>
    >>
    copy() override {
        return std::make_shared<CudaProductTask>(this->numberThreads());
    }
};

#endif //HH3_MATMUL_CUDA_PRODUCT_TASK_H

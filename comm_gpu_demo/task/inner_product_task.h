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


#ifndef HH3_MATMUL_INNER_PRODUCT_TASK_H
#define HH3_MATMUL_INNER_PRODUCT_TASK_H

#include "pair_generator_task.h"//FIXME

template<class MatrixType, Order Ord>
class InnerProductTask:
    public hh::AbstractCUDATask<1,
        BlockTriplets<MatrixType, Ord>,         //inp1
        MatrixBlockData<MatrixType, 'p', Ord>   //out1
    > {
private:
    cublasHandle_t handle_ {};
    size_t blockSize_ = 0;
private:
    template<char Id>
    void copyToDevice(std::shared_ptr<MatrixBlockData<MatrixType, Id, Ord>> blockData, std::shared_ptr<CudaBlockData<MatrixType>> cudaBlockData) {
        checkCudaErrors(cublasSetMatrixAsync(
            (int)blockData->blockSizeHeight(), (int)blockData->blockSizeWidth(),
            sizeof(MatrixType),
            blockData->blockData(), (int)blockData->leadingDimension(),
            cudaBlockData->blockData(), (int)blockData->blockSizeHeight(),
            this->stream()
        ));
    }

    template<char Id>
    void copyToCpu(std::shared_ptr<MatrixBlockData<MatrixType, Id, Ord>> blockData, std::shared_ptr<CudaBlockData<MatrixType>> cudaBlockData) {
        checkCudaErrors(cudaMemcpyAsync(
            blockData->blockData(),
            cudaBlockData->blockData(),
            blockData->blockSizeHeight() * blockData->blockSizeWidth() * sizeof(MatrixType),
            cudaMemcpyDeviceToHost,
            this->stream()
        ));
    }
public:
    explicit InnerProductTask(size_t blockSize, size_t threadCount):
        blockSize_(blockSize),
        hh::AbstractCUDATask<1, BlockTriplets<MatrixType, Ord>, MatrixBlockData<MatrixType, 'p', Ord>>(
            "InnerProduct Task",
            threadCount,
            false
        ) {}

    void initializeCuda() override {
        checkCudaErrors(cublasCreate_v2(&handle_));
        checkCudaErrors(cublasSetStream_v2(handle_, this->stream()));
    }

    void shutdownCuda() override {
        checkCudaErrors(cublasDestroy_v2(handle_));
    }

    void execute(std::shared_ptr<BlockTriplets<MatrixType, Ord>> blockTriplets) {
        auto blockA = std::get<0>(*blockTriplets);
        auto blockB = std::get<1>(*blockTriplets);
        auto blockP = std::get<2>(*blockTriplets);

        auto cudaBlockA = std::static_pointer_cast<CudaBlockData<MatrixType>>(this->getManagedMemory());
        copyToDevice(blockA, cudaBlockA);

        auto cudaBlockB = std::static_pointer_cast<CudaBlockData<MatrixType>>(this->getManagedMemory());
        copyToDevice(blockB, cudaBlockB);

        auto cudaBlockP = std::static_pointer_cast<CudaBlockData<MatrixType>>(this->getManagedMemory());

        const MatrixType alpha = 1, beta = 0;
        if constexpr(std::is_same_v<MatrixType, float>) {
            checkCudaErrors(cublasSgemm_v2(
                handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                blockA->blockSizeHeight(), blockB->blockSizeWidth(), blockA->blockSizeWidth(), &alpha,
                (float *) cudaBlockA->blockData(), blockA->blockSizeHeight(),
                (float *) cudaBlockB->blockData(), blockB->blockSizeHeight(), &beta,
                (float *) cudaBlockP->blockData(), blockA->blockSizeHeight()
            ));
        }
        else if constexpr(std::is_same_v<MatrixType, double>){
            checkCudaErrors(cublasDgemm_v2(
                handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                blockA->blockSizeHeight(), blockB->blockSizeWidth(), blockA->blockSizeWidth(), &alpha,
                (double *) cudaBlockA->blockData(), blockA->blockSizeHeight(),
                (double *) cudaBlockB->blockData(), blockB->blockSizeHeight(), &beta,
                (double *) cudaBlockP->blockData(), blockA->blockSizeHeight()
            ));
        }
        else {
            std::cerr << "The matrix can't be multiplied" << std::endl;
            exit(43);
        }

        checkCudaErrors(cudaStreamSynchronize(this->stream()));
        copyToCpu(blockP, cudaBlockP);
        cudaBlockA->returnToMemoryManager();
        cudaBlockB->returnToMemoryManager();
        blockP->recordEvent(this->stream());//FIXME?
        checkCudaErrors(cudaStreamSynchronize(this->stream()));
        cudaBlockP->returnToMemoryManager();

        this->addResult(blockP);
    }

    std::shared_ptr<hh::AbstractTask<1, BlockTriplets<MatrixType, Ord>, MatrixBlockData<MatrixType, 'p', Ord>>>
    copy() override{
        return std::make_shared<InnerProductTask>(blockSize_, this->numberThreads());
    }
};

#endif //HH3_MATMUL_INNER_PRODUCT_TASK_H

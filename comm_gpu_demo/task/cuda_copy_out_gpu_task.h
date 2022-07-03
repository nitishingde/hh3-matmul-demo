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


#ifndef HH3_MATMUL_CUDA_COPY_OUT_GPU_TASK_H
#define HH3_MATMUL_CUDA_COPY_OUT_GPU_TASK_H

#include "../data/cuda_matrix_block_data.h"

template<class MatrixType, Order Ord>
class CudaCopyOutGpuTask: public hh::AbstractCUDATask<1,
        CudaMatrixBlockData<MatrixType, 'p', Ord>,  //inp1
        MatrixBlockData<MatrixType, 'p', Ord>       //out1
    > {
public:
    explicit CudaCopyOutGpuTask(size_t threadCount):
        hh::AbstractCUDATask<1, CudaMatrixBlockData<MatrixType, 'p', Ord>, MatrixBlockData<MatrixType, 'p', Ord>>(
            "Cuda Copy out GPU Task",
            threadCount,
            false,
            false
        ) {}

    void execute(std::shared_ptr<CudaMatrixBlockData<MatrixType, 'p', Ord>> cudaMatrixBlockData) override {
        auto productBlockData = std::make_shared<MatrixBlockData<MatrixType, 'p', Order::Column>>();
        productBlockData->rowIdx(cudaMatrixBlockData->rowIdx());
        productBlockData->colIdx(cudaMatrixBlockData->colIdx());
        productBlockData->blockSizeHeight(cudaMatrixBlockData->blockSizeHeight());
        productBlockData->blockSizeWidth(cudaMatrixBlockData->blockSizeWidth());
        productBlockData->leadingDimension(cudaMatrixBlockData->leadingDimension());
        productBlockData->blockData(new MatrixType[productBlockData->blockSizeWidth() * productBlockData->blockSizeHeight()]());
        productBlockData->fullMatrixData(productBlockData->blockData());

        checkCudaErrors(cudaMemcpyAsync(
            productBlockData->blockData(),
            cudaMatrixBlockData->blockData(),
            productBlockData->blockSizeHeight() * productBlockData->blockSizeWidth() * sizeof(MatrixType),
            cudaMemcpyDeviceToHost,
            this->stream()
        ));

        checkCudaErrors(cudaStreamSynchronize(this->stream()));

        cudaMatrixBlockData->used();
        cudaMatrixBlockData->returnToMemoryManager();

        this->addResult(productBlockData);
    }

    std::shared_ptr<hh::AbstractTask<1, CudaMatrixBlockData<MatrixType, 'p', Ord>, MatrixBlockData<MatrixType, 'p', Ord>>>
    copy() override {
        return std::make_shared<CudaCopyOutGpuTask>(this->numberThreads());
    }
};

#endif //HH3_MATMUL_CUDA_COPY_OUT_GPU_TASK_H

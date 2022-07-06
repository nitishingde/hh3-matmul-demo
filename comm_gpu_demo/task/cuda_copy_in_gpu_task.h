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


#ifndef HH3_MATMUL_CUDA_COPY_IN_GPU_TASK_H
#define HH3_MATMUL_CUDA_COPY_IN_GPU_TASK_H

#include "../data/cuda_matrix_block_data.h"

// Utils macro to convert row based intex to column based
#define IDX2C(i, j, ld) (((j)*(ld))+(i))

template<class MatrixType, char Id, Order Ord>
class CudaCopyInGpuTask: public hh::AbstractCUDATask<1,
        MatrixBlockData<MatrixType, Id, Ord>,
        CudaMatrixBlockData<MatrixType, Id, Ord>
    > {
private:
    int32_t ttl_ = 0;//FIXME int or uint?

public:
    explicit CudaCopyInGpuTask(int32_t ttl, size_t threadCount):
        ttl_(ttl),
        hh::AbstractCUDATask<1, MatrixBlockData<MatrixType, Id, Ord>, CudaMatrixBlockData<MatrixType, Id, Ord>>(
            "Cuda Copy in GPU Task",
            threadCount,
            false,
            false
        ) {}

    void execute(std::shared_ptr<MatrixBlockData<MatrixType, Id, Ord>> matrixBlockData) override {
        auto cudaMatrixBlockData = std::static_pointer_cast<CudaMatrixBlockData<MatrixType, Id, Ord>>(this->getManagedMemory());
        cudaMatrixBlockData->rowIdx(matrixBlockData->rowIdx());
        cudaMatrixBlockData->colIdx(matrixBlockData->colIdx());
        cudaMatrixBlockData->blockSizeHeight(matrixBlockData->blockSizeHeight());
        cudaMatrixBlockData->blockSizeWidth(matrixBlockData->blockSizeWidth());
        cudaMatrixBlockData->leadingDimension(cudaMatrixBlockData->blockSizeHeight());
        cudaMatrixBlockData->fullMatrixData(matrixBlockData->fullMatrixData());
        cudaMatrixBlockData->ttl(ttl_);

        if(matrixBlockData->leadingDimension() == cudaMatrixBlockData->leadingDimension()) {
            checkCudaErrors(cudaMemcpyAsync(
                cudaMatrixBlockData->blockData(),
                (const void *)matrixBlockData->blockData(),
                sizeof(MatrixType) * cudaMatrixBlockData->blockSizeHeight() * cudaMatrixBlockData->blockSizeWidth(),
                cudaMemcpyHostToDevice,
                this->stream()
            ));
        } else {
            void *pHost = matrixBlockData->fullMatrixData() + IDX2C(
                        matrixBlockData->rowIdx() * matrixBlockData->blockSizeHeight(),
                        matrixBlockData->colIdx() * matrixBlockData->blockSizeWidth(),
                        matrixBlockData->leadingDimension()
                    );
            checkCudaErrors(cublasSetMatrixAsync(
                (int)cudaMatrixBlockData->blockSizeHeight(), (int)cudaMatrixBlockData->blockSizeWidth(),
                sizeof(MatrixType),
                pHost, (int)matrixBlockData->leadingDimension(),
                cudaMatrixBlockData->blockData(), (int)cudaMatrixBlockData->leadingDimension(),
                this->stream()
            ));
        }

        cudaMatrixBlockData->recordEvent(this->stream());
        this->addResult(cudaMatrixBlockData);
    }

    std::shared_ptr<hh::AbstractTask<1, MatrixBlockData<MatrixType, Id, Ord>, CudaMatrixBlockData<MatrixType, Id, Ord>>>
    copy() override {
        return std::make_shared<CudaCopyInGpuTask>(ttl_, this->numberThreads());
    }
};

#endif //HH3_MATMUL_CUDA_COPY_IN_GPU_TASK_H

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

#include "../data/cuda_matrix_tile.h"

template<class MatrixType, char InpIdA, char InpIdB, char OutId, Order Ord>
class CudaProductTask: public hh::AbstractCUDATask<1,
        std::pair<std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>>, std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>>>,  //inp1
        CudaMatrixTile<MatrixType, OutId, Ord>                                                                                          //out1
    > {
private:
    using InputTilePair = std::pair<std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>>, std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>>>;
    cublasHandle_t handle_{};

public:
    explicit CudaProductTask(uint32_t threadCount):
        hh::AbstractCUDATask<1, InputTilePair, CudaMatrixTile<MatrixType, OutId, Ord>>(
            "Cuda Product Task",
            threadCount,
            false,
            false
        ) {}

    void initializeCuda() override {
        checkCudaErrors(cublasCreate_v2(&handle_));
        checkCudaErrors(cublasSetStream_v2(handle_, this->stream()));
    }

    void shutdownCuda() override {
        checkCudaErrors(cublasDestroy_v2(handle_));
    }

    void execute(std::shared_ptr<InputTilePair> tilePair) override {
        auto cudaTileA = tilePair->first;
        auto cudaTileB = tilePair->second;

        MatrixType alpha = 1., beta = 0.;

        auto cudaTileP = std::static_pointer_cast<CudaMatrixTile<MatrixType, OutId, Ord>>(this->getManagedMemory());
        cudaTileP->matrixTileMetaData(MatrixTileMetaData{
            .rowIdx = cudaTileA->rowIdx(),
            .colIdx = cudaTileB->colIdx(),
            .height = cudaTileA->height(),
            .width = cudaTileB->width()
        });
        cudaTileP->ttl(1);//FIXME

        cudaTileA->synchronizeEvent();
        cudaTileB->synchronizeEvent();

        if constexpr(Ord == Order::Col) {
            if constexpr(std::is_same_v<MatrixType, float>) {
                checkCudaErrors(cublasSgemm_v2(
                    handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                    cudaTileA->height(), cudaTileB->width(), cudaTileA->width(), &alpha,
                    (float *) cudaTileA->cudaMemory(), cudaTileA->leadingDimension(),
                    (float *) cudaTileB->cudaMemory(), cudaTileB->leadingDimension(), &beta,
                    (float *) cudaTileP->cudaMemory(), cudaTileP->leadingDimension()
                ));
            } else if constexpr (std::is_same_v<MatrixType, double>) {
                checkCudaErrors(cublasDgemm_v2(
                    handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                    cudaTileA->height(), cudaTileB->width(), cudaTileA->width(), &alpha,
                    (double *) cudaTileA->cudaMemory(), cudaTileA->leadingDimension(),
                    (double *) cudaTileB->cudaMemory(), cudaTileB->leadingDimension(), &beta,
                    (double *) cudaTileP->cudaMemory(), cudaTileP->leadingDimension()
                ));
            } else {
                throw std::runtime_error("Datatype not supported for cuda product task.");
            }
        }
        else {
            throw std::runtime_error("Order::Row not supported for cuda product task.");
        }
        checkCudaErrors(cudaStreamSynchronize(this->stream()));

        this->addResult(cudaTileP);
        cudaTileA->returnToMemoryManager();
        cudaTileB->returnToMemoryManager();
    }

    std::shared_ptr<hh::AbstractTask<1, InputTilePair, CudaMatrixTile<MatrixType, OutId, Ord>>>
    copy() override {
        return std::make_shared<CudaProductTask>(this->numberThreads());
    }
};

#endif //HH3_MATMUL_CUDA_PRODUCT_TASK_H

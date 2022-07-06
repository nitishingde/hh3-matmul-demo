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


#ifndef HH3_MATMUL_GPU_COMPUTATION_GRAPH_H
#define HH3_MATMUL_GPU_COMPUTATION_GRAPH_H

#include "../data/cuda_matrix_block_data.h"
#include "../state/cuda_input_block_state.h"
#include "../task/cuda_copy_in_gpu_task.h"
#include "../task/cuda_copy_out_gpu_task.h"
#include "../task/cuda_product_task.h"

template<class MatrixType, Order Ord>
class GPUComputationGraph: public hh::Graph<2,
        MatrixBlockData<MatrixType, 'a', Ord>,  //inp1
        MatrixBlockData<MatrixType, 'b', Ord>,  //inp2
        MatrixBlockData<MatrixType, 'p', Ord>   //out1
    > {
public:
    GPUComputationGraph(
            size_t M,
            size_t K,
            size_t N,
            size_t blockSize
        ): hh::Graph<2, MatrixBlockData<MatrixType, 'a', Ord>, MatrixBlockData<MatrixType, 'b', Ord>, MatrixBlockData<MatrixType, 'p', Ord>>("GPU Computation Graph") {

        using InputBlockPair = std::pair<std::shared_ptr<CudaMatrixBlockData<MatrixType, 'a', Ord>>, std::shared_ptr<CudaMatrixBlockData<MatrixType, 'b', Ord>>>;
        size_t mBlocks = std::ceil(M / blockSize) + (M % blockSize == 0 ? 0 : 1);
        size_t kBlocks = std::ceil(K / blockSize) + (K % blockSize == 0 ? 0 : 1);
        size_t nBlocks = std::ceil(N / blockSize) + (N % blockSize == 0 ? 0 : 1);

        // cuda tasks
        auto copyInATask = std::make_shared<CudaCopyInGpuTask<MatrixType, 'a', Ord>>(mBlocks, 1);
        auto copyInBTask = std::make_shared<CudaCopyInGpuTask<MatrixType, 'b', Ord>>(nBlocks, 1);
        auto productTask = std::make_shared<CudaProductTask<MatrixType, Ord>>(4);
        auto copyOutTask = std::make_shared<CudaCopyOutGpuTask<MatrixType, Ord>>(4);

        // memory managers
        // FIXME: ((mBlocks + nBlocks + productTask->numberThreads()) * blockSize * blockSize * 8) / 2^30 <= VRAM
        auto cudaMemoryManagerA = std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'a', Ord>, size_t>>(mBlocks, blockSize);
        auto cudaMemoryManagerB = std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'b', Ord>, size_t>>(nBlocks, blockSize);
        auto cudaMemoryManagerProduct = std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'p', Ord>, size_t>>(productTask->numberThreads(), blockSize);

        // connect the memory manager
        copyInATask->connectMemoryManager(cudaMemoryManagerA);
        copyInBTask->connectMemoryManager(cudaMemoryManagerB);
        productTask->connectMemoryManager(cudaMemoryManagerProduct);

        auto cudaInputBlockState = std::make_shared<CudaInputBlockState<MatrixType, Ord>>(mBlocks, kBlocks, nBlocks);
        auto cudaInputBlockStateManager = std::make_shared<hh::StateManager<2,
                CudaMatrixBlockData<MatrixType, 'a', Ord>,
                CudaMatrixBlockData<MatrixType, 'b', Ord>,
                std::pair<std::shared_ptr<CudaMatrixBlockData<MatrixType, 'a', Ord>>, std::shared_ptr<CudaMatrixBlockData<MatrixType, 'b', Ord>>>
        >>(cudaInputBlockState, "Input State Manager", false);

        this->template input<MatrixBlockData<MatrixType, 'a', Ord>>(copyInATask);
        this->template input<MatrixBlockData<MatrixType, 'b', Ord>>(copyInBTask);
        this->template edge<CudaMatrixBlockData<MatrixType, 'a', Ord>>(copyInATask, cudaInputBlockStateManager);
        this->template edge<CudaMatrixBlockData<MatrixType, 'b', Ord>>(copyInBTask, cudaInputBlockStateManager);
        this->template edge<InputBlockPair>(cudaInputBlockStateManager, productTask);
        this->template edge<CudaMatrixBlockData<MatrixType, 'p', Ord>>(productTask, copyOutTask);
        this->template output<MatrixBlockData<MatrixType, 'p', Ord>>(copyOutTask);
    }
};

#endif //HH3_MATMUL_GPU_COMPUTATION_GRAPH_H

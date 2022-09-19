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


#ifndef HH3_MATMUL_OUTER_PRODUCT_CUDA_GRAPH_H
#define HH3_MATMUL_OUTER_PRODUCT_CUDA_GRAPH_H

#include "../data/cuda_matrix_tile.h"
#include "../state/outer_product_cuda_state.h"
#include "../task/cuda_copy_in_gpu_task.h"
#include "../task/cuda_copy_out_gpu_task.h"
#include "../task/cuda_product_task.h"
#include "../task/ttl_managed_memory_recycler_task.h"
#include <hedgehog/hedgehog.h>

template<class MatrixType, char InpIdA, char InpIdB, char OutId, Order Ord>
class OuterProductCudaGraph:
    public hh::Graph<2,
        MatrixTile<MatrixType, InpIdA, Ord>,    //inp1
        MatrixTile<MatrixType, InpIdB, Ord>,    //inp2
        MatrixTile<MatrixType, OutId, Ord>      //out1
    > {
public:
    OuterProductCudaGraph(size_t mTiles, size_t kTiles, size_t nTiles, size_t tileSize):
        hh::Graph<2, MatrixTile<MatrixType, InpIdA, Ord>, MatrixTile<MatrixType, InpIdB, Ord>, MatrixTile<MatrixType, OutId, Ord>>("GPU Computation Graph") {

        using InputBlockPair  = std::pair<std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>>, std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>>>;
        size_t productThreads = 4;

        // cuda tasks
        auto copyInATask  = std::make_shared<CudaCopyInGpuTask<MatrixType, InpIdA, Ord>>(1);
        auto copyInBTask  = std::make_shared<CudaCopyInGpuTask<MatrixType, InpIdB, Ord>>(1);
        auto productTask  = std::make_shared<CudaProductTask<MatrixType, InpIdA, InpIdB, OutId, Ord>>(productThreads);
        auto copyOutTask  = std::make_shared<CudaCopyOutGpuTask<MatrixType, OutId, Ord>>(productThreads);
        auto recyclerTask = std::make_shared<TtlManagedMemoryRecyclerTask>();

        // memory managers
        // FIXME: ((mTiles + nTiles + productTask->numberThreads()) * tileSize * tileSize * 8) / 2^30 <= VRAM
        auto cudaMatrixTileAMemoryManager = std::make_shared<hh::StaticMemoryManager<CudaMatrixTile<MatrixType, InpIdA, Ord>, uint64_t>>(mTiles, tileSize);
        auto cudaMatrixTileBMemoryManager = std::make_shared<hh::StaticMemoryManager<CudaMatrixTile<MatrixType, InpIdB, Ord>, uint64_t>>(nTiles, tileSize);
        auto cudaMatrixTilePMemoryManager = std::make_shared<hh::StaticMemoryManager<CudaMatrixTile<MatrixType, OutId, Ord>, uint64_t>>(productThreads, tileSize);
        auto matrixTilePMemoryManager     = std::make_shared<hh::StaticMemoryManager<MatrixTile<MatrixType, OutId, Ord>, uint64_t>>(productThreads, tileSize);

        // connect the memory manager
        copyInATask->connectMemoryManager(cudaMatrixTileAMemoryManager);
        copyInBTask->connectMemoryManager(cudaMatrixTileBMemoryManager);
        productTask->connectMemoryManager(cudaMatrixTilePMemoryManager);
        copyOutTask->connectMemoryManager(matrixTilePMemoryManager);

        auto cudaState        = std::make_shared<OuterProductCudaState<MatrixType, InpIdA, InpIdB, Ord>>(mTiles, kTiles, nTiles);
        auto cudaStateManager = std::make_shared<hh::StateManager<2,
            CudaMatrixTile<MatrixType, InpIdA, Ord>,
            CudaMatrixTile<MatrixType, InpIdB, Ord>,
            std::pair<std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>>, std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>>>
        >>(cudaState, "Outer Product Cuda State Manager", false);

        this->template input<MatrixTile<MatrixType, InpIdA, Ord>>(copyInATask);
        this->template input<MatrixTile<MatrixType, InpIdB, Ord>>(copyInBTask);
        this->template edge<CudaMatrixTile<MatrixType, InpIdA, Ord>>(copyInATask, cudaStateManager);
        this->template edge<CudaMatrixTile<MatrixType, InpIdB, Ord>>(copyInBTask, cudaStateManager);
        this->template edge<InputBlockPair>(cudaStateManager, productTask);
        this->template edge<CudaMatrixTile<MatrixType, OutId, Ord>>(productTask, copyOutTask);
        this->template edge<TtlManagedMemory>(productTask, recyclerTask);
        this->template edge<TtlManagedMemory>(copyOutTask, recyclerTask);
        this->template output<MatrixTile<MatrixType, OutId, Ord>>(copyOutTask);
    }
};

#endif //HH3_MATMUL_OUTER_PRODUCT_CUDA_GRAPH_H

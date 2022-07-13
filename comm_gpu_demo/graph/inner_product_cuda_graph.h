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


#ifndef HH3_MATMUL_INNER_PRODUCT_CUDA_GRAPH_H
#define HH3_MATMUL_INNER_PRODUCT_CUDA_GRAPH_H

#include <hedgehog/hedgehog.h>
#include "../data/cuda_block_data.h"
#include "../state/inner_product_computation_state.h"
#include "../task/addition_task.h"
#include "../task/inner_product_task.h"
#include "../task/pair_generator_task.h"

template<class MatrixType, Order Ord>
class InnerProductCudaGraph:
    public hh::Graph<1,
        MatrixBlockData<MatrixType, 'c', Ord>,  //inp1
        MatrixBlockData<MatrixType, 'c', Ord>   //out1
    > {
public:
    explicit InnerProductCudaGraph(
        size_t M,
        size_t K,
        size_t N,
        size_t blockSize,
        const std::shared_ptr<MatrixData<MatrixType, 'a', Ord>> &matrixA,
        const std::shared_ptr<MatrixData<MatrixType, 'b', Ord>> &matrixB
    ): hh::Graph<1, MatrixBlockData<MatrixType, 'c', Ord>, MatrixBlockData<MatrixType, 'c', Ord>>(
            "Inner Product CudaGraph"
        ) {

        size_t mBlocks = std::ceil(double(M) / double(blockSize));
        size_t kBlocks = std::ceil(double(K) / double(blockSize));
        size_t nBlocks = std::ceil(double(N) / double(blockSize));
        size_t productThreads = 6;

        auto pairGenTask = std::make_shared<PairGeneratorTask<MatrixType, Ord>>(matrixA, matrixB);
        auto productTask = std::make_shared<InnerProductTask<MatrixType, Ord>>(blockSize, productThreads);
        auto additionTask = std::make_shared<AdditionTask<MatrixType, Ord>>(1);//TODO?: cannot be greater than 1, otherwise it will result in a race condition
        auto computationState = std::make_shared<InnerProductComputationState<MatrixType, Ord>>(mBlocks, kBlocks, nBlocks);
        auto computationStateManager = std::make_shared<hh::StateManager<2,
                MatrixBlockData<MatrixType, 'c', Ord>,
                MatrixBlockData<MatrixType, 'p', Ord>,
                std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>,
                MatrixBlockData<MatrixType, 'c', Ord>
            >>(computationState, "InnerProduct Computation StateManager", false);

        auto pairGenMemoryManager = std::make_shared<hh::StaticMemoryManager<MatrixBlockData<MatrixType, 'p', Ord>, size_t>>(productThreads*2, blockSize);
        pairGenTask->connectMemoryManager(pairGenMemoryManager);
        auto productMemoryManager = std::make_shared<hh::StaticMemoryManager<CudaBlockData<MatrixType>, size_t>>(3*productThreads+2, blockSize);
        productTask->connectMemoryManager(productMemoryManager);

        this->template input<MatrixBlockData<MatrixType, 'c', Ord>>(computationStateManager);
        this->template input<MatrixBlockData<MatrixType, 'c', Ord>>(pairGenTask);
        this->template edge<BlockTriplets<MatrixType, Ord>>(pairGenTask, productTask);
        this->template edge<MatrixBlockData<MatrixType, 'p', Ord>>(productTask, computationStateManager);
        this->template edge<std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>>(
            computationStateManager,
            additionTask
        );
        this->template output<MatrixBlockData<MatrixType, 'c', Ord>>(computationStateManager);
    }
};


#endif //HH3_MATMUL_INNER_PRODUCT_CUDA_GRAPH_H

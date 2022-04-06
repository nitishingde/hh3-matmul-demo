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


#include <hedgehog/hedgehog.h>
#include <random>
#include <memory>

#include "data/matrix_data.h"
#include "data/matrix_block_data.h"

#include "task/accumulate_task.h"
#include "task/addition_task.h"
#include "task/comm_task.h"
#include "task/product_task.h"
#include "task/matrix_row_traversal_task.h"
#include "task/matrix_column_traversal_task.h"

#include "state/input_block_state.h"
#include "state/output_state.h"
#include "state/partial_computation_state.h"
#include "state/partial_computation_state_manager.h"

int main(int argc, char **argv) {
  using MatrixType = float;
  constexpr Order Ord = Order::Row;
  using namespace std::chrono_literals;

    comm::MPI_GlobalLockGuard globalLockGuard(&argc, &argv, 6ms);

  // Mersenne Twister Random Generator
//  uint64_t timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
//  std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> (uint64_t) 32)};
//  std::mt19937_64 rng(ss);
//
//  // Choose your distribution depending on the type of MatrixType
//  std::uniform_real_distribution<MatrixType> unif(0, 10);
////  std::uniform_int_distribution<MatrixType> unif(0, 10);

  // Args
    const size_t
            n = 12,
            m = 4,
            p = 12,
            blockSize = 2,
    numberThreadProduct = 2,
    numberThreadAddition = 2;

    size_t
            nBlocks = 0,
            mBlocks = 0,
            pBlocks = 0;

  // Allocate matrices
  MatrixType
      *dataA = nullptr,
      *dataB = nullptr,
      *dataC = nullptr;

  // Allocate and fill the matrices' data randomly
  dataA = new MatrixType[n * m]();
  dataB = new MatrixType[m * p]();
  dataC = new MatrixType[n * p]();

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < m; ++j) {
      dataA[i * m + j] = i+j+4*comm::getMpiNodeId();
    }
  }

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < p; ++j) {
      dataB[i * p + j] = std::abs(MatrixType(j)-MatrixType(i+4*comm::getMpiNodeId()));
    }
  }

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < p; ++j) {
      dataC[i * p + j] = comm::isMpiRootPid() ? 1: 0;
    }
  }
  // Wrap them to convenient object representing the matrices
  auto subMatrixA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(n, m, blockSize, dataA);
  auto subMatrixB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(m, p, blockSize, dataB);
  auto partialMatrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(n, p, blockSize, dataC);

  nBlocks = std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1),
  mBlocks = std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1),
  pBlocks = std::ceil(p / blockSize) + (p % blockSize == 0 ? 0 : 1);

  // Graph
  auto matrixMultiplicationGraph =
      hh::Graph<3,
                MatrixData<MatrixType, 'a', Ord>, MatrixData<MatrixType, 'b', Ord>, MatrixData<MatrixType, 'c', Ord>,
                void*>
          ("Matrix Multiplication Graph");

  // Tasks
  auto taskTraversalA =
      std::make_shared<MatrixRowTraversalTask<MatrixType, 'a', Ord>>();
  auto taskTraversalB =
      std::make_shared<MatrixColumnTraversalTask<MatrixType, 'b', Ord>>();
  auto taskTraversalC =
      std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Ord>>();
  auto productTask =
      std::make_shared<ProductTask<MatrixType, Ord>>(numberThreadProduct, p);
  auto additionTask =
      std::make_shared<AdditionTask<MatrixType, Ord>>(numberThreadAddition);

  // State
  auto stateInputBlock =
      std::make_shared<InputBlockState<MatrixType, Ord>>(nBlocks, mBlocks, pBlocks);
  auto statePartialComputation =
      std::make_shared<PartialComputationState<MatrixType, Ord>>(nBlocks, pBlocks, nBlocks * mBlocks * pBlocks);
  auto stateOutput =
      std::make_shared<OutputState<MatrixType, Ord>>(nBlocks, pBlocks, mBlocks);

  // StateManager
  auto stateManagerInputBlock =
      std::make_shared<
          hh::StateManager<2, MatrixBlockData<MatrixType, 'a', Ord>, MatrixBlockData<MatrixType, 'b', Ord>,
              std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'a', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'b', Ord>>> // Pair of block as output
      >>(stateInputBlock, "Input State Manager");
  auto stateManagerPartialComputation =
      std::make_shared<PartialComputationStateManager<MatrixType, Ord>>(statePartialComputation);

  auto stateManagerOutputBlock =
      std::make_shared<hh::StateManager<1,
          MatrixBlockData<MatrixType, 'c', Ord>,
          MatrixBlockData<MatrixType, 'c', Ord>>>(stateOutput, "Output State Manager");

  // Build the graph
  matrixMultiplicationGraph.addInputs(taskTraversalA);
  matrixMultiplicationGraph.addInputs(taskTraversalB);
  matrixMultiplicationGraph.addInputs(taskTraversalC);
  matrixMultiplicationGraph.addEdges(taskTraversalA, stateManagerInputBlock);
  matrixMultiplicationGraph.addEdges(taskTraversalB, stateManagerInputBlock);
  matrixMultiplicationGraph.addEdges(taskTraversalC, stateManagerPartialComputation);
  matrixMultiplicationGraph.addEdges(stateManagerInputBlock, productTask);
  matrixMultiplicationGraph.addEdges(productTask, stateManagerPartialComputation);
  matrixMultiplicationGraph.addEdges(stateManagerPartialComputation, additionTask);
  matrixMultiplicationGraph.addEdges(additionTask, stateManagerPartialComputation);
  matrixMultiplicationGraph.addEdges(additionTask, stateManagerOutputBlock);
    uint32_t blockCount = (n/blockSize)*(p/blockSize);
    if(comm::isMpiRootPid()) {
        auto receiverTask =
                std::make_shared<ReceiverTask<MatrixBlockData<MatrixType, 'c', Ord>>>((comm::getMpiNumNodes()-1)*blockCount);
        auto accumulateTask =
                std::make_shared<AccumulateTask<MatrixType, 'c', Ord>>(n/blockSize, p/blockSize, comm::getMpiNumNodes()-1);
        matrixMultiplicationGraph.addEdges(stateManagerOutputBlock, accumulateTask);
        matrixMultiplicationGraph.addEdges(receiverTask, accumulateTask);
        matrixMultiplicationGraph.addOutputs(accumulateTask);
    } else {
        auto senderTask =
                std::make_shared<SenderTask<MatrixBlockData<MatrixType, 'c', Ord>>>(0, blockCount);
        matrixMultiplicationGraph.addEdges(stateManagerOutputBlock, senderTask);
        matrixMultiplicationGraph.addOutputs(senderTask);
    }

  // Execute the graph
  matrixMultiplicationGraph.executeGraph();

  // Push the matrices
  matrixMultiplicationGraph.pushData(subMatrixA);
  matrixMultiplicationGraph.pushData(subMatrixB);
  matrixMultiplicationGraph.pushData(partialMatrixC);

  // Notify push done
  matrixMultiplicationGraph.finishPushingData();

  // Wait for the graph to terminate
  matrixMultiplicationGraph.waitForTermination();

  //Print the result matrix
    if(comm::isMpiRootPid()) {
        std::cout << *partialMatrixC << std::endl;
    }

    matrixMultiplicationGraph.createDotFile(
            "comm_cpu_demo_node" + std::to_string(comm::getMpiNodeId()) + ".dot",
            hh::ColorScheme::EXECUTION,
            hh::StructureOptions::NONE
    );

    // Deallocate the Matrices
  delete[] dataA;
  delete[] dataB;
  delete[] dataC;

  return 0;
}
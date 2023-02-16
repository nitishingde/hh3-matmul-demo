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


#ifndef HH3_MATMUL_INNER_PRODUCT_COMPUTATION_STATE_H
#define HH3_MATMUL_INNER_PRODUCT_COMPUTATION_STATE_H

#include <atomic>
#include <hedgehog/hedgehog.h>
#include "../data/matrix_block_data.h"

template<class MatrixType, Order Ord>
class InnerProductComputationState: public hh::State<2,
        MatrixBlockData<MatrixType, 'c', Ord>,                                                                                      //inp1
        MatrixBlockData<MatrixType, 'p', Ord>,                                                                                      //inp2
        std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>,  //out1
        MatrixBlockData<MatrixType, 'c', Ord>                                                                                       //out2
> {
private:
    using BlockPair = std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>, std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>;
    std::vector<std::vector<size_t>> ttl_;
    std::vector<std::vector<std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>>> gridPartialProduct_;
    std::vector<std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>>> gridMatrixC_;

private:
    bool isPAvailable(size_t i, size_t j) { return !gridPartialProduct_[i][j].empty(); }

    bool isCAvailable(size_t i, size_t j) { return gridMatrixC_[i][j] != nullptr; }

    std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>> partialProduct(size_t i, size_t j) {
        assert(isPAvailable(i, j));
        std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>> p = gridPartialProduct_[i][j].back();
        gridPartialProduct_[i][j].pop_back();
        return p;
    }

    void partialProduct(std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>> p) {
        gridPartialProduct_[p->rowIdx()][p->colIdx()].emplace_back(p);
    }

    std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>> blockMatrixC(size_t i, size_t j) {
        assert(isCAvailable(i, j));
        return gridMatrixC_[i][j];
    }

    void blockMatrixC(std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>> c) {
        assert(!isCAvailable(c->rowIdx(), c->colIdx()));
        gridMatrixC_[c->rowIdx()][c->colIdx()] = c;
    }

public:
    explicit InnerProductComputationState(size_t mBlocks, size_t kBlocks, size_t nBlocks) {
        ttl_.resize(mBlocks, std::vector<size_t>(nBlocks, kBlocks));
        gridPartialProduct_.resize(mBlocks, std::vector<std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>>>>(nBlocks));
        gridMatrixC_.resize(mBlocks, std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>>>(nBlocks));
    }

    void execute(std::shared_ptr<MatrixBlockData<MatrixType, 'c', Ord>> blockC) override {
        auto row = blockC->rowIdx(), col = blockC->colIdx();
        if(isPAvailable(row, col)) {
            this->addResult(std::make_shared<BlockPair>(std::make_pair(blockC, partialProduct(row, col))));
            ttl_[row][col]--;
            if(ttl_[row][col] == 0) {
                this->addResult(blockMatrixC(row, col));
            }
        }
        else {
            blockMatrixC(blockC);
        }
    }

    void execute(std::shared_ptr<MatrixBlockData<MatrixType, 'p', Ord>> blockP) override {
        auto row = blockP->rowIdx(), col = blockP->colIdx();
        if(isCAvailable(row, col)) {
            this->addResult(std::make_shared<BlockPair>(std::make_pair(blockMatrixC(row, col), blockP)));
            ttl_[row][col]--;
            if(ttl_[row][col] == 0) {
                this->addResult(blockMatrixC(row, col));
            }
        }
        else {
            partialProduct(blockP);
        }
    }
};


#endif //HH3_MATMUL_INNER_PRODUCT_COMPUTATION_STATE_H

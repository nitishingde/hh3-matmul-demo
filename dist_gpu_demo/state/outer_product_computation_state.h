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


#ifndef HH3_MATMUL_OUTER_PRODUCT_COMPUTATION_STATE_H
#define HH3_MATMUL_OUTER_PRODUCT_COMPUTATION_STATE_H

#include "../data/matrix_tile.h"
#include <hedgehog/hedgehog.h>

template<class MatrixType, char InpIdC, char ProdId, char NetId, Order Ord,
    class MatrixTileC = MatrixTile<MatrixType, InpIdC, Ord>,
    class MatrixTileP = MatrixTile<MatrixType, ProdId, Ord>,
    class MatrixTileN = MatrixTile<MatrixType, NetId, Ord>
>
class OuterProductComputationState:
    public hh::AbstractState<3,
        MatrixTileC,                                                           //inp1
        MatrixTileP,                                                           //inp2
        MatrixTileN,                                                           //inp3
        std::pair<std::shared_ptr<MatrixTileC>, std::shared_ptr<MatrixTileP>>, //out1
        std::pair<std::shared_ptr<MatrixTileC>, std::shared_ptr<MatrixTileN>>  //out2
    > {
public:
    explicit OuterProductComputationState(uint64_t gridHeightResults, uint64_t gridWidthResults, int32_t ttl): gridHeightResults_(gridHeightResults), gridWidthResults_(gridWidthResults), ttl_(ttl) {
        gridPartialProductP_ = std::vector<std::vector<std::shared_ptr<MatrixTileP>>>(gridHeightResults_ * gridWidthResults_);
        //FIXME: don't allocate if unnecessary
        gridPartialProductN_ = std::vector<std::vector<std::shared_ptr<MatrixTileN>>>(gridHeightResults_ * gridWidthResults_);
        gridMatrixC_ = std::vector<std::shared_ptr<MatrixTileC>>(gridHeightResults_ * gridWidthResults_, nullptr);
    }

    void execute(std::shared_ptr<MatrixTileC> tileC) override {
        auto i = tileC->rowIdx(), j = tileC->colIdx();
        if(isPartialProductTilePAvailable(i, j)) {
            this->addResult(std::make_shared<std::pair<std::shared_ptr<MatrixTileC>, std::shared_ptr<MatrixTileP>>>(
                tileC,
                partialProductTileP(i, j)
            ));
            --ttl_;
        }
        else if(isPartialProductTileNAvailable(i, j)) {
            this->addResult(std::make_shared<std::pair<std::shared_ptr<MatrixTileC>, std::shared_ptr<MatrixTileN>>>(
                tileC,
                partialProductTileN(i, j)
            ));
            --ttl_;
        }
        else {
            matrixTileC(tileC);
        }
    }

    void execute(std::shared_ptr<MatrixTileP> tileP) override {
        tileP->ttl(1);
        auto i = tileP->rowIdx(), j = tileP->colIdx();
        if(isMatrixTileCAvailable(i, j)) {
            this->addResult(std::make_shared<std::pair<std::shared_ptr<MatrixTileC>, std::shared_ptr<MatrixTileP>>>(
                matrixTileC(i, j),
                tileP
            ));
            --ttl_;
        }
        else {
            partialProductTileP(tileP);
        }
    }

    void execute(std::shared_ptr<MatrixTileN> tileN) override {
        tileN->ttl(1);
        auto i = tileN->rowIdx(), j = tileN->colIdx();
        if(isMatrixTileCAvailable(i, j)) {
            this->addResult(std::make_shared<std::pair<std::shared_ptr<MatrixTileC>, std::shared_ptr<MatrixTileN>>>(
                matrixTileC(i, j),
                tileN
            ));
            --ttl_;
        }
        else {
            partialProductTileN(tileN);
        }
    }

    bool isDone() { return ttl_ == 0; }

private:
    bool isPartialProductTilePAvailable(uint64_t i, uint64_t j) { return gridPartialProductP_[i * gridWidthResults_ + j].size() != 0; }

    bool isPartialProductTileNAvailable(uint64_t i, uint64_t j) { return gridPartialProductN_[i * gridWidthResults_ + j].size() != 0; }

    bool isMatrixTileCAvailable(uint64_t i, uint64_t j) { return gridMatrixC_[i*gridWidthResults_ + j] != nullptr; }

    std::shared_ptr<MatrixTileP> partialProductTileP(uint64_t i, uint64_t j) {
        assert(isPartialProductTilePAvailable(i, j));
        auto tileP = gridPartialProductP_[i * gridWidthResults_ + j].back();
        gridPartialProductP_[i * gridWidthResults_ + j].pop_back();
        return tileP;
    }

    void partialProductTileP(std::shared_ptr<MatrixTileP> tileP) {
        gridPartialProductP_[tileP->rowIdx() * gridWidthResults_ + tileP->colIdx()].emplace_back(tileP);
    }

    std::shared_ptr<MatrixTileN> partialProductTileN(uint64_t i, uint64_t j) {
        assert(isPartialProductTileNAvailable(i, j));
        auto tileN = gridPartialProductN_[i * gridWidthResults_ + j].back();
        gridPartialProductN_[i * gridWidthResults_ + j].pop_back();
        return tileN;
    }

    void partialProductTileN(std::shared_ptr<MatrixTileN> tileN) {
        gridPartialProductN_[tileN->rowIdx() * gridWidthResults_ + tileN->colIdx()].emplace_back(tileN);
    }

    std::shared_ptr<MatrixTileC> matrixTileC(uint64_t i, uint64_t j) {
        assert(isMatrixTileCAvailable(i, j));
        auto tileC = gridMatrixC_[i*gridWidthResults_ + j];
        gridMatrixC_[i*gridWidthResults_ + j] = nullptr;
        return tileC;
    }

    void matrixTileC(std::shared_ptr<MatrixTileC> tileC) {
        assert(!isMatrixTileCAvailable(tileC->rowIdx(), tileC->colIdx()));
        gridMatrixC_[tileC->rowIdx() * gridWidthResults_ + tileC->colIdx()] = tileC;
    }

private:
    uint64_t gridHeightResults_                                                 = 0;
    uint64_t gridWidthResults_                                                  = 0;
    std::vector<std::vector<std::shared_ptr<MatrixTileP>>> gridPartialProductP_ = {};
    std::vector<std::vector<std::shared_ptr<MatrixTileN>>> gridPartialProductN_ = {};
    std::vector<std::shared_ptr<MatrixTileC>> gridMatrixC_                      = {};
    int32_t ttl_                                                                = 0;
};

#endif //HH3_MATMUL_OUTER_PRODUCT_COMPUTATION_STATE_H

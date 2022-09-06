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

template<class MatrixType, char InpIdC, char ProdId, Order Ord>
class OuterProductComputationState:
    public hh::State<2,
        MatrixTile<MatrixType, InpIdC, Ord>,                                                                                    //inp1
        MatrixTile<MatrixType, ProdId, Ord>,                                                                                    //inp2
        std::pair<std::shared_ptr<MatrixTile<MatrixType, InpIdC, Ord>>, std::shared_ptr<MatrixTile<MatrixType, ProdId, Ord>>>   //out1
    > {
public:
    explicit OuterProductComputationState(uint32_t gridHeightResults, uint32_t gridWidthResults, int32_t ttl): gridHeightResults_(gridHeightResults), gridWidthResults_(gridWidthResults), ttl_(ttl) {
        gridPartialProduct_ = std::vector<std::vector<std::shared_ptr<MatrixTile<MatrixType, ProdId, Ord>>>>(gridHeightResults_ * gridWidthResults_);
        gridMatrixC_ = std::vector<std::shared_ptr<MatrixTile<MatrixType, InpIdC, Ord>>>(gridHeightResults_ * gridWidthResults_, nullptr);
    }

    void execute(std::shared_ptr<MatrixTile<MatrixType, InpIdC, Ord>> tileC) override {
        auto i = tileC->rowIdx(), j = tileC->colIdx();
        if(isPartialProductTilePAvailable(i, j)) {
            this->addResult(std::make_shared<std::pair<std::shared_ptr<MatrixTile<MatrixType, InpIdC, Ord>>, std::shared_ptr<MatrixTile<MatrixType, ProdId, Ord>>>>(
                tileC,
                partialProductTileP(i, j)
            ));
            --ttl_;
        }
        else {
            matrixTileC(tileC);
        }
    }

    void execute(std::shared_ptr<MatrixTile<MatrixType, ProdId, Ord>> tileP) override {
        tileP->ttl(1);
        auto i = tileP->rowIdx(), j = tileP->colIdx();
        if(isMatrixTileCAvailable(i, j)) {
            this->addResult(std::make_shared<std::pair<std::shared_ptr<MatrixTile<MatrixType, InpIdC, Ord>>, std::shared_ptr<MatrixTile<MatrixType, ProdId, Ord>>>>(
                matrixTileC(i, j),
                tileP
            ));
            --ttl_;
        }
        else {
            partialProductTileP(tileP);
        }
    }

    bool isDone() { return ttl_ == 0; }

private:
    bool isPartialProductTilePAvailable(uint32_t i, uint32_t j) { return gridPartialProduct_[i*gridWidthResults_ + j].size() != 0; }

    bool isMatrixTileCAvailable(uint32_t i, uint32_t j) { return gridMatrixC_[i*gridWidthResults_ + j] != nullptr; }

    std::shared_ptr<MatrixTile<MatrixType, ProdId, Ord>> partialProductTileP(uint32_t i, uint32_t j) {
        assert(isPartialProductTilePAvailable(i, j));
        auto tileP = gridPartialProduct_[i*gridWidthResults_ + j].back();
        gridPartialProduct_[i*gridWidthResults_ + j].pop_back();
        return tileP;
    }

    void partialProductTileP(std::shared_ptr<MatrixTile<MatrixType, ProdId, Ord>> tileP) {
        gridPartialProduct_[tileP->rowIdx()*gridWidthResults_ + tileP->colIdx()].emplace_back(tileP);
    }

    std::shared_ptr<MatrixTile<MatrixType, InpIdC, Ord>> matrixTileC(uint32_t i, uint32_t j) {
        assert(isMatrixTileCAvailable(i, j));
        auto tileC = gridMatrixC_[i*gridWidthResults_ + j];
        gridMatrixC_[i*gridWidthResults_ + j] = nullptr;
        return tileC;
    }

    void matrixTileC(std::shared_ptr<MatrixTile<MatrixType, InpIdC, Ord>> tileC) {
        assert(!isMatrixTileCAvailable(tileC->rowIdx(), tileC->colIdx()));
        gridMatrixC_[tileC->rowIdx() * gridWidthResults_ + tileC->colIdx()] = tileC;
    }

private:
    uint32_t gridHeightResults_                                                                        = 0;
    uint32_t gridWidthResults_                                                                         = 0;
    std::vector<std::vector<std::shared_ptr<MatrixTile<MatrixType, ProdId, Ord>>>> gridPartialProduct_ = {};
    std::vector<std::shared_ptr<MatrixTile<MatrixType, InpIdC, Ord>>> gridMatrixC_                     = {};
    int32_t ttl_                                                                                       = 0;
};

#endif //HH3_MATMUL_OUTER_PRODUCT_COMPUTATION_STATE_H

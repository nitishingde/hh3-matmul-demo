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


#ifndef HH3_MATMUL_OUTER_PRODUCT_OUTPUT_STATE_H
#define HH3_MATMUL_OUTER_PRODUCT_OUTPUT_STATE_H

#include "../data/matrix_tile.h"
#include <hedgehog/hedgehog.h>

template<class MatrixType, char OutIdC, Order Ord>
class OuterProductOutputState:
    public hh::AbstractState<1,
        MatrixTile<MatrixType, OutIdC, Ord>,  //inp1
        MatrixTile<MatrixType, OutIdC, Ord>   //out1
    > {
public:
    OuterProductOutputState(uint64_t mTiles, uint64_t nTiles, int32_t ttl):
        ttl_(std::vector<int32_t>(mTiles*nTiles, ttl)),
        mTiles_(mTiles), nTiles_(nTiles) {}

    virtual ~OuterProductOutputState() = default;

    void execute(std::shared_ptr<MatrixTile<MatrixType, OutIdC, Ord>> matrixTile) override {
        uint64_t idx = matrixTile->rowIdx()*nTiles_ + matrixTile->colIdx();
        --ttl_[idx];
        if(ttl_[idx] == 0) {
            this->addResult(matrixTile);
        }
    }

    friend std::ostream &operator<<(std::ostream &os, OuterProductOutputState const &state) {
        for(uint64_t i = 0; i < state.gridHeightTTL_; ++i){
            for(uint64_t j = 0; j < state.gridWidthTTL_; ++j) {
                os << state.ttl_[i*state.gridWidthTTL_ + j] << " ";
            }
            os << std::endl;
        }
        return os;
    }

private:
    std::vector<int32_t>  ttl_ = {};
    uint64_t mTiles_           = 0;
    uint64_t nTiles_           = 0;
};

#endif //HH3_MATMUL_OUTER_PRODUCT_OUTPUT_STATE_H

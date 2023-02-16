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


#ifndef HH3_MATMUL_ACCUMULATE_TASK_H
#define HH3_MATMUL_ACCUMULATE_TASK_H

#include "../data/matrix_order.h"
#include "../data/matrix_tile.h"
#include <hedgehog/hedgehog.h>

template<class MatrixType, char InpIdC, char ProdId, char NetId, Order Ord,
    class MatrixTileC = MatrixTile<MatrixType, InpIdC, Ord>,
    class MatrixTileP = MatrixTile<MatrixType, ProdId, Ord>,
    class MatrixTileN = MatrixTile<MatrixType, NetId, Ord>
>
class AccumulateTask:
    public hh::AbstractTask<2,
        std::pair<std::shared_ptr<MatrixTileC>, std::shared_ptr<MatrixTileP>>, //inp1
        std::pair<std::shared_ptr<MatrixTileC>, std::shared_ptr<MatrixTileN>>, //inp2
        MatrixTileC                                                            //out1
    > {
private:
    using InputTilePair1 = std::pair<std::shared_ptr<MatrixTileC>, std::shared_ptr<MatrixTileP>>;
    using InputTilePair2 = std::pair<std::shared_ptr<MatrixTileC>, std::shared_ptr<MatrixTileN>>;

public:
    explicit AccumulateTask(uint32_t threadCount):
        hh::AbstractTask<2,
            InputTilePair1,
            InputTilePair2,
            MatrixTile<MatrixType, InpIdC, Ord>
        >("Accumulate Task", threadCount, false) {}

    void execute(std::shared_ptr<InputTilePair1> tilePair) override {
        auto tileC = tilePair->first;
        auto tileP = tilePair->second;

        if constexpr(std::is_same_v<MatrixTileP, UnifiedMatrixTile<MatrixType, ProdId, Ord>>) {
            tileP->synchronizeEvent();
        }

        assert(tileC->width() == tileP->width());
        assert(tileC->height() == tileP->height());

        accumulate(tileC->data(), tileC->leadingDimension(), tileP->data(), tileP->leadingDimension(), tileC->width(), tileC->height());

        this->addResult(tileC);
        tileP->returnToMemoryManager();
    }

    void execute(std::shared_ptr<InputTilePair2> tilePair) override {
        auto tileC = tilePair->first;
        auto tileP = tilePair->second;

        assert(tileC->width() == tileP->width());
        assert(tileC->height() == tileP->height());

        accumulate(tileC->data(), tileC->leadingDimension(), tileP->data(), tileP->leadingDimension(), tileC->width(), tileC->height());

        this->addResult(tileC);
        tileP->returnToMemoryManager();
    }

    std::shared_ptr<hh::AbstractTask<2, InputTilePair1, InputTilePair2, MatrixTile<MatrixType, InpIdC, Ord>>>
    copy() override {
        return std::make_shared<AccumulateTask>(this->numberThreads());
    };

private:
    void accumulate(MatrixType *dataC, uint64_t leadingDimensionC, MatrixType *dataP, uint64_t leadingDimensionP, uint64_t tileWidth, uint64_t tileHeight) {
        if constexpr(Ord == Order::Col) {
            if(leadingDimensionC == tileHeight and leadingDimensionP == tileHeight) {
                std::transform(dataC, dataC+tileWidth*tileHeight, dataP, dataC, std::plus<MatrixType>());
                return;
            }
            for(uint64_t j = 0; j < tileWidth; ++j) {
                for(uint64_t i = 0; i < tileHeight; ++i) {
                    dataC[j*leadingDimensionC + i] += dataP[j*leadingDimensionP + i];
                }
            }
        }
        else {
            throw std::runtime_error("Order::Row not supported for accumulate task.");
        }
    }
};

#endif //HH3_MATMUL_ACCUMULATE_TASK_H

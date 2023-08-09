#ifndef HH3_MATMUL_STATES
#define HH3_MATMUL_STATES

#include "data.h"

template<typename MatrixType, char IdA, char IdB, char IdC>
class InputState: public hh::AbstractState<
        1,
        std::tuple<std::shared_ptr<MatrixContainer<MatrixType, IdA>>, std::shared_ptr<MatrixContainer<MatrixType, IdB>>, std::shared_ptr<MatrixContainer<MatrixType, IdC>>>,
        MatrixContainer<MatrixType, IdA>,
        MatrixContainer<MatrixType, IdB>,
        MatrixContainer<MatrixType, IdC>,
        std::tuple<std::shared_ptr<MatrixContainer<MatrixType, IdA>>, std::shared_ptr<MatrixContainer<MatrixType, IdB>>, std::shared_ptr<MatrixContainer<MatrixType, IdC>>>
    > {
private:
    using MatrixA = MatrixContainer<MatrixType, IdA>;
    using MatrixB = MatrixContainer<MatrixType, IdB>;
    using MatrixC = MatrixContainer<MatrixType, IdC>;
    using Triplet = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>;

public:
    void execute(std::shared_ptr<Triplet> triplet) override {
        auto matrixA = std::get<std::shared_ptr<MatrixA>>(*triplet);
        auto matrixB = std::get<std::shared_ptr<MatrixB>>(*triplet);
        auto matrixC = std::get<std::shared_ptr<MatrixC>>(*triplet);
        assert(matrixA->matrixNumColTiles() == matrixB->matrixNumRowTiles());
        assert(matrixA->matrixNumRowTiles() == matrixC->matrixNumRowTiles());
        assert(matrixB->matrixNumColTiles() == matrixC->matrixNumColTiles());

        this->addResult(matrixA);
        this->addResult(matrixB);
        this->addResult(matrixC);
        this->addResult(triplet);
    };
};

template<typename MatrixType, char IdA, char IdB, char IdC, char IdP>
class OuterProductComputationState: public hh::AbstractState<
        3,
        MatrixContainer<MatrixType, IdC>,
        MatrixTile<MatrixType, IdC>,
        MatrixTile<MatrixType, IdP>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdC>>, std::shared_ptr<MatrixTile<MatrixType, IdP>>>,
        MatrixTile<MatrixType, IdC>
    > {
private:
    template<class GridT>
    using Grid    = std::vector<std::vector<GridT>>;
    using MatrixC = MatrixContainer<MatrixType, IdC>;
    using TileC   = MatrixTile<MatrixType, IdC>;
    using TileP   = MatrixTile<MatrixType, IdP>;
    using Pair    = std::tuple<std::shared_ptr<TileC>, std::shared_ptr<TileP>>;

public:
    explicit OuterProductComputationState(const int64_t MT, const int64_t KT, const int64_t NT): KT_(KT) {
        gridP_.resize(MT, std::vector<std::vector<std::shared_ptr<TileP>>>(NT, std::vector<std::shared_ptr<TileP>>{}));
    }

    void execute(std::shared_ptr<MatrixC> matrixC) override {
        assert(matrixC != nullptr);
        matrixC_ = matrixC;

        ttl_ = 0;
        for(int64_t rowIdx = 0; rowIdx < matrixC->matrixNumRowTiles(); ++rowIdx) {
            for(int64_t colIdx = 0; colIdx < matrixC->matrixNumColTiles(); ++colIdx) {
                if(auto tileC = matrixC->tile(rowIdx, colIdx); tileC != nullptr) {
                    tileC->memoryState(MemoryState::SHARED);
                    tileC->ttl(KT_);
                    ttl_ += KT_;
                }
            }
        }
    }

    void execute(std::shared_ptr<TileC> tileC) override {
        tileC->memoryState(MemoryState::SHARED);

        ttl_--;
        if(ttl_ == 0) {
            matrixC_ = nullptr;
        }

        tileC->used();
        if(tileC->canBeRecycled()) {
            this->addResult(tileC);
            return;
        }

        auto rowIdx = tileC->rowIdx(), colIdx = tileC->colIdx();
        if(!gridP_[rowIdx][colIdx].empty()) {
            auto tileP = gridP_[rowIdx][colIdx].back();
            tileC->memoryState(MemoryState::ON_HOLD);
            this->addResult(std::make_shared<Pair>(std::make_tuple(tileC, tileP)));
            gridP_[rowIdx][colIdx].pop_back();
            return;
        }
    }

    void execute(std::shared_ptr<TileP> tileP) override {
        assert(matrixC_ != nullptr);
        auto rowIdx = tileP->rowIdx(), colIdx = tileP->colIdx();
        if(auto tileC = matrixC_->tile(rowIdx, colIdx); tileC != nullptr and tileC->memoryState() == MemoryState::SHARED) {
            tileC->memoryState(MemoryState::ON_HOLD);
            this->addResult(std::make_shared<Pair>(std::make_tuple(tileC, tileP)));
        }
        else {
            gridP_[rowIdx][colIdx].emplace_back(tileP);
        }
    }

    bool isDone() {
        return ttl_ == 0;
    }

private:
    std::shared_ptr<MatrixC>                  matrixC_ = nullptr;
    int64_t                                   KT_      = 0;
    int64_t                                   ttl_     = -1;
    Grid<std::vector<std::shared_ptr<TileP>>> gridP_   = {};
};

template<typename MatrixType, char IdA, char IdB, char IdC, char IdP>
class OuterProductComputationStateManager: public hh::StateManager<
        3,
        MatrixContainer<MatrixType, IdC>,
        MatrixTile<MatrixType, IdC>,
        MatrixTile<MatrixType, IdP>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdC>>, std::shared_ptr<MatrixTile<MatrixType, IdP>>>,
        MatrixTile<MatrixType, IdC>
    > {
public:
    explicit OuterProductComputationStateManager(std::shared_ptr<OuterProductComputationState<MatrixType, IdA, IdB, IdC, IdP>> state):
        hh::StateManager<3, MatrixContainer<MatrixType, IdC>, MatrixTile<MatrixType, IdC>, MatrixTile<MatrixType, IdP>, std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdC>>, std::shared_ptr<MatrixTile<MatrixType, IdP>>>, MatrixTile<MatrixType, IdC>>(
            state,
            "OuterProductComputationStateManager",
            false
        ) {}

    [[nodiscard]] bool canTerminate() const override {
        this->state()->lock();
        auto ret = std::dynamic_pointer_cast<OuterProductComputationState<MatrixType, IdA, IdB, IdC, IdP>>(this->state())->isDone();
        this->state()->unlock();
        return ret;
    }
};


#endif //HH3_MATMUL_STATES

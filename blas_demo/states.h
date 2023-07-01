#ifndef HH3_MATMUL_STATES
#define HH3_MATMUL_STATES

#include <set>
#include "data.h"

template<typename MatrixType, char IdA, char IdB, char IdC>
class InputState: public hh::AbstractState<
        3,
        MatrixContainer<MatrixType, IdC>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        DbRequest<IdA>,
        DbRequest<IdB>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        MatrixTile<MatrixType, IdC>
    > {
private:
    using MatrixC = MatrixContainer<MatrixType, IdC>;
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using TileC   = MatrixTile<MatrixType, IdC>;

public:
    explicit InputState(const int32_t KT):
        hh::AbstractState<3, MatrixC, TileA, TileB, DbRequest<IdA>, DbRequest<IdB>, TileA, TileB, TileC>(), KT_(KT) {}

    void execute(std::shared_ptr<MatrixC> matrixC) override {
        isStarted_ = true;
        std::set<int32_t> rows, cols;
        for(int32_t col = 0; col < matrixC->matrixNumColTiles(); ++col) {
            for(int32_t row = 0; row < matrixC->matrixNumRowTiles(); ++row) {
                if(auto tileC = matrixC->tile(row, col); tileC != nullptr) {
                    this->addResult(tileC);
                    rows.emplace(row);
                    cols.emplace(col);
                }
            }
        }

        MT_ = rows.size();
        NT_ = cols.size();

        for(int32_t k = 0; k < KT_; ++k) {
            for(const auto row: rows) {
                this->addResult(std::make_shared<DbRequest<IdA>>(row, k));
                reqCount_++;
            }

            for(const auto col: cols) {
                this->addResult(std::make_shared<DbRequest<IdB>>(k, col));
                reqCount_++;
            }
        }
        this->addResult(std::make_shared<DbRequest<IdA>>(-1, -1, true));
        this->addResult(std::make_shared<DbRequest<IdB>>(-1, -1, true));
    }

    void execute(std::shared_ptr<TileA> tileA) override {
        assert(NT_ != 0);
        tileA->ttl(NT_);
        this->addResult(tileA);
        reqCount_--;
    }

    void execute(std::shared_ptr<TileB> tileB) override {
        assert(MT_ != 0);
        tileB->ttl(MT_);
        this->addResult(tileB);
        reqCount_--;
    }

    [[nodiscard]] bool isDone() const {
//        printf("[node %d] reqCount_ %d\n", getNodeId(), reqCount_);FIXME
        return isStarted_ and reqCount_ == 0;
    }

private:
    int32_t MT_        = 0;
    int32_t KT_        = 0;
    int32_t NT_        = 0;
    int32_t reqCount_  = 0;
    bool    isStarted_ = false;
};

// TODO
template<typename MatrixType, char IdA, char IdB, char IdC>
class InputStateManager: public hh::StateManager<
        3,
        MatrixContainer<MatrixType, IdC>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        DbRequest<IdA>,
        DbRequest<IdB>,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        MatrixTile<MatrixType, IdC>
    >{
    using MatrixC = MatrixContainer<MatrixType, IdC>;
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using TileC   = MatrixTile<MatrixType, IdC>;

public:
    explicit InputStateManager(const std::shared_ptr<InputState<MatrixType, IdA, IdB, IdC>> &state):
    hh::StateManager<3, MatrixC, TileA, TileB, DbRequest<IdA>, DbRequest<IdB>, TileA, TileB, TileC>(state, "Input StateManager", false) {}

    [[nodiscard]] bool canTerminate() const override {
        this->state()->lock();
        auto ret = std::dynamic_pointer_cast<InputState<MatrixType, IdA, IdB, IdC>>(this->state())->isDone();
        this->state()->unlock();
        return ret;
    }
};

template<typename MatrixType, char IdA, char IdB, char IdC>
class ComputationState: public hh::AbstractState<
        4,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        MatrixTile<MatrixType, IdC>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>, std::shared_ptr<MatrixTile<MatrixType, IdC>>>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>, std::shared_ptr<MatrixTile<MatrixType, IdC>>>,
        MatrixTile<MatrixType, IdC>
    >{
private:
    template<class GridT>
    using Grid    = std::vector<std::vector<GridT>>;
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using TileC   = MatrixTile<MatrixType, IdC>;
    using Triplet = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>, std::shared_ptr<TileC>>;

public:
    ComputationState(const int32_t MT, const int32_t KT, const int32_t NT):
        hh::AbstractState<4, TileA, TileB, TileC, Triplet, Triplet, TileC>(), MT_(MT), KT_(KT), NT_(NT) {
        gridA_.resize(MT, std::vector<std::shared_ptr<TileA>>(KT, nullptr));
        gridB_.resize(KT, std::vector<std::shared_ptr<TileB>>(NT, nullptr));
        gridC_.resize(MT, std::vector<std::shared_ptr<TileC>>(NT, nullptr));
    }

    void execute(std::shared_ptr<TileA> tileA) override {
        auto i = tileA->rowIdx(), k = tileA->colIdx();
        gridA_[i][k] = tileA;

        for(int32_t j = 0; j < NT_; ++j) {
            auto tileB = gridB_[k][j];
            if(tileB != nullptr) {
                workQueue_.emplace_back(std::make_tuple(i, j, k));
            }
        }
        trigger();
    }

    void execute(std::shared_ptr<TileB> tileB) override {
        auto k = tileB->rowIdx(), j = tileB->colIdx();
        gridB_[k][j] = tileB;

        for(int32_t i = 0; i < MT_; ++i) {
            auto tileA = gridA_[i][k];
            if(tileA != nullptr) {
                workQueue_.emplace_back(std::make_tuple(i, j, k));
            }
        }
        trigger();
    }

    void execute(std::shared_ptr<TileC> tileC) override {
        isStarted_ = true;
        ttl_++;
        auto rowIdx = tileC->rowIdx(), colIdx = tileC->colIdx();
        tileC->ttl(KT_);
        gridC_[rowIdx][colIdx] = tileC;
        trigger();
    }

    void execute(std::shared_ptr<Triplet> triplet) override {
        auto tileA = std::get<std::shared_ptr<TileA>>(*triplet);
        auto tileB = std::get<std::shared_ptr<TileB>>(*triplet);
        auto tileC = std::get<std::shared_ptr<TileC>>(*triplet);

        tileA->used();
//        printf("[node %d] tileA(%d, %d).ttl %d\n", getNodeId(), tileA->rowIdx(), tileA->colIdx(), tileA->ttl());
        if(tileA->canBeRecycled()) {
            gridA_[tileA->rowIdx()][tileA->colIdx()] = nullptr;
            if(tileA->isMemoryManagerConnected()) {
//                printf("a>> [node %d] tileA(%d, %d).ttl %d\n", getNodeId(), tileA->rowIdx(), tileA->colIdx(), tileA->ttl());
                tileA->returnToMemoryManager();
            }
        }

        tileB->used();
//        printf("[node %d] tileB(%d, %d).ttl %d\n", getNodeId(), tileB->rowIdx(), tileB->colIdx(), tileB->ttl());
        if(tileB->canBeRecycled()) {
            gridB_[tileB->rowIdx()][tileB->colIdx()] = nullptr;
            if(tileB->isMemoryManagerConnected()) {
//                printf("b>> [node %d] tileB(%d, %d).ttl %d\n", getNodeId(), tileB->rowIdx(), tileB->colIdx(), tileB->ttl());
                tileB->returnToMemoryManager();
            }
        }

        tileC->used();
        if(tileC->canBeRecycled()) {
            gridC_[tileC->rowIdx()][tileC->colIdx()] = nullptr;
//            printf("[node %d] tileC(%d, %d).ttl %d\n", getNodeId(), tileC->rowIdx(), tileC->colIdx(), tileC->ttl());
            ttl_--;
            this->addResult(tileC);
        } else {
            gridC_[tileC->rowIdx()][tileC->colIdx()] = tileC;
            trigger();
        }
    }

    [[nodiscard]] bool isDone() const {
//        printf("[node %d] workQueue_ %zu, ttl_ %d\n", getNodeId(), workQueue_.size(), ttl_);FIXME
        return isStarted_ and workQueue_.empty() and ttl_ == 0;
    }
private:
    void trigger() {
        for(auto it = workQueue_.begin(); it != workQueue_.end();) {
            auto [i, j, k] = *it;
            auto tileA = gridA_[i][k];
            auto tileB = gridB_[k][j];
            auto tileC = gridC_[i][j];
            if(tileC) {
                this->addResult(std::make_shared<Triplet>(std::make_tuple(tileA, tileB, tileC)));
                gridC_[i][j] = nullptr;
                it = workQueue_.erase(it);
            }
            else {
                it++;
            }
        }
    }

private:
    std::list<std::tuple<int32_t, int32_t, int32_t>> workQueue_ = {};
    Grid<std::shared_ptr<TileA>> gridA_     = {};
    Grid<std::shared_ptr<TileB>> gridB_     = {};
    Grid<std::shared_ptr<TileC>> gridC_     = {};
    int32_t                      ttl_       = 0;
    bool                         isStarted_ = false;
    int32_t                      MT_        = 0;
    int32_t                      KT_        = 0;
    int32_t                      NT_        = 0;
};

template<typename MatrixType, char IdA, char IdB, char IdC>
class ComputationStateManager: public hh::StateManager<
        4,
        MatrixTile<MatrixType, IdA>,
        MatrixTile<MatrixType, IdB>,
        MatrixTile<MatrixType, IdC>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>, std::shared_ptr<MatrixTile<MatrixType, IdC>>>,
        std::tuple<std::shared_ptr<MatrixTile<MatrixType, IdA>>, std::shared_ptr<MatrixTile<MatrixType, IdB>>, std::shared_ptr<MatrixTile<MatrixType, IdC>>>,
        MatrixTile<MatrixType, IdC>
>{
private:
    using TileA   = MatrixTile<MatrixType, IdA>;
    using TileB   = MatrixTile<MatrixType, IdB>;
    using TileC   = MatrixTile<MatrixType, IdC>;
    using Triplet = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>, std::shared_ptr<TileC>>;

public:
    explicit ComputationStateManager(const std::shared_ptr<ComputationState<MatrixType, IdA, IdB, IdC>> &state):
    hh::StateManager<4, TileA, TileB, TileC, Triplet, Triplet, TileC>(state, "Computation StateManager", false) {}

    [[nodiscard]] bool canTerminate() const override {
        this->state()->lock();
        auto ret = std::dynamic_pointer_cast<ComputationState<MatrixType, IdA, IdB, IdC>>(this->state())->isDone();
        this->state()->unlock();
        return ret;
    }
};

#endif //HH3_MATMUL_STATES

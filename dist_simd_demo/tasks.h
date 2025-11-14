#ifndef HH3_MATMUL_TASKS_H
#define HH3_MATMUL_TASKS_H

#include <queue>
#include <hedgehog/hedgehog.h>
#include "common_tasks.h"

template<std::floating_point Float, char IdA, char IdB, char IdC
    , class MatrixA = MatrixContainer<Float, IdA>
    , class MatrixB = MatrixContainer<Float, IdB>
    , class MatrixC = MatrixContainer<Float, IdC>
    , class TileA   = MatrixTile<Float, IdA>
    , class TileB   = MatrixTile<Float, IdB>
    , class TileC   = MatrixTile<Float, IdC>
    , class TileTriplet = std::tuple<std::shared_ptr<TileA>, std::shared_ptr<TileB>, std::shared_ptr<TileC>>
    , class MatrixTriplet = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>, std::shared_ptr<MatrixC>>
>
class JobScheduler final: public hh::AbstractTask<4, MatrixTriplet, TileA, TileB, TileTriplet, TileTriplet, TileC> {
private:
    struct Duplet {
        std::shared_ptr<TileA> tileA_ = nullptr;
        std::shared_ptr<TileB> tileB_ = nullptr;

        explicit Duplet(const std::shared_ptr<TileA> tileA, const std::shared_ptr<TileB> tileB):
            tileA_(tileA), tileB_(tileB) {

            priority_ += static_cast<int8_t>(tileA->isMemoryManagerConnected());
            priority_ += static_cast<int8_t>(tileB->isMemoryManagerConnected());
        }

        // higher the value, higher the priority
        bool operator<(const Duplet& other) const {
            return priority_ < other.priority_;
        }

    private:
        int8_t priority_ = 0;
    };

    struct TupleHash {
        template<std::integral T1, std::integral T2>
        size_t operator()(const std::tuple<T1, T2>& t) const {
            size_t h1 = std::hash<T1>{}(std::get<0>(t));
            size_t h2 = std::hash<T2>{}(std::get<1>(t));
            return (h1 << 32) | h2;
        }
    };
public:
    using base = hh::AbstractTask<4, MatrixTriplet, TileA, TileB, TileTriplet, TileTriplet, TileC>;

    explicit JobScheduler(const std::string &name = "JobScheduler"): base(name, 1, false) {}

    void execute(std::shared_ptr<MatrixTriplet> triplet) override {
        auto [matrixA, matrixB, matrixC] = *triplet;
        matrixA_ = matrixA;
        matrixB_ = matrixB;
        matrixC_ = matrixC;

        const auto [pNodeId, qNodeId]   = getGridNodeId();
        const auto [pNodeDim, qNodeDim] = getGridDim();

        ttl_ = 0;
        for(int64_t mt = pNodeId; mt < matrixC_->matrixNumRowTiles(); mt += pNodeDim) {
            for(int64_t nt = qNodeId; nt < matrixC_->matrixNumColTiles(); nt += qNodeDim) {
                auto tileC = matrixC_->tile(mt, nt);
                assert(tileC != nullptr);

                const auto KT = matrixA_->matrixNumColTiles();
                ttl_ += KT;
                tileC->ttl(KT);
                const auto key = std::make_tuple(mt, nt);
                auto [it, isInserted] = queueMapping_.emplace(key, std::make_shared<std::priority_queue<Duplet>>());
                assert(isInserted);
                auto &priorityQueue = it->second;
                for(int64_t kt = 0; kt < KT; ++kt) {
                    auto tileA = matrixA_->tile(mt, kt);
                    auto tileB = matrixB_->tile(kt, nt);
                    if(tileA == nullptr or tileB == nullptr) continue;
                    priorityQueue->emplace(tileA, tileB);
                }

                if(priorityQueue->empty()) continue;
                const auto &duplet = priorityQueue->top();
                this->addResult(std::make_shared<TileTriplet>(std::make_tuple(duplet.tileA_, duplet.tileB_, tileC)));
                priorityQueue->pop();
                matrixC_->tile(mt, nt, nullptr);
            }
        }
    }

    void execute(std::shared_ptr<TileA> tileA) override {
        const auto mt = tileA->rowIdx();
        const auto kt = tileA->colIdx();

        const auto [pNodeId, qNodeId]   = getGridNodeId();
        const auto [pNodeDim, qNodeDim] = getGridDim();
        tileA->ttl((matrixB_->matrixNumColTiles() +qNodeDim-1 - qNodeId)/qNodeDim);
        matrixA_->tile(mt, kt, tileA);
        for(int64_t nt = qNodeId; nt < matrixB_->matrixNumColTiles(); nt += qNodeDim) {
            const auto key = std::make_tuple(mt, nt);
            assert(queueMapping_.contains(key));

            auto tileB = matrixB_->tile(kt, nt);
            if(tileB == nullptr) continue;

            auto tileC = matrixC_->tile(mt, nt);
            if(tileC != nullptr) {
                this->addResult(std::make_shared<TileTriplet>(std::make_tuple(tileA, tileB, tileC)));
                matrixC_->tile(mt, nt, nullptr);
            }
            else {
                auto &priorityQueue = queueMapping_.at(key);
                priorityQueue->emplace(tileA, tileB);
            }
        }
    }

    void execute(std::shared_ptr<TileB> tileB) override {
        const auto kt = tileB->rowIdx();
        const auto nt = tileB->colIdx();

        const auto [pNodeId, qNodeId]   = getGridNodeId();
        const auto [pNodeDim, qNodeDim] = getGridDim();
        tileB->ttl((matrixB_->matrixNumColTiles() +pNodeDim-1 - pNodeId)/pNodeDim);
        matrixB_->tile(kt, nt, tileB);
        for(int64_t mt = pNodeId; mt < matrixA_->matrixNumRowTiles(); mt += pNodeDim) {
            const auto key = std::make_tuple(mt, nt);
            assert(queueMapping_.contains(key));

            auto tileA = matrixA_->tile(mt, kt);
            if(tileA == nullptr) continue;

            auto tileC = matrixC_->tile(mt, nt);
            if(tileC != nullptr) {
                this->addResult(std::make_shared<TileTriplet>(std::make_tuple(tileA, tileB, tileC)));
                matrixC_->tile(mt, nt, nullptr);
            }
            else {
                auto &priorityQueue = queueMapping_.at(key);
                priorityQueue->emplace(tileA, tileB);
            }
        }
    }

    void execute(std::shared_ptr<TileTriplet> triplet) override {
        ttl_--;

        auto &[tileA, tileB, tileC] = *triplet;
        if(tileA->used(); tileA->isMemoryManagerConnected() and tileA->canBeRecycled()) {
            matrixA_->tile(tileA->rowIdx(), tileA->colIdx(), nullptr);
            tileA->returnToMemoryManager();
        }
        if(tileB->used(); tileB->isMemoryManagerConnected() and tileB->canBeRecycled()) {
            matrixB_->tile(tileB->rowIdx(), tileB->colIdx(), nullptr);
            tileB->returnToMemoryManager();
        }

        const auto mt = tileC->rowIdx();
        const auto nt = tileC->colIdx();
        matrixC_->tile(mt, nt, tileC);
        if(tileC->used(); tileC->ttl() != 0) {
            if(auto &priorityQueue = queueMapping_.at(std::make_tuple(mt, nt)); !priorityQueue->empty()) {
                const auto &duplet = priorityQueue->top();
                matrixC_->tile(mt, nt, nullptr);
                this->addResult(std::make_shared<TileTriplet>(std::make_tuple(duplet.tileA_, duplet.tileB_, tileC)));
                priorityQueue->pop();
            }
        }
        else {
            this->addResult(tileC);
        }
    }

    [[nodiscard]] bool canTerminate() const override {
        return ttl_ == 0;
    }

private:
    std::shared_ptr<MatrixA> matrixA_ = nullptr;
    std::shared_ptr<MatrixB> matrixB_ = nullptr;
    std::shared_ptr<MatrixC> matrixC_ = nullptr;
    int64_t                  ttl_     = -1;

    std::unordered_map<
        std::tuple<int64_t, int64_t>,
        std::shared_ptr<std::priority_queue<Duplet>>,
        TupleHash
    > queueMapping_ = {};
};

template<char Id>
struct CommRequestList {
    std::vector<std::tuple<int64_t, int64_t>> data = {};
};

template<char Id>
struct CommSendList {
    std::vector<std::tuple<int64_t, int64_t, int64_t>> data = {};
};

template<typename MatrixType, char IdA, char IdB, char IdC
    , class Triplet = std::tuple<std::shared_ptr<MatrixContainer<MatrixType, IdA>>, std::shared_ptr<MatrixContainer<MatrixType, IdB>>, std::shared_ptr<MatrixContainer<MatrixType, IdC>>>
>
class JobGenerator final: public hh::AbstractTask<1, Triplet, CommRequestList<IdA>, CommRequestList<IdB>, CommSendList<IdA>, CommSendList<IdB>> {
public:
    using base =  hh::AbstractTask<1, Triplet, CommRequestList<IdA>, CommRequestList<IdB>, CommSendList<IdA>, CommSendList<IdB>>;

    explicit JobGenerator(): base("JobGenerator", 1, false) {}

    void execute(std::shared_ptr<Triplet> triplet) {
        const auto [matrixA, matrixB, matrixC] = *triplet;

        const auto MT                 = matrixC->matrixNumRowTiles();
        const auto KT                 = matrixA->matrixNumColTiles();
        const auto NT                 = matrixC->matrixNumColTiles();
        const auto nodeId             = getNodeId();
        const auto [pDim, qDim]       = getGridDim();
        const auto [pNodeId, qNodeId] = getGridNodeId();

        auto rowIndicesSet = std::set<int64_t>();
        auto colIndicesSet = std::set<int64_t>();
        for(auto mt = 0; mt < MT; ++mt) {
            for(auto nt = 0; nt < NT; ++nt) {
                if(matrixC->owner(mt, nt) != nodeId) continue;
                rowIndicesSet.insert(mt);
                colIndicesSet.insert(nt);
            }
        }
        const auto rowIndices = std::vector(rowIndicesSet.begin(), rowIndicesSet.end());
        const auto colIndices = std::vector(colIndicesSet.begin(), colIndicesSet.end());

        std::vector<int64_t> broadcastListA = {};
        for(int64_t q = 0; q < qDim; ++q) {
            const int64_t destinationNodeId = pNodeId*qDim + q;
            if(destinationNodeId == getNodeId()) continue;
            broadcastListA.emplace_back(destinationNodeId);
        }

        std::vector<int64_t> broadcastListB = {};
        for(int64_t p = 0; p < pDim; ++p) {
            const int64_t destinationNodeId = p*qDim + qNodeId;
            if(destinationNodeId == getNodeId()) continue;
            broadcastListB.emplace_back(destinationNodeId);
        }

        auto commSendListA = std::make_shared<CommSendList<IdA>>();
        for(int64_t kt = qNodeId; kt < KT; kt += qDim) {
            for(const auto mt: rowIndices) {
                for(const auto destinationId: broadcastListA) {
                    assert(matrixA->tile(mt, kt) != nullptr);
                    commSendListA->data.emplace_back(mt, kt, destinationId);
                }
            }
        }
        this->addResult(commSendListA);

        auto commSendListB = std::make_shared<CommSendList<IdB>>();
        for(int64_t kt = pNodeId; kt < KT; kt += pDim) {
            for(const auto nt: colIndices) {
                for(const auto destinationId: broadcastListB) {
                    assert(matrixB->tile(kt, nt) != nullptr);
                    commSendListB->data.emplace_back(kt, nt, destinationId);
                }
            }
        }
        this->addResult(commSendListB);

        auto commRequestListA = std::make_shared<CommRequestList<IdA>>();
        auto commRequestListB = std::make_shared<CommRequestList<IdB>>();
        for(int64_t kt = 0; kt < KT; ++kt) {
            for(const auto mt: rowIndices) {
                if(matrixA->owner(mt, kt) == nodeId) continue;
                commRequestListA->data.emplace_back(mt, kt);
            }

            for(const auto nt: colIndices) {
                if(matrixB->owner(kt, nt) == nodeId) continue;
                commRequestListB->data.emplace_back(kt, nt);
            }
        }
        this->addResult(commRequestListA);
        this->addResult(commRequestListB);
    }
};

template<typename MatrixType, char Id>
int32_t getBufferSizeInBytes(MatrixTile<MatrixType, Id> &tile) {
    return tile.byteSize();
}

template<typename MatrixType, char Id>
void* getBuffer(MatrixTile<MatrixType, Id> &tile) {
    return tile.data();
}

template<typename MatrixType, char Id>
class MatrixCommTask: public AbstractMpiCommTask<MatrixTile<MatrixType, Id>, CommRequestList<Id>, CommSendList<Id>> {
public:
    using Matrix = MatrixContainer<MatrixType, Id>;
    using Tile   = MatrixTile<MatrixType, Id>;

    explicit MatrixCommTask(const std::string &name, std::shared_ptr<MatrixContainer<MatrixType, Id>> matrix, const int32_t sendLimit = 64):
        AbstractMpiCommTask<MatrixTile<MatrixType, Id>, CommRequestList<Id>, CommSendList<Id>>(name, matrix->mpiComm(), false), matrix_(matrix),
        sendTtl_(sendLimit) {}

    void execute(std::shared_ptr<Tile> tile) override {
        tile->ttl(1);

        std::lock_guard lg(receiveMutex_);
        updateStateThreadUnsafe();
    }

    void execute(std::shared_ptr<CommRequestList<Id>> commRequestList) override {
        if constexpr(true) {
            std::lock_guard lg(receiveMutex_);
            requestQueue_.insert(requestQueue_.end(), commRequestList->data.begin(), commRequestList->data.end());
            requestQueueSize_.store(requestQueue_.size());
        }
        processLastRequest();
    }

    void execute(std::shared_ptr<CommSendList<Id>> commSendList) override {
        if constexpr(true) {
            std::lock_guard lg(sendMutex_);
            sendQueue_.insert(sendQueue_.end(), commSendList->data.begin(), commSendList->data.end());
            sendQueueSize_.store(sendQueue_.size());
        }
        processLastRequest();
    }

    [[nodiscard]] std::tuple<std::shared_ptr<Tile>, int32_t, int32_t, bool> mpiReceiveProtocol() override {
        std::lock_guard lg(receiveMutex_);

        updateStateThreadUnsafe();
        while(!requestQueue_.empty() and this->memoryManager()->currentSize() != 0) {
            auto [rowIdx, colIdx] = requestQueue_.front();
            requestQueue_.pop_front();
            requestQueueSize_.store(requestQueue_.size());
            if(auto tile = matrix_->tile(rowIdx, colIdx); tile != nullptr) {
                tile->ttl(1);
                if(receivedTiles_.empty()) {
                    this->addResult(tile);
                }
                else {
                    receivedTiles_.emplace_back(tile);
                    receivedTilesSize_.store(receivedTiles_.size());
                }
                continue;
            }

            const auto NT = matrix_->matrixNumColTiles();
            auto tile     = std::dynamic_pointer_cast<Tile>(this->getManagedMemory());
            tile->init(rowIdx, colIdx, matrix_->tileHeight(rowIdx, colIdx), matrix_->tileWidth(rowIdx, colIdx));
            tile->ttl(0);
            receivedTiles_.emplace_back(tile);
            receivedTilesSize_.store(receivedTiles_.size());
            return std::make_tuple(tile, static_cast<int32_t>(matrix_->owner(rowIdx, colIdx)), static_cast<int32_t>(rowIdx * NT + colIdx), false);
        }

        requestQueueSize_.store(requestQueue_.size());
        return std::make_tuple(nullptr, MPI_ANY_SOURCE, MPI_ANY_TAG, false);
    }

    void preProcessMpiReceivedData(std::shared_ptr<MatrixTile<MatrixType, Id>> &tile, [[maybe_unused]] const int32_t sourceNodeId, const int32_t tagId, [[maybe_unused]] const int32_t sizeInBytes) override {
        const auto NT     = matrix_->matrixNumColTiles();
        const auto rowIdx = tagId/NT;
        const auto colIdx = tagId%NT;
        tile->memoryState(MemoryState::SHARED);
        tile->ttl(1);
    }

    [[nodiscard]] std::tuple<std::shared_ptr<Tile>, int32_t, int32_t> mpiSendProtocol() override {
        if(sendQueueSize_.load() == 0) {
            return std::make_tuple(nullptr, MPI_ANY_SOURCE, MPI_ANY_TAG);
        }

        std::lock_guard lg(sendMutex_);
        if(sendTtl_ <= 0 or sendQueue_.empty()) {
            return std::make_tuple(nullptr, MPI_ANY_SOURCE, MPI_ANY_TAG);
        }

        const auto [rowIdx, colIdx, destinationNodeId] = sendQueue_.front();
        sendQueue_.pop_front();
        sendQueueSize_.store(sendQueue_.size());

        auto tile = matrix_->tile(rowIdx, colIdx);
        assert(tile != nullptr);

        return std::make_tuple(tile, destinationNodeId, rowIdx*matrix_->matrixNumColTiles() + colIdx);
    }

    void postProcessMpiSentData([[maybe_unused]] std::shared_ptr<MatrixTile<MatrixType, Id>> &tile, [[maybe_unused]] const int32_t destinationNodeId, [[maybe_unused]] const int32_t tagId, [[maybe_unused]] const int32_t sizeInBytes) override {
        std::lock_guard lg(sendMutex_);
        sendTtl_++;
    }

    [[nodiscard]] bool canTerminateComm() const override {
        return true
            and ttl_.load() == 0
            and requestQueueSize_.load() == 0
            and receivedTilesSize_.load() == 0
            and sendQueueSize_.load() == 0
            and this->isCommSendQueueEmpty()
            and this->isCommReceiveQueueEmpty();
    }

private:
    void updateStateThreadUnsafe() {
        while(!receivedTiles_.empty()) {
            auto tile = receivedTiles_.front();
            if(tile->ttl() == 0) break;//tile is not ready

            this->addResult(tile);
            receivedTiles_.pop_front();
        }
        receivedTilesSize_.store(receivedTiles_.size());
    }

    void processLastRequest() {
        ttl_.fetch_sub(1);
        if(ttl_.load() == 0) {
            this->joinCommThreads();
        }
    }

private:
    std::shared_ptr<Matrix> matrix_               = nullptr;
    std::atomic_int32_t     ttl_                  = 2;

    std::mutex                               receiveMutex_      = {};
    std::deque<std::shared_ptr<Tile>>        receivedTiles_     = {};
    std::atomic_int32_t                      receivedTilesSize_ = 0;
    int32_t                                  receiveLimit_      = 64;
    std::deque<std::tuple<int64_t, int64_t>> requestQueue_      = {};
    std::atomic_int32_t                      requestQueueSize_  = 0;

    std::mutex                                        sendMutex_     = {};
    std::deque<std::tuple<int64_t, int64_t, int64_t>> sendQueue_     = {};
    std::atomic_int32_t                               sendQueueSize_ = 0;
    int32_t                                           sendTtl_       = 0;
};

// Major::ROW => traverse along the row
// Major::COL => traverse along the column
template<Major Traversal>
class IndexGenerator2D {
public:
    explicit IndexGenerator2D(const std::array<int64_t, 2> index0, const std::array<int64_t, 2> stride, const std::array<int64_t, 2> extent)
        :index0_(index0), index_(index0), stride_(stride), extent_(extent) {}

    std::tuple<int64_t, int64_t> operator++() {
        increment();
        return indices();
    }

    std::tuple<int64_t, int64_t> operator++(int32_t) {
        const auto ret = indices();
        increment();
        return ret;
    }

    [[nodiscard]] std::tuple<int64_t, int64_t> indices() const {
        if(empty()) return std::make_tuple(-1, -1);
        return std::make_tuple(index_[0], index_[1]);
    }

    [[nodiscard]] bool empty() const { return extent_[0] <= index_[0] or extent_[1] <= index_[1]; }

private:
    void increment() {
        if constexpr(Traversal == Major::ROW) {
            index_[1] += stride_[1];
            if(extent_[1] <= index_[1]) {
                index_[0] += stride_[0];
                index_[1]  = index0_[1];
            }
        }
        if constexpr(Traversal == Major::COL) {
            index_[0] += stride_[0];
            if(extent_[0] <= index_[0]) {
                index_[0]  = index0_[0];
                index_[1] += stride_[1];
            }
        }
    }

    std::array<int64_t, 2> index0_ = {};
    std::array<int64_t, 2> index_  = {};
    std::array<int64_t, 2> stride_ = {};
    std::array<int64_t, 2> extent_ = {};
};

template<typename MatrixType, char IdA, char IdB
    , class MatrixA  = MatrixContainer<MatrixType, IdA>
    , class MatrixB  = MatrixContainer<MatrixType, IdB>
    , class MatrixAB = std::tuple<std::shared_ptr<MatrixA>, std::shared_ptr<MatrixB>>
    , class TileA    = MatrixTile<MatrixType, IdA>
    , class TileB    = MatrixTile<MatrixType, IdB>
>
class BroadcastTask final: public hh::AbstractTask<1, MatrixAB, TileA, TileB> {
public:
    using base = hh::AbstractTask<1, MatrixAB, TileA, TileB>;

    explicit BroadcastTask(const std::string &name, const MPI_Comm gridComm, const MPI_Comm groupCommA, const MPI_Comm groupCommB, const int32_t mmACap, const int64_t mmBCap, const int64_t tileSize):
        base(name, 1, false), gridComm_(gridComm), groupCommA_(groupCommA), groupCommB_(groupCommB), limitA_(mmACap), limitB_(mmBCap), tileSize_(tileSize) {}

    void initialize() override {
        mmA_ = std::make_shared<hh::StaticMemoryManager<TileA, int64_t, MemoryType>>(limitA_, tileSize_, MemoryType::HOST);
        mmB_ = std::make_shared<hh::StaticMemoryManager<TileB, int64_t, MemoryType>>(limitB_, tileSize_, MemoryType::HOST);

        mmA_->initialize();
        mmB_->initialize();
    }

    void execute(std::shared_ptr<MatrixAB> matAB) override {
        auto [matrixA, matrixB] = *matAB;

        int32_t groupSizeA = {};
        int32_t groupSizeB = {};
        if constexpr(true) {
            auto mpiLg = std::lock_guard(mpiMutex);
            checkMpiErrors(MPI_Comm_size(groupCommA_, &groupSizeA));
            checkMpiErrors(MPI_Comm_size(groupCommB_, &groupSizeB));
        }
        const auto [p0, q0]     = getGridNodeId();
        const auto [pDim, qDim] = getGridDim();

        const int32_t MT = matrixA->matrixNumRowTiles();
        const int32_t KT = matrixA->matrixNumColTiles();
        const int32_t NT = matrixB->matrixNumColTiles();

        auto generatorA   = IndexGenerator2D<Major::COL>({p0, 0}, {pDim, 1}, {MT, KT});
        auto tilesA       = std::vector(limitA_, std::shared_ptr<MatrixTile<MatrixType, IdA>>(nullptr));
        auto freeIndicesA = std::deque(limitA_, 0);
        std::iota(freeIndicesA.begin(), freeIndicesA.end(), 0);

        auto generatorB   = IndexGenerator2D<Major::ROW>({0, q0}, {1, qDim}, {KT, NT});
        auto tilesB       = std::vector(limitB_, std::shared_ptr<MatrixTile<MatrixType, IdB>>(nullptr));
        auto freeIndicesB = std::deque(limitB_, 0);
        std::iota(freeIndicesB.begin(), freeIndicesB.end(), 0);

        const auto limit       = limitA_ + limitB_;
        auto       mpiRequests = std::vector(limit, MPI_REQUEST_NULL);
        auto       indices     = std::vector(limit, -1);

        while(!generatorA.empty() or !generatorB.empty() or static_cast<int64_t>(freeIndicesA.size()) < limitA_ or static_cast<int64_t>(freeIndicesB.size()) < limitB_) {
            for(;!freeIndicesA.empty() and !generatorA.empty() and 0 < mmA_->currentSize(); freeIndicesA.pop_front(), ++generatorA) {
                const auto indexA           = freeIndicesA.front();
                const auto [rowIdx, colIdx] = generatorA.indices();
                auto       tileA            = matrixA->tile(rowIdx, colIdx);
                if(tileA == nullptr) {
                    tileA          = std::dynamic_pointer_cast<TileA>(mmA_->getManagedMemory());
                    tilesA[indexA] = tileA;
                }
                tileA->init(rowIdx, colIdx, matrixA->tileHeight(rowIdx, colIdx), matrixA->tileWidth(rowIdx, colIdx));

                auto mpiLG = std::lock_guard(mpiMutex);
                checkMpiErrors(MPI_Ibcast(
                    tileA->data(),
                    tileA->byteSize(),
                    MPI_BYTE,
                    colIdx%groupSizeA,
                    groupCommA_,
                    &mpiRequests[indexA]
                ));
            }

            for(;!freeIndicesB.empty() and !generatorB.empty() and 0 < mmB_->currentSize(); freeIndicesB.pop_front(), ++generatorB) {
                const auto indexB           = freeIndicesB.front();
                const auto [rowIdx, colIdx] = generatorB.indices();
                auto       tileB            = matrixB->tile(rowIdx, colIdx);
                if(tileB == nullptr) {
                    tileB          = std::dynamic_pointer_cast<TileB>(mmB_->getManagedMemory());
                    tilesB[indexB] = tileB;
                }
                tileB->init(rowIdx, colIdx, matrixB->tileHeight(rowIdx, colIdx), matrixB->tileWidth(rowIdx, colIdx));

                auto mpiLG = std::lock_guard(mpiMutex);
                checkMpiErrors(MPI_Ibcast(
                    tileB->data(),
                    tileB->byteSize(),
                    MPI_BYTE,
                    rowIdx%groupSizeB,
                    groupCommB_,
                    &mpiRequests[limitA_ + indexB]
                ));
            }

            int32_t outCount;
            if constexpr(true) {
                auto mpiLG = std::lock_guard(mpiMutex);
                checkMpiErrors(MPI_Testsome(
                    static_cast<int32_t>(mpiRequests.size()),
                    mpiRequests.data(),
                    &outCount,
                    indices.data(),
                    MPI_STATUSES_IGNORE
                ));
            }
            for(int32_t i = 0; i < outCount; ++i) {
                const auto index = indices[i];
                if(const auto indexA = index; 0 <= indexA and indexA < limitA_) {
                    if(std::ranges::find(freeIndicesA, indexA) != freeIndicesA.end()) continue;
                    if(tilesA[indexA]) this->addResult(tilesA[indexA]);
                    tilesA[indexA]     = nullptr;
                    freeIndicesA.emplace_back(indexA);
                    mpiRequests[index] = MPI_REQUEST_NULL;
                }
                else if(const auto indexB = indices[i]-limitA_; 0 <= indexB and indexB < limitB_) {
                    if(std::ranges::find(freeIndicesB, indexB) != freeIndicesB.end()) continue;
                    if(tilesB[indexB]) this->addResult(tilesB[indexB]);
                    tilesB[indexB]     = nullptr;
                    freeIndicesB.emplace_back(indexB);
                    mpiRequests[index] = MPI_REQUEST_NULL;
                }
            }

            if((freeIndicesA.empty() or generatorA.empty()) and (freeIndicesB.empty() or generatorB.empty()))
                std::this_thread::yield();
        }

        while(mmA_->currentSize() != mmA_->capacity() or mmB_->currentSize() != mmB_->capacity())
            std::this_thread::yield();
    }

private:
    MPI_Comm                                   gridComm_   = MPI_COMM_NULL;
    MPI_Comm                                   groupCommA_ = MPI_COMM_NULL;
    MPI_Comm                                   groupCommB_ = MPI_COMM_NULL;
    int64_t                                    limitA_     = 1;
    int64_t                                    limitB_     = 1;
    int64_t                                    tileSize_   = 1;
    std::shared_ptr<hh::AbstractMemoryManager> mmA_        = nullptr;
    std::shared_ptr<hh::AbstractMemoryManager> mmB_        = nullptr;
};

#endif //HH3_MATMUL_TASKS_H
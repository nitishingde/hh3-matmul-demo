#ifndef HH3_MATMUL_DATA_H
#define HH3_MATMUL_DATA_H

#include <atomic>
#include <string>
#include <utility>
#include <hedgehog/hedgehog.h>
#include "utility.h"

struct Vec2 {
    int64_t x;
    int64_t y;
};

enum class MemoryType {
    HOST,
    CUDA,
    CUDA_UNIFIED_MEMORY,
};

// TODO: state machine, refer MOSI protocol used in SLATE. May or may not be needed.
enum class MemoryState {
    SHARED,
    MODIFIED,
    ON_HOLD,
    INVALID,
};

enum class MemoryOwner {
    WORKSPACE,
    USER,
};

enum class Major {
    ROW,
    COL,
};

// A wrapper around the tile data
template<typename MatrixType, char Id>
class MatrixTile: public hh::ManagedMemory {
public:
    explicit MatrixTile(const int64_t rowIdx, const int64_t colIdx, const int64_t height, const int64_t width, const MemoryType memoryType = MemoryType::HOST, const Major major = Major::COL):
        rowIdx_(rowIdx), colIdx_(colIdx),
        byteSize_(width*height*sizeof(MatrixType)),
        height_(height), width_(width),
        memoryType_(memoryType), major_(major), memoryOwner_(MemoryOwner::USER), memoryState_(MemoryState::SHARED) {

        ttl_.store(-1);
        if(memoryType_ == MemoryType::HOST) {
            pData_ = new uint8_t[byteSize_];
        }
        else if(memoryType_ == MemoryType::CUDA) {
            checkCudaErrors(cudaMalloc(&pData_, byteSize_));
        }
        else if(memoryType_ == MemoryType::CUDA_UNIFIED_MEMORY) {
            checkCudaErrors(cudaMallocManaged(&pData_, byteSize_));
        }

        initCudaEvent();
    }

    // For memory management
    explicit MatrixTile(const int64_t tileSize, const MemoryType memoryType):
        byteSize_(tileSize*tileSize*sizeof(MatrixType)),
        width_(tileSize),
        height_(tileSize),
        memoryType_(memoryType),
        memoryOwner_(MemoryOwner::WORKSPACE),
        memoryState_(MemoryState::SHARED) {

        ttl_.store(-1);
        if(memoryType_ == MemoryType::HOST) {
            pData_ = new uint8_t[byteSize_];
        }
        else if(memoryType_ == MemoryType::CUDA) {
            checkCudaErrors(cudaMalloc(&pData_, byteSize_));
        }
        else if(memoryType_ == MemoryType::CUDA_UNIFIED_MEMORY) {
            checkCudaErrors(cudaMallocManaged(&pData_, byteSize_));
        }

        initCudaEvent();
    }

    ~MatrixTile() override {
        if(memoryType_ == MemoryType::HOST) {
            delete[] static_cast<uint8_t*>(pData_);
        }
        else if(memoryType_ == MemoryType::CUDA_UNIFIED_MEMORY) {
            checkCudaErrors(cudaFree(pData_));
        }

        for(size_t id = 0; id < cudaEvents_.size(); ++id) {
            if(cudaEventCreated_[id]) {
                checkCudaErrors(cudaEventDestroy(cudaEvents_[id]));
            }
            cudaEventCreated_[id] = false;
        }
    }

//    void postProcess() override {}

    bool canBeRecycled() override {
        return ttl_.load() == 0;
    }

    void clean() override {
        ttl_.store(-1);
    }

    void preProcess() override {
        ManagedMemory::preProcess();
    }

    /**
     * Record cudaAsync API call using stream for later synchronization.
     *
     * @param cudaStream
     */
    void recordEvent(cudaStream_t cudaStream, int32_t deviceId = 0) {
        if (!cudaEventCreated_[deviceId]) {
            checkCudaErrors(cudaEventCreate(&cudaEvents_[deviceId]));
            cudaEventCreated_[deviceId] = true;
        }
        checkCudaErrors(cudaEventRecord(cudaEvents_[deviceId], cudaStream));
    }

    /**
     * Synchronize the cudaAsync API called previously.
     */
    void synchronizeEvent(int32_t deviceId = 0) {
        if(cudaEventCreated_[deviceId]) {
            checkCudaErrors(cudaEventSynchronize(cudaEvents_[deviceId]));
        }
    }

    void used() {
        ttl_.fetch_sub(1);
    }

    // Getters
    [[nodiscard]] void*       data()                   { return pData_;                                }
    [[nodiscard]] int64_t     byteSize()         const { return width_*height_*sizeof(MatrixType);     }
    [[nodiscard]] int64_t     width()            const { return width_;                                }
    [[nodiscard]] int64_t     height()           const { return height_;                               }
    [[nodiscard]] int64_t     rowIdx()           const { return rowIdx_;                               }
    [[nodiscard]] int64_t     colIdx()           const { return colIdx_;                               }
    [[nodiscard]] int64_t     leadingDimension() const { return major_ == Major::COL? height_: width_; }
    [[nodiscard]] MemoryType  memoryType()       const { return memoryType_;                           }
    [[nodiscard]] Major       major()            const { return major_;                                }
    [[nodiscard]] MemoryOwner memoryOwner()      const { return memoryOwner_;                          }
    [[nodiscard]] MemoryState memoryState()      const { return memoryState_;                          }
    [[nodiscard]] int64_t     ttl()              const { return ttl_;                                  }

    // Setters
    void init(const int64_t rowIdx, const int64_t colIdx, const int64_t height, const int64_t width) {
        assert(int64_t(height*width*sizeof(MatrixType)) <= byteSize_);
        rowIdx_ = rowIdx;
        colIdx_ = colIdx;
        width_  = width;
        height_ = height;
    }
    void memoryState(const MemoryState memoryState) { memoryState_ = memoryState; }
    void major(const Major major)                   { major_ = major;             }
    void ttl(const int64_t ttl)                     { ttl_.store(ttl);            }

private:
    void initCudaEvent() {
        int32_t gpuCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&gpuCount));
        cudaEvents_.resize(gpuCount);
        cudaEventCreated_.resize(gpuCount, false);
    }

private:
    void        *pData_           = nullptr;                // untouchable
    int64_t     byteSize_         = 0;                      // untouchable
    int64_t     width_            = 0;
    int64_t     height_           = 0;
    int64_t     rowIdx_           = 0;
    int64_t     colIdx_           = 0;
    MemoryType  memoryType_       = MemoryType::HOST;
    MemoryOwner memoryOwner_      = MemoryOwner::WORKSPACE; // Only initialized via constructor
    MemoryState memoryState_      = MemoryState::SHARED;    //FIXME: needed?
    Major       major_            = Major::COL;

    // CUDA related data members
    std::vector<cudaEvent_t> cudaEvents_       = {};
    std::vector<bool>        cudaEventCreated_ = {};

    // Managed Memory
    std::atomic_int64_t ttl_ = {};
};

template<typename MatrixType, char Id>
class MatrixContainer {
private:
    using Tile = MatrixTile<MatrixType, Id>;
    template<typename GridT>
    using Grid = std::vector<std::vector<GridT>>;

public:
    explicit MatrixContainer(const MemoryType memoryType, const int64_t height, const int64_t width, const int64_t tileDim, const int64_t pGridDim, const int64_t qGridDim, MPI_Comm mpiComm):
        tileMemoryType_(memoryType), height_(height), width_(width), tileDim_({tileDim, tileDim}), pGridDim_(pGridDim), qGridDim_(qGridDim), mpiComm_(mpiComm) {

        int32_t temp;
        checkMpiErrors(MPI_Comm_rank(mpiComm, &temp));
        nodeId_ = temp;
        checkMpiErrors(MPI_Comm_size(mpiComm, &temp));
        numNodes_ = temp;
        assert(pGridDim_*qGridDim_ == numNodes_);

        tileGrid_.resize((height+tileDim-1)/tileDim, std::vector<std::shared_ptr<Tile>>((width+tileDim-1)/tileDim, nullptr));
        tileOwnership_.resize((height+tileDim-1)/tileDim, std::vector<int64_t>((width+tileDim-1)/tileDim, 0));
    }

    virtual bool init() = 0;
    [[maybe_unused]] virtual int64_t typeId() = 0;

    // Getters
    [[nodiscard]] int64_t         matrixHeight()                                    const { return height_;                         }
    [[nodiscard]] int64_t         matrixWidth()                                     const { return width_;                          }
    [[nodiscard]] int64_t         matrixNumRowTiles()                               const { return tileGrid_.size();                }
    [[nodiscard]] int64_t         matrixNumColTiles()                               const { return tileGrid_[0].size();             }
    [[nodiscard]] int64_t         owner(const int64_t rowIdx, const int64_t colIdx) const { return tileOwnership_[rowIdx][colIdx];  }
    [[nodiscard]]
    std::shared_ptr<Tile>         tile(const int64_t rowIdx, const int64_t colIdx)        { return this->tileGrid_[rowIdx][colIdx]; }
    [[nodiscard]] int64_t         tileDim()                                         const { return std::max(tileDim_.x, tileDim_.y);}
    [[nodiscard]] const MPI_Comm& mpiComm()                                         const { return mpiComm_;                        }
    [[nodiscard]] int64_t         nodeId()                                          const { return nodeId_;                         }
    [[nodiscard]] int64_t         numNodes()                                        const { return numNodes_;                       }
    [[nodiscard]] bool            isRootNodeId()                                    const { return nodeId_ == 0;                    }
    [[nodiscard]] bool            isLastNodeId()                                    const { return nodeId_ == (numNodes_-1);        }

    [[nodiscard]] int64_t tileHeight(int64_t rowIdx, [[maybe_unused]]int64_t colIdx) const {
        return std::min(tileDim_.y, int64_t(height_-tileDim_.y*rowIdx));
    }

    [[nodiscard]] int64_t tileWidth([[maybe_unused]]int64_t rowIdx, int64_t colIdx) const {
        return std::min(tileDim_.x, int64_t(width_-tileDim_.x*colIdx));
    }

    // Setters
    void tile(int64_t rowIdx, int64_t colIdx, std::shared_ptr<Tile> tile) {
        assert(owner(rowIdx, colIdx) != nodeId_);
        assert(tile->memoryOwner() == MemoryOwner::WORKSPACE);
        tileGrid_[rowIdx][colIdx] = tile;
    }

protected:
    MemoryType                  tileMemoryType_ = MemoryType::HOST;
    Grid<std::shared_ptr<Tile>> tileGrid_       = {};
    Grid<int64_t>               tileOwnership_  = {};
    int64_t                     width_          = -1;
    int64_t                     height_         = -1;
    Vec2                        tileDim_        = {-1, -1};
    MPI_Comm                    mpiComm_        = {};
    int64_t                     nodeId_         = -1;
    int64_t                     numNodes_       = -1;
    int64_t                     pGridDim_       = -1;
    int64_t                     qGridDim_       = -1;
};

template<typename MatrixType, char Id>
class TwoDBlockCyclicMatrix: public MatrixContainer<MatrixType, Id> {
private:
    using Tile = MatrixTile<MatrixType, Id>;

public:
    explicit TwoDBlockCyclicMatrix(const MemoryType memoryType, const int64_t height, const int64_t width, const int64_t tileDim, const int64_t pGridDim, const int64_t qGridDim, MPI_Comm mpiComm):
        MatrixContainer<MatrixType, Id>(memoryType, height, width, tileDim, pGridDim, qGridDim, mpiComm) {
        init();
    }

    bool init() override {
        const auto MT = this->matrixNumRowTiles();
        const auto NT = this->matrixNumColTiles();
        int64_t p0 = this->nodeId_/this->qGridDim_, q0 = this->nodeId_%this->qGridDim_;
        for(int64_t p = p0; p < MT; p+=this->pGridDim_) {
            for(int64_t q = q0; q < NT; q+=this->qGridDim_) {
                auto tile = std::make_shared<Tile>(p, q, this->tileHeight(p, q), this->tileWidth(p, q), this->tileMemoryType_);
                auto pData = (MatrixType *)tile->data();
                for(int64_t i = 0; i < tile->width()*tile->height(); ++i) pData[i] = fast_rand()%10;
                this->tileGrid_[p][q] = tile;
            }
        }

        for(int64_t p = 0; p < MT; ++p) {
            for(int64_t q = 0; q < NT; ++q) {
                this->tileOwnership_[p][q] = (q%this->qGridDim_ + p*this->qGridDim_)%this->numNodes_;
            }
        }

        return true;
    }

    int64_t typeId() override {
        return typeid(*this).hash_code();
    }
};

template<typename MatrixType, char Id>
class TwoDBlockCyclicContiguousSubMatrix: public MatrixContainer<MatrixType, Id> {
private:
    using Tile = MatrixTile<MatrixType, Id>;

public:
    explicit TwoDBlockCyclicContiguousSubMatrix(const MemoryType memoryType, const int64_t height, const int64_t width, const int64_t tileDim, const int64_t pGridDim, const int64_t qGridDim, MPI_Comm mpiComm):
        MatrixContainer<MatrixType, Id>(memoryType, height, width, tileDim, pGridDim, qGridDim, mpiComm) {
        init();
    }

    bool init() override {
        const auto MT = this->matrixNumRowTiles();
        const auto NT = this->matrixNumColTiles();
        const auto P = this->pGridDim_, Q = this->qGridDim_;

        for(int64_t p = 0; p < MT; ++p) {
            for(int64_t q = 0; q < NT; ++q) {
                this->tileOwnership_[p][q] = (q%Q + p*Q)%this->numNodes_;
            }
        }

        for(auto &row: this->tileOwnership_) {
            std::sort(row.begin(), row.end());
        }
        std::sort(this->tileOwnership_.begin(), this->tileOwnership_.end(), [](const auto &a, const auto &b) {
            return a[0] < b[0];
        });

        for(int64_t r = 0; r < MT; ++r) {
            for(int64_t c = 0; c < NT; ++c) {
                if(this->tileOwnership_[r][c] == this->nodeId_) {
                    auto tile = std::make_shared<Tile>(r, c, this->tileHeight(r, c), this->tileWidth(r, c), this->tileMemoryType_);
                    auto pData = (MatrixType *)tile->data();
                    for(int64_t i = 0; i < tile->width()*tile->height(); ++i) pData[i] = fast_rand()%10;
                    this->tileGrid_[r][c] = tile;
                }
            }
        }

        return true;
    }

    int64_t typeId() override {
        return typeid(*this).hash_code();
    }
};

template<char Id>
struct DbRequest {
    int64_t rowIdx = -1;
    int64_t colIdx = -1;
    bool    quit   = false;
//    int64_t priority = 0; // zero is the highest priority
//    int64_t deviceId = 0;
    explicit DbRequest(const int64_t r, const int64_t c, const bool q = false): rowIdx(r), colIdx(c), quit(q) {}
};

struct GpuToken: public hh::ManagedMemory {
    int32_t id = -1;

    explicit GpuToken(const int32_t val = -1): id(val) {}
};

class GpuTokenMemoryManager: public hh::MemoryManager<GpuToken> {
public:
    explicit GpuTokenMemoryManager(uint32_t capacity): hh::MemoryManager<GpuToken>(capacity) {
        std::iota(devices_.begin(), devices_.end(), 0);
    }

    explicit GpuTokenMemoryManager(const std::vector<int32_t> &devices):
        hh::MemoryManager<GpuToken>(devices.size()), devices_(devices) {}

    ~GpuTokenMemoryManager() override = default;

    std::shared_ptr<AbstractMemoryManager> copy() override {
        return std::make_shared<GpuTokenMemoryManager>(this->devices_);
    }

    void initializeMemoryManager() override {
        int32_t i = 0;
        for(auto& mm: *this->pool()) {
            auto token = std::static_pointer_cast<GpuToken>(mm);
            token->id = devices_[i];
            i++;
        }
    }

private:
    std::vector<int32_t> devices_ = {};
};

template<typename MatrixType, char IdA, char IdB, char IdC>
class GpuJob {
private:
    using TileA      = MatrixTile<MatrixType, IdA>;
    using TileB      = MatrixTile<MatrixType, IdB>;
    using time_point = std::chrono::time_point<std::chrono::system_clock>;

public:
    explicit GpuJob(bool shouldQuit = false): quit_(shouldQuit), gpuToken_(std::make_shared<GpuToken>(-1)) {}

    ~GpuJob() {
        finished();
    }

    void startTimer() {
        startTime_ = std::chrono::system_clock::now();
    }

    void stopTimer() {
        endTime_ = std::chrono::system_clock::now();
    }

    double timeIt() {
        return double(std::chrono::duration_cast<std::chrono::nanoseconds>(endTime_ - startTime_).count());
    }

    void finished() {
        if(gpuToken_ and gpuToken_->isMemoryManagerConnected()) {
            gpuToken_->returnToMemoryManager();
        }
        gpuToken_ = nullptr;
    }

    // Getters
    [[nodiscard]] int32_t gpuId()            const { return gpuToken_->id; }
    [[nodiscard]] auto&   tilesFromMatrixA()       { return colA_;         }
    [[nodiscard]] auto&   tilesFromMatrixB()       { return rowB_;         }
    [[nodiscard]] bool    shouldQuit()       const { return quit_;         }

    // Setters
    void token(std::shared_ptr<GpuToken> token) { gpuToken_ = std::move(token); }
    void addTileA(std::shared_ptr<TileA> tileA) { colA_.emplace_back(tileA);    }
    void addTileB(std::shared_ptr<TileB> tileB) { rowB_.emplace_back(tileB);    }
    void quit(const bool flag)                  { quit_ = flag;                 }

    std::string toString() {
        std::stringstream ss;
        ss << "[GPU] " << gpuToken_->id << "\n";

        ss << "[A]: ";
        for(auto& tile: colA_) ss << "(" << tile->rowIdx() << ", " << tile->colIdx() << ") ";
        ss << "\n";

        ss << "[B]: ";
        for(auto& tile: rowB_) ss << "(" << tile->rowIdx() << ", " << tile->colIdx() << ") ";
        ss << "\n";

        return ss.str();
    }

private:
    std::shared_ptr<GpuToken>                          gpuToken_  = nullptr;
    std::deque<std::shared_ptr<TileA>>                 colA_      = {};
    std::deque<std::shared_ptr<TileB>>                 rowB_      = {};
    bool                                               quit_      = false;
    std::chrono::time_point<std::chrono::system_clock> startTime_ = {};
    std::chrono::time_point<std::chrono::system_clock> endTime_   = {};
};

template<typename MatrixType, char Id>
struct GcMatrixTile {
    explicit GcMatrixTile(std::shared_ptr<MatrixTile<MatrixType, Id>> tile): tile(tile) {}
    std::shared_ptr<MatrixTile<MatrixType, Id>> tile = nullptr;
};

#endif //HH3_MATMUL_DATA_H

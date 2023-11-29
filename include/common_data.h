#ifndef HH3_MATMUL_COMMON_DATA_H
#define HH3_MATMUL_COMMON_DATA_H

#include <hedgehog/hedgehog.h>
#include <atomic>
#include <string>
#include <utility>
#include "common_utility.h"

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
        memoryType_(memoryType), major_(major), memoryOwner_(MemoryOwner::WORKSPACE), memoryState_(MemoryState::SHARED) {

        allocate();
        ttl_.store(-1);
        initCudaEvent();
    }

    explicit MatrixTile(void *pData, const int64_t rowIdx, const int64_t colIdx, const int64_t height, const int64_t width, const MemoryType memoryType, const Major major = Major::COL):
        pData_(pData),
        rowIdx_(rowIdx), colIdx_(colIdx),
        byteSize_(width*height*sizeof(MatrixType)),
        height_(height), width_(width),
        memoryType_(memoryType), major_(major), memoryOwner_(MemoryOwner::USER), memoryState_(MemoryState::SHARED) {

        assert(pData != nullptr);
        ttl_.store(-1);
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

        allocate();
        ttl_.store(-1);
        initCudaEvent();
    }

    // For memory management
    explicit MatrixTile(void *pData, const int64_t tileSize, const MemoryType memoryType):
        pData_(pData),
        byteSize_(tileSize*tileSize*sizeof(MatrixType)),
        width_(tileSize),
        height_(tileSize),
        memoryType_(memoryType),
        memoryOwner_(MemoryOwner::USER),
        memoryState_(MemoryState::SHARED) {

        assert(pData != nullptr);
        ttl_.store(-1);
        initCudaEvent();
    }

    ~MatrixTile() override {
        if(memoryOwner_ == MemoryOwner::WORKSPACE) {
            switch(memoryType_) {
                case MemoryType::CUDA:
                case MemoryType::CUDA_UNIFIED_MEMORY:
                    checkCudaErrors(cudaFree(pData_));
                    break;

                case MemoryType::HOST:
                default:
                    delete[] static_cast<uint8_t*>(pData_);
                    break;
            }
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
    void allocate() {
        assert(memoryOwner_ == MemoryOwner::WORKSPACE);

        switch(memoryType_) {
            case MemoryType::CUDA:
                checkCudaErrors(cudaMalloc(&pData_, byteSize_));
                break;

            case MemoryType::CUDA_UNIFIED_MEMORY:
                checkCudaErrors(cudaMallocManaged(&pData_, byteSize_));
                break;

            case MemoryType::HOST:
            default:
                pData_ = new uint8_t[byteSize_];
                break;
        }
    }

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
    void allocate() {
        switch(tileMemoryType_) {
            case MemoryType::CUDA_UNIFIED_MEMORY:
                checkCudaErrors(cudaMallocManaged(&pData_, byteSize_));
                break;

            case MemoryType::HOST:
            default:
                pData_ = new uint8_t[byteSize_];
                break;
        }
    }

protected:
    uint8_t                     *pData_         = nullptr;
    int64_t                     byteSize_       = 0;
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

    ~TwoDBlockCyclicMatrix() {
        delete[] this->pData_;
        this->pData_    = nullptr;
        this->byteSize_ = 0;
    }


    bool init() override {
        const auto MT = this->matrixNumRowTiles();
        const auto NT = this->matrixNumColTiles();
        int64_t p0 = this->nodeId_/this->qGridDim_, q0 = this->nodeId_%this->qGridDim_;

        this->byteSize_ = 0;
        for(int64_t p = p0; p < MT; p+=this->pGridDim_) {
            for(int64_t q = q0; q < NT; q+=this->qGridDim_) {
                this->byteSize_ += this->tileHeight(p, q)*this->tileWidth(p, q)*sizeof(MatrixType);
            }
        }
        this->allocate();

        int64_t pos = 0;
        for(int64_t p = p0; p < MT; p+=this->pGridDim_) {
            for(int64_t q = q0; q < NT; q+=this->qGridDim_) {
                auto tile  = std::make_shared<Tile>((void*)&(this->pData_[pos]), p, q, this->tileHeight(p, q), this->tileWidth(p, q), this->tileMemoryType_);
                auto pData = (MatrixType *)tile->data();
                pos       += tile->byteSize();
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

    ~TwoDBlockCyclicContiguousSubMatrix() {
        delete[] this->pData_;
        this->pData_    = nullptr;
        this->byteSize_ = 0;
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

        this->byteSize_ = 0;
        for(int64_t r = 0; r < MT; ++r) {
            for(int64_t c = 0; c < NT; ++c) {
                if(this->tileOwnership_[r][c] == this->nodeId_) {
                    this->byteSize_ += this->tileHeight(r, c)*this->tileWidth(r, c)*sizeof(MatrixType);
                }
            }
        }
        this->allocate();

        int64_t pos = 0;
        for(int64_t r = 0; r < MT; ++r) {
            for(int64_t c = 0; c < NT; ++c) {
                if(this->tileOwnership_[r][c] == this->nodeId_) {
                    auto tile  = std::make_shared<Tile>((void*)&(this->pData_[pos]), r, c, this->tileHeight(r, c), this->tileWidth(r, c), this->tileMemoryType_);
                    auto pData = (MatrixType *)tile->data();
                    pos       += tile->byteSize();
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
    int64_t                                   srcNode = {};
    std::vector<std::tuple<int64_t, int64_t>> indices = {};
    bool                                      quit    = false;

    explicit DbRequest(int64_t srcNode): srcNode(srcNode) {}

    explicit DbRequest(bool quit): quit(quit) {}

    explicit DbRequest(int64_t rowIdx, int64_t colIdx) {
        indices.emplace_back(rowIdx, colIdx);
    }

    void addIndex(int64_t rowIdx, int64_t colIdx) {
        indices.emplace_back(rowIdx, colIdx);
    }
};

template<char Id>
struct DwBatchRequest {
    bool                                               quit = false;
    std::vector<std::tuple<int32_t, int32_t, int32_t>> data = {};

    explicit DwBatchRequest() = default;

    explicit DwBatchRequest(bool shouldQuit) {
        quit = shouldQuit;
    }

    void addIndex(int32_t rowIdx, int32_t colIdx) {
        data.emplace_back(rowIdx, colIdx, tagGenerator());
    }
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

class DotTimer {
public:
    explicit DotTimer() = default;

    void start() {
        isStarted_ = true;
        startTime_ = std::chrono::system_clock::now();
    }

    void stop() {
        endTime_ = std::chrono::system_clock::now();
        if(!isStarted_) throw std::runtime_error("DotTimer::stop() is called before calling DotTimer::start()\n");
        isStarted_ = false;
        log();
    }

    void merge(const DotTimer &dotTimer) {
        count_ += dotTimer.count_;
        sum_   += dotTimer.sum_;
        min_    = std::min(min_, dotTimer.min_);
        max_    = std::max(max_, dotTimer.max_);
    }

    [[nodiscard]] std::string format() {
        if(min_/1.e9 > .999) {
            factor_ = 1.e9;
            suffix_ = "s";
        }
        else if(min_/1.e6 > .999) {
            factor_ = 1.e6;
            suffix_ = "ms";
        }
        else if(min_/1.e3 > .999) {
            factor_ = 1.e3;
            suffix_ = "us";
        }

        return suffix_;
    }

    // Getters
    [[nodiscard]] double  min()       const { return min_/factor_;       }
    [[nodiscard]] double  avg()       const { return totalTime()/count_; }
    [[nodiscard]] double  max()       const { return max_/factor_;       }
    [[nodiscard]] double  totalTime() const { return sum_/factor_;       }
    [[nodiscard]] int32_t count()     const { return count_;             }

private:
    void log() {
        auto time = double(std::chrono::duration_cast<std::chrono::nanoseconds>(endTime_ - startTime_).count());
        count_++;
        sum_ += time;
        min_  = std::min(min_, time);
        max_  = std::max(max_, time);
    }

private:
    bool                                               isStarted_ = false;
    std::chrono::time_point<std::chrono::system_clock> startTime_ = {};
    std::chrono::time_point<std::chrono::system_clock> endTime_   = {};
    double                                             min_       = std::numeric_limits<double>::max();
    double                                             max_       = std::numeric_limits<double>::min();
    double                                             sum_       = 0.f;
    int32_t                                            count_     = 0;
    double                                             factor_    = 1.e9;
    std::string                                        suffix_    = "s";
};


#endif //HH3_MATMUL_COMMON_DATA_H

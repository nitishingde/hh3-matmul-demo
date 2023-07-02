#ifndef HH3_MATMUL_DATA
#define HH3_MATMUL_DATA

#include <string>
#include <hedgehog/hedgehog.h>
#include "utility.h"

struct Dim {
    int64_t x;
    int64_t y;
};

enum class MemoryType {
    HOST,
    CUDA_UNIFIED_MEMORY,
};

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
    explicit MatrixTile(int64_t rowIdx, int64_t colIdx, int64_t height, int64_t width, MemoryType memoryType = MemoryType::HOST, Major major = Major::COL):
        rowIdx_(rowIdx), colIdx_(colIdx),
        byteSize_(width*height*sizeof(MatrixType)),
        height_(height), width_(width),
        memoryType_(memoryType), major_(major), memoryOwner_(MemoryOwner::USER), memoryState_(MemoryState::SHARED) {

        if(memoryType_ == MemoryType::HOST) {
            pData_ = new uint8_t[byteSize_];
        }
        else if(memoryType_ == MemoryType::CUDA_UNIFIED_MEMORY) {
            checkCudaErrors(cudaMallocManaged(&pData_, byteSize_));
        }
    }

    // For memory management
    explicit MatrixTile(int64_t tileSize, MemoryType memoryType):
        byteSize_(tileSize*tileSize*sizeof(MatrixType)),
        width_(tileSize),
        height_(tileSize),
        memoryType_(memoryType),
        memoryOwner_(MemoryOwner::WORKSPACE),
        memoryState_(MemoryState::INVALID) {

        if(memoryType_ == MemoryType::HOST) {
            pData_ = new uint8_t[byteSize_];
        }
        else if(memoryType_ == MemoryType::CUDA_UNIFIED_MEMORY) {
            checkCudaErrors(cudaMallocManaged(&pData_, byteSize_));
        }
    }

    ~MatrixTile() override {
        if(memoryType_ == MemoryType::HOST) {
            delete[] static_cast<uint8_t*>(pData_);
        }
        else if(memoryType_ == MemoryType::CUDA_UNIFIED_MEMORY) {
            checkCudaErrors(cudaFree(pData_));
        }
    }

//    void postProcess() override {}

    bool canBeRecycled() override {
        return ttl_ == 0;
    }

    void clean() override {
        memoryState_ = MemoryState::INVALID;
    }

    void preProcess() override {
        ManagedMemory::preProcess();
    }

    /**
     * Record cudaAsync API call using stream for later synchronization.
     *
     * @param cudaStream
     */
    void recordEvent(cudaStream_t cudaStream) {
        assert(memoryType_ == MemoryType::CUDA_UNIFIED_MEMORY);
        if (!cudaEventCreated_) {
            checkCudaErrors(cudaEventCreate(&cudaEvent_));
            cudaEventCreated_ = true;
        }
        checkCudaErrors(cudaEventRecord(cudaEvent_, cudaStream));
    }

    /**
     * Synchronize the cudaAsync API called previously.
     */
    void synchronizeEvent() {
        assert(memoryType_ == MemoryType::CUDA_UNIFIED_MEMORY);
        if(cudaEventCreated_) {
            checkCudaErrors(cudaEventSynchronize(cudaEvent_));
        }
    }

    void used() {
        ttl_--;
    }

    // Getters
    [[nodiscard]] void*       data()                   { return pData_;                                }
    [[nodiscard]] int64_t    byteSize()         const { return width_*height_*sizeof(MatrixType);     }
    [[nodiscard]] int64_t    width()            const { return width_;                                }
    [[nodiscard]] int64_t    height()           const { return height_;                               }
    [[nodiscard]] int64_t     rowIdx()           const { return rowIdx_;                               }
    [[nodiscard]] int64_t     colIdx()           const { return colIdx_;                               }
    [[nodiscard]] int64_t    leadingDimension() const { return major_ == Major::COL? height_: width_; }
//    [[nodiscard]] MemoryType  memoryType()       const { return memoryType_;                           }
    [[nodiscard]] Major       major()            const { return major_;                                }
    [[nodiscard]] MemoryOwner memoryOwner()      const { return memoryOwner_;                          }
    [[nodiscard]] int64_t     ttl()              const {return ttl_;                                   }

    // Setters
    void init(int64_t rowIdx, int64_t colIdx, int64_t height, int64_t width) {
        assert(int64_t(height*width*sizeof(MatrixType)) <= byteSize_);
        rowIdx_ = rowIdx;
        colIdx_ = colIdx;
        width_  = width;
        height_ = height;
        memoryState_ = MemoryState::ON_HOLD;
    }
    void memoryState(MemoryState memoryState) { memoryState_ = memoryState; }
//    void width(int64_t width)              { width_ = width;           }
//    void height(int64_t height)            { height_ = height;         }
//    void rowIdx(int64_t rowIdx)            { rowIdx_ = rowIdx;         }
//    void colIdx(int64_t colIdx)            { colIdx_ = colIdx;         }
//    void memoryType(MemoryType memoryType)  { memoryType_ = memoryType; }
//    void order(Order order)                 { order_ = order;           }
    void ttl(const int64_t ttl)               { ttl_ = ttl; }

private:
    void        *pData_           = nullptr;                // untouchable
    int64_t    byteSize_         = 0;                      // untouchable
    int64_t    width_            = 0;
    int64_t    height_           = 0;
    int64_t     rowIdx_           = 0;
    int64_t     colIdx_           = 0;
    MemoryType  memoryType_       = MemoryType::HOST;
    MemoryOwner memoryOwner_      = MemoryOwner::WORKSPACE; // Only initialized via constructor
    MemoryState memoryState_      = MemoryState::SHARED;    //FIXME: needed?
    Major       major_            = Major::COL;

    // CUDA related data members
    cudaEvent_t cudaEvent_        = {};
    bool        cudaEventCreated_ = false;

    // Managed Memory
    int64_t ttl_ = 0;
};

template<typename MatrixType, char Id>
class MatrixContainer {
private:
    using Tile = MatrixTile<MatrixType, Id>;
    template<typename GridT>
    using Grid = std::vector<std::vector<GridT>>;

public:
    explicit MatrixContainer(int64_t height, int64_t width, int64_t tileDim): height_(height), width_(width), tileDim_({tileDim, tileDim}) {
        tileGrid_.resize((height+tileDim-1)/tileDim, std::vector<std::shared_ptr<Tile>>((width+tileDim-1)/tileDim, nullptr));
        tileOwnership_.resize((height+tileDim-1)/tileDim, std::vector<int64_t>((width+tileDim-1)/tileDim, 0));
    }

    virtual bool init() = 0;
//    virtual std::shared_ptr<Tile> tile(int64_t rowIdx, int64_t colIdx) = 0;
    [[maybe_unused]] virtual int64_t typeId() = 0;

    // Getters
//    [[nodiscard]] int64_t        matrixHeight()                                const { return height_;                         }
//    [[nodiscard]] int64_t        matrixWidth()                                 const { return width_;                          }
    [[nodiscard]] int64_t        matrixNumRowTiles()                           const { return tileGrid_.size();                }
    [[nodiscard]] int64_t        matrixNumColTiles()                           const { return tileGrid_[0].size();             }
    [[nodiscard]] int64_t         owner(int64_t rowIdx, int64_t colIdx)         const { return tileOwnership_[rowIdx][colIdx];  }
    std::shared_ptr<Tile>         tile(int64_t rowIdx, int64_t colIdx)                { return this->tileGrid_[rowIdx][colIdx]; }
//    [[nodiscard]] Dim             tileDimension(int64_t rowIdx, int64_t colIdx) const { return {tileWidth(rowIdx, colIdx), tileHeight(rowIdx, colIdx)}; }

    [[nodiscard]] int64_t tileHeight(int64_t rowIdx, [[maybe_unused]]int64_t colIdx) const {
        return std::min(tileDim_.y, int64_t(height_-tileDim_.y*rowIdx));
    }

    [[nodiscard]] int64_t tileWidth([[maybe_unused]]int64_t rowIdx, int64_t colIdx) const {
        return std::min(tileDim_.x, int64_t(width_-tileDim_.x*colIdx));
    }
//    [[nodiscard]] const MPI_Comm& mpiComm()                                 const { return mpiComm_; }
//    [[nodiscard]] int64_t        nodeId()                                  const { return nodeId_; }
//    [[nodiscard]] int64_t        numNodes()                                const { return numNodes_; }
//    [[nodiscard]] bool            isRootNodeId()                            const { return nodeId_ == 0; }
//    [[nodiscard]] bool            isLastNodeId()                            const { return nodeId_ == (numNodes_-1); }

    // Setters
    void tile(int64_t rowIdx, int64_t colIdx, std::shared_ptr<Tile> tile) {
        assert(owner(rowIdx, colIdx) != getNodeId());
        assert(tile->memoryOwner() == MemoryOwner::WORKSPACE);
        tileGrid_[rowIdx][colIdx] = tile;
    }

protected:
    Grid<std::shared_ptr<Tile>> tileGrid_      = {};
    Grid<int64_t>              tileOwnership_ = {};
    int64_t width_       = 0;
    int64_t height_      = 0;
    Dim      tileDim_     = {};
//    MPI_Comm mpiComm_     = {};
//    int64_t nodeId_      = 0;
//    int64_t numNodes_    = 0;
};

template<typename MatrixType, char Id>
class TwoDBlockCyclicMatrix: public MatrixContainer<MatrixType, Id> {
private:
    using Tile = MatrixTile<MatrixType, Id>;

public:
    explicit TwoDBlockCyclicMatrix(int64_t height, int64_t width, int64_t tileDim)
        : MatrixContainer<MatrixType, Id>(height, width, tileDim) {
        init();
    }

    bool init() override {
        auto nodeId = getNodeId();
        auto numNodes = getNumNodes();
        const auto [P, Q] = getGridDim();
        const auto MT = this->matrixNumRowTiles();
        const auto NT = this->matrixNumColTiles();
        int64_t p0 = nodeId/Q, q0 = nodeId%Q;
        // TODO: populate only the relevant MatrixTiles
        for(int64_t p = p0; p < int64_t(MT); p+=P) {
            for(int64_t q = q0; q < int64_t(NT); q+=Q) {
                auto tile = std::make_shared<Tile>(p, q, this->tileHeight(p, q), this->tileWidth(p, q));
                auto pData = (MatrixType *)tile->data();
                for(int64_t i = 0; i < tile->width()*tile->height(); ++i) pData[i] = 1;
                this->tileGrid_[p][q] = tile;
            }
        }

        for(int64_t p = 0; p < int64_t(MT); ++p) {
            for(int64_t q = 0; q < int64_t(NT); ++q) {
                this->tileOwnership_[p][q] = (q%Q + p*Q)%numNodes;
            }
        }

        return true;
    }

//    std::shared_ptr<Tile> tile(int64_t rowIdx, int64_t colIdx) override {
//        return this->tileGrid_[rowIdx][colIdx];
//    }

    int64_t typeId() override {
        return typeid(TwoDBlockCyclicMatrix).hash_code();
    }
};

template<char Id>
struct DbRequest {
    int64_t rowIdx = -1;
    int64_t colIdx = -1;
    bool    quit   = false;
//    int64_t priority = 0; // zero is the highest priority
//    int64_t deviceId = 0;
    explicit DbRequest(int64_t r, int64_t c, bool q = false): rowIdx(r), colIdx(c), quit(q) {}
};

#endif //HH3_MATMUL_DATA

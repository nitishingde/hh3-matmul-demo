#ifndef HH3_MATMUL_DATA_H
#define HH3_MATMUL_DATA_H

#include "common_data.h"

template<typename MatrixType, char IdA, char IdB, char IdC>
class GpuJob {
private:
    using TileA      = MatrixTile<MatrixType, IdA>;
    using TileB      = MatrixTile<MatrixType, IdB>;

public:
    explicit GpuJob(bool shouldQuit = false): quit_(shouldQuit), gpuToken_(std::make_shared<GpuToken>(-1)) {}

    ~GpuJob() {
        finished();
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
};

template<typename MatrixType, char Id>
struct GcMatrixTile {
    explicit GcMatrixTile(std::shared_ptr<MatrixTile<MatrixType, Id>> tile): tile(tile) {}
    std::shared_ptr<MatrixTile<MatrixType, Id>> tile = nullptr;
};

#endif //HH3_MATMUL_DATA_H

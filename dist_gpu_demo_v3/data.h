#ifndef HH3_MATMUL_DATA_H
#define HH3_MATMUL_DATA_H

#include <unordered_map>

#include "common_data.h"

template<typename MatrixType, char IdA, char IdB, char IdC>
class GpuJob {
private:
    using TileA      = MatrixTile<MatrixType, IdA>;
    using TileB      = MatrixTile<MatrixType, IdB>;
    using TileC      = MatrixTile<MatrixType, IdC>;

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
    [[nodiscard]] int32_t gpuId()            const { return gpuToken_->id;      }
    [[nodiscard]] bool    shouldQuit()       const { return quit_;              }
    [[nodiscard]] auto&   tilesFromMatrixC()       { return tileCs_;            }
    [[nodiscard]] bool    hasBeenProcessed()       { return isProcessed.load(); }

    // Setters
    void token(std::shared_ptr<GpuToken> token) { gpuToken_ = std::move(token); }
    void quit(const bool flag)                  { quit_ = flag;                 }
    void addTileC(std::shared_ptr<TileC> tileC) { tileCs_.emplace_back(tileC);  }
    void processed()                            { isProcessed.store(true);      }

public:
    int32_t                             height      = 0;
    int32_t                             width       = 0;

private:
    std::vector<std::shared_ptr<TileC>> tileCs_     = {};
    std::shared_ptr<GpuToken>           gpuToken_   = nullptr;
    bool                                quit_       = false;
    std::atomic_bool                    isProcessed = false;
};

struct GraphFilterState {
public:
    explicit GraphFilterState(const std::vector<int32_t> &deviceIds) {
        for(auto deviceId: deviceIds) {
            rowIndices.insert({deviceId, std::set<int64_t>{}});
            colIndices.insert({deviceId, std::set<int64_t>{}});
        }
    }

    std::unordered_map<int32_t, std::set<int64_t>> rowIndices = {};
    std::unordered_map<int32_t, std::set<int64_t>> colIndices = {};
};

#endif //HH3_MATMUL_DATA_H

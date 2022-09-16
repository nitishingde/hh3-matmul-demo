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


#ifndef HH3_MATMUL_CYCLIC2D_MATRIX_CONTAINER_H
#define HH3_MATMUL_CYCLIC2D_MATRIX_CONTAINER_H

#include "matrix_container_interface.h"

/**
 * Cyclic2D Matrix Container FIXME
 * Inherits: MatrixContainer
 *
 * @tparam MatrixType
 * @tparam Id
 * @tparam Ord
 */
template<class MatrixType, char Id, Order Ord = Order::Col>
class Cyclic2dMatrixContainer: public MatrixContainer<MatrixType, Id, Ord> {
public:
    explicit Cyclic2dMatrixContainer(const uint32_t contextId, const uint32_t matrixHeight, const uint32_t matrixWidth, const uint32_t tileSize, const MPI_Comm mpiComm)
        : MatrixContainer<MatrixType, Id, Ord>(contextId, matrixHeight, matrixWidth, tileSize, mpiComm) {
        assert(mpiComm != MPI_COMM_WORLD);
        grid_.resize(this->matrixNumRowTiles(), std::vector<std::shared_ptr<MatrixTile<MatrixType, Id, Ord>>>(this->matrixNumColTiles(), nullptr));
        init();
    }

    ~Cyclic2dMatrixContainer() {}

    bool init() override {
        // TODO: populate only the relevant MatrixTiles
        for(uint32_t idx = 0; idx < this->matrixNumRowTiles()*this->matrixNumColTiles(); ++idx) {
            uint32_t rowIdx = idx/this->matrixNumColTiles(), colIdx = idx%this->matrixNumColTiles();
            auto tile = std::make_shared<MatrixTile<MatrixType, Id, Ord>>(
                this->contextId(),
                idx%this->numNodes(),
                rowIdx, colIdx
            );
            grid_[rowIdx][colIdx] = std::move(tile);
        }
        return true;
    }

    std::shared_ptr<MatrixTile<MatrixType, Id, Ord>> getTile(uint32_t rowIdx, uint32_t colIdx) override {
        return grid_[rowIdx][colIdx];
    }

    uint32_t typeId() override {
        return typeid(Cyclic2dMatrixContainer).hash_code();
    }

    void shrink() {
        for(uint32_t i = 0; i < grid_.size(); ++i) {
            for(uint32_t j = 0; j < grid_[0].size(); ++j) {
                if(auto tile = grid_[i][j]; (tile == nullptr) or (tile->sourceNodeId() != this->nodeId())) {
                    grid_[i][j] = nullptr;
                }
            }
        }
    }

private:
    std::vector<std::vector<std::shared_ptr<MatrixTile<MatrixType, Id, Ord>>>> grid_ = {};
};

#endif //HH3_MATMUL_CYCLIC2D_MATRIX_CONTAINER_H

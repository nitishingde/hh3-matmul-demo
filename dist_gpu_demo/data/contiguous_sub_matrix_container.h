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


#ifndef HH3_MATMUL_CONTIGUOUS_SUB_MATRIX_CONTAINER_H
#define HH3_MATMUL_CONTIGUOUS_SUB_MATRIX_CONTAINER_H

#include "matrix_container_interface.h"

/**
 * Contiguous sub matrix container FIXME
 * Inherits: MatrixContainer
 * TODO: Add support to handle outside data
 *
 * @tparam subMatrixOrder
 * @tparam MatrixType
 * @tparam Id
 * @tparam Ord
 */
template<Order subMatrixOrder, class MatrixType, char Id, Order Ord = Order::Col>
class ContiguousSubMatrixContainer: public MatrixContainer<MatrixType, Id, Ord> {
public:
    explicit ContiguousSubMatrixContainer(const uint32_t contextId, const uint64_t matrixHeight, const uint64_t matrixWidth, const uint64_t tileSize, const MPI_Comm mpiComm)
        : MatrixContainer<MatrixType, Id, Ord>(contextId, matrixHeight, matrixWidth, tileSize, mpiComm) {

        uint64_t totalRowTiles = this->matrixNumRowTiles();
        uint64_t totalColTiles = this->matrixNumColTiles();
        if constexpr(subMatrixOrder == Order::Col) {
            uint64_t numColTiles = totalColTiles/this->numNodes();
            uint64_t noOfNodesWith1ExtraTile = totalColTiles%this->numNodes();
            numColTiles += (this->nodeId() < noOfNodesWith1ExtraTile? 1: 0);

            height_ = matrixHeight;
            width_ = !this->isLastNodeId()? numColTiles*tileSize: matrixWidth-((totalColTiles-numColTiles)*tileSize);

            rowTilesRange_[0] = 0;
            rowTilesRange_[1] = totalRowTiles;
            colTilesRange_[0] = (totalColTiles/this->numNodes())*this->nodeId() + std::min(uint64_t(this->nodeId()), noOfNodesWith1ExtraTile);
            colTilesRange_[1] = colTilesRange_[0]+numColTiles;
        }
        else {
            uint64_t numRowTiles = totalRowTiles/this->numNodes();
            uint64_t noOfNodesWith1ExtraTile = totalRowTiles%this->numNodes();
            numRowTiles += (this->nodeId() < noOfNodesWith1ExtraTile? 1: 0);

            height_ = !this->isLastNodeId()? numRowTiles*tileSize: matrixHeight-((totalRowTiles-numRowTiles)*tileSize);
            width_ = matrixWidth;

            rowTilesRange_[0] = (totalRowTiles/this->numNodes())*this->nodeId() + std::min(uint64_t(this->nodeId()), noOfNodesWith1ExtraTile);
            rowTilesRange_[1] = rowTilesRange_[0]+numRowTiles;
            colTilesRange_[0] = 0;
            colTilesRange_[1] = totalColTiles;
        }

        if constexpr(Ord == Order::Col) {
            leadingDimension_ = height_;
        }
        else {
            leadingDimension_ = width_;
        }

        pData_ = new MatrixType[height_*width_];
    }

    ~ContiguousSubMatrixContainer() {
        delete[] pData_;
        pData_ = nullptr;
    }

    bool init() override {
        return true;
    }

    std::shared_ptr<MatrixTile<MatrixType, Id, Ord>> getTile(uint64_t rowIdx, uint64_t colIdx) override {
        if((rowIdx < rowTilesRange_[0]) or (rowTilesRange_[1] <= rowIdx)) {
            return nullptr;
        }

        if((colIdx < colTilesRange_[0]) or (colTilesRange_[1] <= colIdx)) {
            return nullptr;
        }

        uint64_t i = rowIdx - rowTilesRange_[0];
        uint64_t j = colIdx - colTilesRange_[0];
        if constexpr(Ord == Order::Row) assert(false);//FIXME
        return std::make_shared<MatrixTile<MatrixType, Id, Ord>>(
            this->contextId(),
            this->nodeId(),
            rowIdx, colIdx,
            leadingDimension_, &pData_[j*leadingDimension_*this->matrixTileSize() + i*this->matrixTileSize()]
        );
    }

    uint64_t typeId() override {
        return typeid(ContiguousSubMatrixContainer).hash_code();
    }

    // Getters/Setters
    [[nodiscard]] uint64_t subMatrixHeight() const { return height_; }
    [[nodiscard]] uint64_t subMatrixWidth() const { return width_; }
    [[nodiscard]] uint64_t subMatrixNumRowTiles() const { return rowTilesRange_[1]-rowTilesRange_[0]; }
    [[nodiscard]] uint64_t subMatrixNumColTiles() const { return colTilesRange_[1]-colTilesRange_[0]; }
    [[nodiscard]] std::tuple<uint64_t, uint64_t> subMatrixRowTileRange() const { return std::make_tuple(rowTilesRange_[0], rowTilesRange_[1]); }
    [[nodiscard]] std::tuple<uint64_t, uint64_t> subMatrixColTileRange() const { return std::make_tuple(colTilesRange_[0], colTilesRange_[1]); }
    [[nodiscard]] uint64_t leadingDimension() const { return leadingDimension_; }
    [[nodiscard]] MatrixType* data() const { return pData_; }
    [[nodiscard]] uint64_t dataSize() const { return height_*width_; }

    friend std::ostream& operator<<(std::ostream &os, const ContiguousSubMatrixContainer &data) {
        os << "Contiguous SubMatrix Data " << Id
           << "\nmatrix size: (" << data.matrixHeight() << ", " << data.matrixWidth() << ")"
           << "\nmatrix grid size: (" << data.matrixNumRowTiles() << ", " << data.matrixNumColTiles() << ")"
           << "\nsub matrix size: (" << data.subMatrixHeight() << ", " << data.subMatrixWidth() << ")"
           << "\nsub matrix grid size: (" << data.subMatrixNumRowTiles() << ", " << data.subMatrixNumColTiles() << ")"
           << "\nmatrix/submatrix tile size: " << data.matrixTileSize() << " leading dimension: " << data.leadingDimension()
           << std::endl;

        if constexpr(Ord == Order::Col) {
            for(uint64_t i = 0; i < data.subMatrixHeight(); ++i) {
                for(uint64_t j = 0; j < data.subMatrixWidth(); ++j) {
                    os << std::setprecision(std::numeric_limits<MatrixType>::digits10 + 1)
                       << data.pData_[j*data.leadingDimension() + i] << " ";
                }
                os << std::endl;
            }
        } else {
            for(uint64_t i = 0; i < data.subMatrixHeight(); ++i) {
                for(uint64_t j = 0; j < data.subMatrixWidth(); ++j) {
                    os << std::setprecision(std::numeric_limits<MatrixType>::digits10 + 1)
                       << data.pData_[i*data.leadingDimension() + j] << " ";
                }
                os << std::endl;
            }
        }
        os << std::endl;

        return os;
    }

private:
    uint64_t height_           = 0;
    uint64_t width_            = 0;
    uint64_t rowTilesRange_[2] = {0, 0};
    uint64_t colTilesRange_[2] = {0, 0};
    uint64_t leadingDimension_ = 0;
    MatrixType *pData_         = nullptr;
};

#endif //HH3_MATMUL_CONTIGUOUS_SUB_MATRIX_CONTAINER_H

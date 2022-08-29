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
    explicit ContiguousSubMatrixContainer(const uint32_t contextId, const uint32_t matrixHeight, const uint32_t matrixWidth, const uint32_t tileSize, const MPI_Comm mpiComm)
        : MatrixContainer<MatrixType, Id, Ord>(contextId, matrixHeight, matrixWidth, tileSize, mpiComm) {

        if constexpr(subMatrixOrder == Order::Col) {
            height_ = matrixHeight;
            width_ = matrixWidth/this->numNodes();
            width_ += (this->nodeId() < matrixWidth%this->numNodes()? 1: 0);
        }
        else {
            height_ = matrixHeight/this->numNodes();
            height_ += (this->nodeId() < matrixHeight%this->numNodes()? 1: 0);
            width_ = matrixWidth;
        }

        numColTiles_ = (width_+tileSize-1)/tileSize;
        numRowTiles_ = (height_+tileSize-1)/tileSize;

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

    std::shared_ptr<MatrixTile<MatrixType, Id, Ord>> getTile(uint32_t rowIdx, uint32_t colIdx) override {
        if(numRowTiles_ <= rowIdx or numColTiles_ <= colIdx) {
            return nullptr;
        }

        return std::make_shared<MatrixTile<MatrixType, Id, Ord>>(
            this->contextId(),
            this->nodeId(),
            rowIdx, colIdx,
            leadingDimension_, pData_
        );
    }

    uint32_t typeId() override {
        return typeid(ContiguousSubMatrixContainer).hash_code();
    }

    // Getters/Setters
    [[nodiscard]] uint32_t subMatrixHeight() const { return height_; }
    [[nodiscard]] uint32_t subMatrixWidth() const { return width_; }
    [[nodiscard]] uint32_t subMatrixNumRowTiles() const { return numRowTiles_; }
    [[nodiscard]] uint32_t subMatrixNumColTiles() const { return numColTiles_; }
    [[nodiscard]] uint32_t leadingDimension() const { return leadingDimension_; }
    [[nodiscard]] MatrixType* data() const { return pData_; }

    friend std::ostream& operator<<(std::ostream &os, const ContiguousSubMatrixContainer &data) {
        os << "Contiguous SubMatrix Data " << Id
           << " matrix size: (" << data.matrixHeight() << ", " << data.matrixWidth() << ")"
           << " matrix grid size: (" << data.matrixHeight() << ", " << data.matrixWidth() << ")"
           << " sub matrix size: (" << data.subMatrixHeight() << ", " << data.subMatrixWidth() << ")"
           << " sub matrix grid size: (" << data.subMatrixNumRowTiles() << ", " << data.subMatrixNumColTiles() << ")"
           << " matrix/submatrix tile size: " << data.matrixTileSize() << " leading dimension: " << data.leadingDimension()
           << std::endl;

        if constexpr(Ord == Order::Col) {
            for(size_t i = 0; i < data.matrixHeight(); ++i) {
                for(size_t j = 0; j < data.matrixWidth(); ++j) {
                    os << std::setprecision(std::numeric_limits<MatrixType>::digits10 + 1)
                       << data.pData_[j*data.leadingDimension() + i] << " ";
                }
                os << std::endl;
            }
        } else {
            for(size_t i = 0; i < data.matrixHeight(); ++i) {
                for(size_t j = 0; j < data.matrixWidth(); ++j) {
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
    uint32_t height_           = 0;
    uint32_t width_            = 0;
    uint32_t numRowTiles_      = 0;
    uint32_t numColTiles_      = 0;
    uint32_t leadingDimension_ = 0;
    MatrixType *pData_         = nullptr;
};

#endif //HH3_MATMUL_CONTIGUOUS_SUB_MATRIX_CONTAINER_H

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


#ifndef HH3_MATMUL_REDUNDANT_MATRIX_CONTAINER_H
#define HH3_MATMUL_REDUNDANT_MATRIX_CONTAINER_H

#include "matrix_container_interface.h"

/**
 * Redundant Matrix Container FIXME
 * Inherits: MatrixContainer
 *
 * @tparam MatrixType
 * @tparam Id
 * @tparam Ord
 */
template<class MatrixType, char Id, Order Ord = Order::Col>
class RedundantMatrixContainer: public MatrixContainer<MatrixType, Id, Ord> {
public:
    explicit RedundantMatrixContainer(const uint32_t contextId, const uint64_t matrixHeight, const uint64_t matrixWidth, const uint64_t tileSize, const MPI_Comm mpiComm, const bool isSourceNode = false, MatrixType *pMatrixData = nullptr)
        : MatrixContainer<MatrixType, Id, Ord>(contextId, matrixHeight, matrixWidth, tileSize, mpiComm),
          pData_(pMatrixData), isSelfAllocated_(pMatrixData == nullptr),
          isSourceNode_(isSourceNode) {

        int32_t sourceNode = isSourceNode? this->nodeId(): 0;
        MPI_Allreduce(&sourceNode, &sourceNode_, 1, MPI_INT32_T, MPI_SUM, this->mpiComm_);
#if not NDEBUG
        if(isSourceNode) {
            assert(sourceNode == this->nodeId());
        }
#endif
        if(isSelfAllocated_) {
            pData_ = new MatrixType[matrixHeight*matrixWidth];
        }
    }

    ~RedundantMatrixContainer() {
        if(isSelfAllocated_) {
            delete[] pData_;
            pData_ = nullptr;
        }
    }

    bool init() override {
        return false;
    }

    std::shared_ptr<MatrixTile<MatrixType, Id, Ord>> getTile(uint64_t rowIdx, uint64_t colIdx) override {
        return std::make_shared<MatrixTile<MatrixType, Id, Ord>>(
            this->contextId(),
            sourceNode_,
            rowIdx, colIdx,
            leadingDimension(), &pData_[colIdx*leadingDimension()*this->matrixTileSize() + rowIdx*this->matrixTileSize()]
        );
    }

    uint64_t typeId() override {
        return typeid(RedundantMatrixContainer).hash_code();
    }

    // Getters/Setters
    [[nodiscard]] uint64_t leadingDimension() const {
        if constexpr(Ord == Order::Col) {
            return this->matrixHeight();
        }
        return this->matrixWidth();
    }
    [[nodiscard]] MatrixType* data() { return pData_; }
    [[nodiscard]] uint64_t dataSize() { return this->matrixHeight()*this->matrixWidth(); }

    friend std::ostream& operator<<(std::ostream &os, const RedundantMatrixContainer &data) {
        os << "Redundant Matrix Data " << Id
           << " matrix size: (" << data.matrixHeight() << ", " << data.matrixWidth() << ")"
           << " matrix grid size: (" << data.matrixNumRowTiles() << ", " << data.matrixNumColTiles() << ")"
           << " matrix tile size: " << data.matrixTileSize() << " leading dimension: " << data.leadingDimension()
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
    MatrixType *pData_       = nullptr;
    bool isSelfAllocated_    = true;
    int32_t sourceNode_      = 0;
    bool isSourceNode_       = false;
};

#endif //HH3_MATMUL_REDUNDANT_MATRIX_CONTAINER_H

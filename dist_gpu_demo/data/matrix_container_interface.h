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


#ifndef HH3_MATMUL_MATRIX_CONTAINER_INTERFACE_H
#define HH3_MATMUL_MATRIX_CONTAINER_INTERFACE_H

#include "matrix_meta_data.h"
#include "matrix_tile.h"
#include "../utility.h"

/**
 * Interface to store matrix meta data and data in a distributed fashion.
 * The subclasses inheriting MatrixContainer will need to implement the appropriate virtual functions only, and not
 * necessarily all of them.
 *
 * @tparam MatrixType : float/double
 * @tparam Id : identifier
 * @tparam Ord : column/row
 */
template<class MatrixType, char Id, Order Ord = Order::Col>
requires std::is_floating_point_v<MatrixType> //FIXME: use concepts
class MatrixContainer {
public:
    explicit MatrixContainer(const uint32_t contextId, const uint64_t matrixHeight, const uint64_t matrixWidth, const uint64_t tileSize, const MPI_Comm mpiComm = MPI_COMM_WORLD):
        contextId_(contextId), mpiComm_(mpiComm) {
        matrixMetaData_ = {
            .matrixHeight = matrixHeight,
            .matrixWidth = matrixWidth,
//            .tileHeight = tileSize,
//            .tileWidth = tileSize,
            .tileSize = tileSize,
            .numRowTiles = (matrixHeight+tileSize-1)/tileSize,
            .numColTiles = (matrixWidth+tileSize-1)/tileSize
        };
        registerContext(contextId, matrixMetaData_);
#if NDEBUG
        MPI_Comm_rank(mpiComm_, &nodeId_);
        MPI_Comm_size(mpiComm_, &numNodes_);
#else
        if((MPI_Comm_rank(mpiComm_, &nodeId_) != MPI_SUCCESS) or (MPI_Comm_size(mpiComm_, &numNodes_) != MPI_SUCCESS)) {
            MPI_Abort(this->mpiComm_, EXIT_FAILURE);
        }
#endif
    }

    virtual bool init() = 0;//FIXME: needed?
    /**
    virtual std::shared_ptr<MatrixTile<MatrixType, Id, Ord>> getTile(uint64_t rowIdx, uint64_t colIdx) = 0;
    virtual uint64_t typeId() = 0;

    // Getters/Setters
    [[nodiscard]] uint32_t contextId() { return contextId_; }
    [[nodiscard]] MatrixMetaData matrixMetaData() const { return matrixMetaData_; }
    [[nodiscard]] uint64_t matrixHeight() const { return matrixMetaData_.matrixHeight; }
    [[nodiscard]] uint64_t matrixWidth() const { return matrixMetaData_.matrixWidth; }
//    [[nodiscard]] uint64_t matrixTileHeight() const { return matrixMetaData_.tileHeight; }
//    [[nodiscard]] uint64_t matrixTileWidth() const { return matrixMetaData_.tileWidth; }
    [[nodiscard]] uint64_t matrixTileSize() const { return matrixMetaData_.tileSize; }
    [[nodiscard]] uint64_t matrixNumRowTiles() const { return matrixMetaData_.numRowTiles; }
    [[nodiscard]] uint64_t matrixNumColTiles() const { return matrixMetaData_.numColTiles; }
    [[nodiscard]] const MPI_Comm& mpiComm() const { return mpiComm_; }
    [[nodiscard]] uint32_t nodeId() const { return nodeId_; }
    [[nodiscard]] uint32_t numNodes() const { return numNodes_; }

protected:
    uint32_t contextId_            = 0;
    MatrixMetaData matrixMetaData_ = {};
    MPI_Comm mpiComm_              = {};
    int32_t nodeId_                = 0;
    int32_t numNodes_              = 1;
};

#endif //HH3_MATMUL_MATRIX_CONTAINER_INTERFACE_H

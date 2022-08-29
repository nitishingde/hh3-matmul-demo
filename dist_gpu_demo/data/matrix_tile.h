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


#ifndef HH3_MATMUL_MATRIX_TILE_H
#define HH3_MATMUL_MATRIX_TILE_H

#include "data_packet.h"
#include "matrix_order.h"
#include "matrix_tile_meta_data.h"
#include "ttl_managed_memory.h"
#include "../utility.h"
#include <memory>
#include <hedgehog/hedgehog.h>

/**
 * MatrixTile
 * Inherits: TtlManagedMemory to enable this class be managed by hedgehog's memory manager with ttl counter.
 *
 * @tparam MatrixType
 * @tparam Id
 * @tparam Ord
 */
template<class MatrixType, char Id, Order Ord = Order::Col>
class MatrixTile: public TtlManagedMemory {//FIXME: override methods
public:
    explicit MatrixTile(
        uint32_t contextId,
        uint32_t sourceNodeId,
        uint32_t rowIdx, uint32_t colIdx,
        uint32_t leadingDimension = 0, MatrixType *pData = nullptr
    ):  contextId_(contextId), matrixMetaData_(getContext<MatrixMetaData>(contextId)),
        leadingDimension_(leadingDimension), pData_(pData), isSelfAllocated_(pData == nullptr) {

        matrixTileMetaData_ = {
            .sourceNodeId = sourceNodeId,
            .rowIdx = rowIdx,
            .colIdx = colIdx,
            .height = std::min(matrixMetaData_.matrixHeight - rowIdx*matrixMetaData_.tileSize, matrixMetaData_.tileSize),
            .width = std::min(matrixMetaData_.matrixWidth - colIdx*matrixMetaData_.tileSize, matrixMetaData_.tileSize)
        };

        if(isSelfAllocated_) {
            uint32_t bufferSize = (matrixMetaData_.tileSize*matrixMetaData_.tileSize+2)*sizeof(MatrixType);//FIXME
            dataPacket_ = std::make_shared<DataPacket>(contextId, bufferSize);
            pData_ = reinterpret_cast<MatrixType*>(dataPacket_->data());
            pData_[(matrixMetaData_.tileSize*matrixMetaData_.tileSize)+0] = MatrixType(rowIdx);
            pData_[(matrixMetaData_.tileSize*matrixMetaData_.tileSize)+1] = MatrixType(colIdx);
            if constexpr(Ord == Order::Col) {
                leadingDimension_ = matrixTileMetaData_.height;
            }
            else {
                leadingDimension_ = matrixTileMetaData_.width;
            }
        }
    }

    explicit MatrixTile(uint32_t tileSize): leadingDimension_(tileSize), isSelfAllocated_(true) {
        matrixTileMetaData_.height = tileSize;
        matrixTileMetaData_.width = tileSize;
        uint32_t bufferSize = (tileSize*tileSize+2)*sizeof(MatrixType);
        dataPacket_ = std::make_shared<DataPacket>(-1, bufferSize);
        pData_ = reinterpret_cast<MatrixType*>(dataPacket_->data());
    }

    ~MatrixTile() {
        if(dataPacket_) dataPacket_->setToRecycle();
    }

    bool operator!=(const MatrixTile<MatrixType, Id, Ord> &other) {
        if(height() != other.height()) return true;
        if(width() != other.width()) return true;
        if(rowIdx() != other.rowIdx()) return true;
        if(colIdx() != other.colIdx()) return true;

        if constexpr(Ord == Order::Col) {
            for(uint32_t j = 0; j < this->width(); ++j) {
                for(uint32_t i = 0; i < this->height(); ++i) {
                    if(0.01 < std::abs(data()[j*leadingDimension()+i] - other.data()[j*other.leadingDimension()+i])) {
                        return true;
                    }
                }
            }
        }
        else {
            // FIXME
            throw std::runtime_error("Order::Row not implemented");
        }

        return false;
    }

    // Getters/Setters
    [[nodiscard]] uint32_t contextId() const { return contextId_; }
    [[nodiscard]] const MatrixMetaData& matrixMetaData() const { return matrixMetaData_; }
    [[nodiscard]] const MatrixTileMetaData& matrixTileMetaData() const { return matrixTileMetaData_; }
    void matrixTileMetaData(const MatrixTileMetaData &matrixTileMetaData) { matrixTileMetaData_ = matrixTileMetaData; }
    [[nodiscard]] uint32_t sourceNodeId() const { return matrixTileMetaData_.sourceNodeId; }
    [[nodiscard]] uint32_t rowIdx() const { return matrixTileMetaData_.rowIdx; }
    [[nodiscard]] uint32_t colIdx() const { return matrixTileMetaData_.colIdx; }
    [[nodiscard]] uint32_t height() const { return matrixTileMetaData_.height; }
    [[nodiscard]] uint32_t width() const { return matrixTileMetaData_.width; }
    [[nodiscard]] uint32_t leadingDimension() const { return leadingDimension_; }
    [[nodiscard]] MatrixType* data() const { return pData_; }
    [[nodiscard]] uint32_t dataSize() const { return matrixTileMetaData_.height*matrixTileMetaData_.width; }
    [[nodiscard]] std::shared_ptr<DataPacket>& dataPacket() { return dataPacket_; }

    [[nodiscard]] std::shared_ptr<DataPacket> serializedBytes(std::shared_ptr<DataPacket> dataPacket = nullptr) const {
        if(isSelfAllocated_) {
            return dataPacket_;
        }
#if not NDEBUG
        if(dataPacket == nullptr) {
            throw std::runtime_error("MatrixTile which is not self allocated needs non null dataPacket argument for storing the serialization data.");
        }

        if(dataPacket->size() < this->serializedSizeInBytes()) {
            throw std::runtime_error("DataPacket provided for serialization is insufficient, check serializedSizeInBytes().");
        }
#endif
        // TODO
        // std::memcpy();
        return nullptr;
    }
    [[nodiscard]] uint32_t serializedSizeInBytes() const { return (dataSize()+2)*sizeof(MatrixType); }//FIXME
    void deserializeBytes() {
        // FIXME
        contextId_ = dataPacket_->contextId();
        matrixMetaData_ = getContext<MatrixMetaData>(contextId_);
        pData_ = reinterpret_cast<MatrixType*>(dataPacket_->data());
        uint32_t rowIdx = pData_[(matrixMetaData_.tileSize*matrixMetaData_.tileSize)+0];
        uint32_t colIdx = pData_[(matrixMetaData_.tileSize*matrixMetaData_.tileSize)+1];
        matrixTileMetaData_ = {
            .sourceNodeId = uint32_t(getNodeId()),
            .rowIdx = rowIdx,
            .colIdx = colIdx,
            .height = std::min(matrixMetaData_.matrixHeight - rowIdx*matrixMetaData_.tileSize, matrixMetaData_.tileSize),
            .width = std::min(matrixMetaData_.matrixWidth - colIdx*matrixMetaData_.tileSize, matrixMetaData_.tileSize)
        };
    }

    // debug
    friend std::ostream& operator<<(std::ostream &os, MatrixTile const &data) {
        // FIXME
        os << "MatrixTile " << Id << " position Grid: (" << data.rowIdx() << ", " << data.colIdx() << ")" << std::endl;
        os << "Tile: (" << data.height() << ", " << data.width() << ") leadingDimension = " << data.leadingDimension() << " sourceNodeId = " << data.sourceNodeId() << std::endl;

        if constexpr(Ord == Order::Col) {
            for(size_t i = 0; i < data.height(); ++i) {
                for(size_t j = 0; j < data.width(); ++j) {
                    os << data.data()[j*data.leadingDimension() + i] << " ";
                }
                os << std::endl;
            }
        } else {
            for(size_t i = 0; i < data.height(); ++i) {
                for(size_t j = 0; j < data.width(); ++j) {
                    os << data.data()[i*data.leadingDimension() + j] << " ";
                }
                os << std::endl;
            }
        }

        return os;
    }

private:
    uint32_t contextId_                     = 0;//FIXME: 2 places where contextId is being stored, here and in dataPacket
    MatrixMetaData matrixMetaData_          = {};
    MatrixTileMetaData matrixTileMetaData_  = {};
    uint32_t leadingDimension_              = 0;
    std::shared_ptr<DataPacket> dataPacket_ = nullptr;
    MatrixType *pData_                      = nullptr;
    bool isSelfAllocated_                   = true;
};

#endif //HH3_MATMUL_MATRIX_TILE_H

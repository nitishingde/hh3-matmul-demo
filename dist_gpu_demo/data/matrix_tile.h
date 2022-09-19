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
        uint32_t matrixContextId,
        uint32_t sourceNodeId,
        uint64_t rowIdx, uint64_t colIdx,
        uint64_t leadingDimension = 0, MatrixType *pTileData = nullptr
    ): sourceNodeId_(sourceNodeId),
       leadingDimension_(leadingDimension), pData_(pTileData), isSelfAllocated_(pTileData == nullptr) {

        initMetaData(matrixContextId, rowIdx, colIdx);

        if(isSelfAllocated_) {
            dataPacket_ = std::make_shared<DataPacket>(matrixContextId, dataPacketSizeInBytes());
            *reinterpret_cast<uint64_t*>(&dataPacket_->data()[rowIdxByteOffset()]) = rowIdx;
            *reinterpret_cast<uint64_t*>(&dataPacket_->data()[colIdxByteOffset()]) = colIdx;
            pData_ = reinterpret_cast<MatrixType*>(&dataPacket_->data()[dataByteOffset()]);
            if constexpr(Ord == Order::Col) {
                leadingDimension_ = matrixTileMetaData_.height;
            }
            else {
                leadingDimension_ = matrixTileMetaData_.width;
            }
        }
    }

    explicit MatrixTile(uint64_t tileSize): leadingDimension_(tileSize), isSelfAllocated_(true) {
        matrixTileMetaData_.height = tileSize;
        matrixTileMetaData_.width = tileSize;
        uint64_t bufferSize = (tileSize*tileSize+2)*sizeof(MatrixType);
        dataPacket_ = std::make_shared<DataPacket>(-1, bufferSize);//FIXME: contextId
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
            for(uint64_t j = 0; j < this->width(); ++j) {
                for(uint64_t i = 0; i < this->height(); ++i) {
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
    [[nodiscard]] uint32_t contextId() const { return matrixContextId_; }
    [[nodiscard]] const MatrixMetaData& matrixMetaData() const { return matrixMetaData_; }
    [[nodiscard]] const MatrixTileMetaData& matrixTileMetaData() const { return matrixTileMetaData_; }
    void matrixTileMetaData(const MatrixTileMetaData &matrixTileMetaData) { matrixTileMetaData_ = matrixTileMetaData; }
    [[nodiscard]] int32_t sourceNodeId() const { return sourceNodeId_; }
    [[nodiscard]] uint64_t rowIdx() const { return matrixTileMetaData_.rowIdx; }
    [[nodiscard]] uint64_t colIdx() const { return matrixTileMetaData_.colIdx; }
    [[nodiscard]] uint64_t height() const { return matrixTileMetaData_.height; }
    [[nodiscard]] uint64_t width() const { return matrixTileMetaData_.width; }
    [[nodiscard]] uint64_t leadingDimension() const { return leadingDimension_; }
    [[nodiscard]] MatrixType* data() const { return pData_; }
    [[nodiscard]] uint64_t dataSize() const { return matrixTileMetaData_.height*matrixTileMetaData_.width; }
    [[nodiscard]] std::shared_ptr<DataPacket>& dataPacket() { return dataPacket_; }

    [[nodiscard]] std::shared_ptr<DataPacket> packDataPacket(std::shared_ptr<DataPacket> dataPacket = nullptr) const {
        if(isSelfAllocated_) {
            return dataPacket_;
        }

        assert(dataPacket != nullptr);
        assert(this->dataPacketSizeInBytes() <= dataPacket->size());

        if constexpr(Ord == Order::Col) {
            for(uint64_t j = 0, offset = dataByteOffset(); j < matrixTileMetaData_.width; ++j) {
                std::memcpy(&dataPacket->data()[offset], &pData_[j*leadingDimension_], sizeof(MatrixType)*matrixTileMetaData_.height);
                offset += (matrixTileMetaData_.height*sizeof(MatrixType));
            }
        } else {
            for(uint64_t i = 0, offset = dataByteOffset(); i < matrixTileMetaData_.height; ++i) {
                std::memcpy(&dataPacket->data()[offset], &pData_[i*leadingDimension_], sizeof(MatrixType)*matrixTileMetaData_.width);
                offset += (matrixTileMetaData_.width*sizeof(MatrixType));
            }
        }

        return nullptr;
    }
    [[nodiscard]] uint64_t dataPacketSizeInBytes() const { return 2*sizeof(uint64_t) + matrixMetaData_.tileSize*matrixMetaData_.tileSize*sizeof(MatrixType); }
    void unPackDataPacket(std::shared_ptr<DataPacket> dataPacket = nullptr) {
        assert(isSelfAllocated_);//FIXME
        uint64_t rowIdx = *reinterpret_cast<uint64_t*>(&dataPacket_->data()[rowIdxByteOffset()]);
        uint64_t colIdx = *reinterpret_cast<uint64_t*>(&dataPacket_->data()[colIdxByteOffset()]);
        initMetaData(dataPacket_->contextId(), rowIdx, colIdx);
        pData_ = reinterpret_cast<MatrixType*>(&dataPacket_->data()[dataByteOffset()]);
        sourceNodeId_ = -1;//FIXME
        if constexpr(Ord == Order::Col) {
            leadingDimension_ = matrixTileMetaData_.height;
        }
        else {
            leadingDimension_ = matrixTileMetaData_.width;
        }
    }

    // debug
    friend std::ostream& operator<<(std::ostream &os, MatrixTile const &data) {
        // FIXME
        os << "MatrixTile " << Id << " position Grid: (" << data.rowIdx() << ", " << data.colIdx() << ")" << std::endl;
        os << "Tile: (" << data.height() << ", " << data.width() << ") leadingDimension = " << data.leadingDimension() << " sourceNodeId = " << data.sourceNodeId() << std::endl;

        if constexpr(Ord == Order::Col) {
            for(uint64_t i = 0; i < data.height(); ++i) {
                for(uint64_t j = 0; j < data.width(); ++j) {
                    os << data.data()[j*data.leadingDimension() + i] << " ";
                }
                os << std::endl;
            }
        } else {
            for(uint64_t i = 0; i < data.height(); ++i) {
                for(uint64_t j = 0; j < data.width(); ++j) {
                    os << data.data()[i*data.leadingDimension() + j] << " ";
                }
                os << std::endl;
            }
        }

        return os;
    }

private:
    void initMetaData(uint32_t matrixContextId, uint64_t rowIdx, uint64_t colIdx) {
        matrixContextId_= matrixContextId;
        matrixMetaData_ = getContext<MatrixMetaData>(matrixContextId);
        matrixTileMetaData_ = {
            .rowIdx = rowIdx,
            .colIdx = colIdx,
            .height = std::min(matrixMetaData_.matrixHeight - rowIdx*matrixMetaData_.tileSize, matrixMetaData_.tileSize),
            .width  = std::min(matrixMetaData_.matrixWidth - colIdx*matrixMetaData_.tileSize, matrixMetaData_.tileSize)
        };
    }
    [[nodiscard]] constexpr uint64_t rowIdxByteOffset() { return 0; }
    [[nodiscard]] constexpr uint64_t colIdxByteOffset() { return sizeof(uint64_t); }
    [[nodiscard]] constexpr uint64_t dataByteOffset() { return 2*sizeof(uint64_t); }

private:
    uint32_t matrixContextId_               = 0;//FIXME: 2 places where contextId is being stored, here and in dataPacket
    MatrixMetaData matrixMetaData_          = {};
    MatrixTileMetaData matrixTileMetaData_  = {};
    int32_t sourceNodeId_                   = 0;//network related
    uint64_t leadingDimension_              = 0;
    std::shared_ptr<DataPacket> dataPacket_ = nullptr;
    MatrixType *pData_                      = nullptr;
    bool isSelfAllocated_                   = true;
};

#endif //HH3_MATMUL_MATRIX_TILE_H

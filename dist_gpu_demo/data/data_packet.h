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


#ifndef HH3_MATMUL_DATA_PACKET_H
#define HH3_MATMUL_DATA_PACKET_H

#include <hedgehog/hedgehog.h>

/**
 * A type of buffer class
 * Inherits: hh::ManagedMemory to enable this class be managed by hedgehog's memory manager.
 */
class DataPacket: public hh::ManagedMemory {
public:
    explicit DataPacket(uint32_t contextId, uint32_t bufferSize): contextId_(contextId), bufferSize_(bufferSize) {
        pBufferData_ = new uint8_t[bufferSize];
    }

    ~DataPacket() {
        delete[] pBufferData_;
        pBufferData_ = nullptr;
        contextId_ = bufferSize_ = 0;
    }

    [[nodiscard]] bool canBeRecycled() override { return canBeRecycled_; }
    void clean() override { canBeRecycled_ = false; }

    // Getters/Setters
    [[nodiscard]] uint32_t contextId() { return contextId_; }
    void contextId(uint32_t contextId) { contextId_ = contextId; }
    [[nodiscard]] uint8_t* data() { return pBufferData_; }
    [[nodiscard]] uint32_t size() { return bufferSize_; }
    void setToRecycle() { canBeRecycled_ = true; }

private:
    uint32_t contextId_   = 0;
    uint8_t *pBufferData_ = nullptr;
    uint32_t bufferSize_  = 0;
    bool canBeRecycled_   = false;
};

#endif //HH3_MATMUL_DATA_PACKET_H

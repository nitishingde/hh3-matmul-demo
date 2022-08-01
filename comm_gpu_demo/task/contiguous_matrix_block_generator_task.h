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


#ifndef HH3_MATMUL_CONTIGUOUS_MATRIX_BLOCK_GENERATOR_TASK_H
#define HH3_MATMUL_CONTIGUOUS_MATRIX_BLOCK_GENERATOR_TASK_H

#include <hedgehog/hedgehog.h>
#include "../data/matrix_meta_data.h"

template<class MatrixType, char Id, Order Ord>
class ContiguousMatrixBlockGeneratorTask: public hh::AbstractTask<1, MatrixMetaData, MatrixBlockData<MatrixType, Id, Ord>> {
public:
void execute(std::shared_ptr<MatrixMetaData> matrixMetaData) override {
auto blockSize = matrixMetaData->blockSize;
for(size_t row = 0; row*blockSize < matrixMetaData->height; ++row) {
for(size_t col = 0; col*blockSize < matrixMetaData->width; ++col) {
auto blockC = std::static_pointer_cast<MatrixBlockData<MatrixType, Id, Ord>>(this->getManagedMemory());
// TODO: if memory manager is not connected:
// auto blockC = std::make_shared<MatrixBlockData<MatrixType, Id, Ord>>(blockSize);
memset(blockC->blockData(), 0, sizeof(MatrixType)*matrixMetaData->blockSize*matrixMetaData->blockSize);
blockC->rowIdx(row);
blockC->colIdx(col);
blockC->blockSizeHeight(std::min(blockSize, matrixMetaData->height - row*blockSize));
blockC->blockSizeWidth(std::min(blockSize, matrixMetaData->width - col*blockSize));
blockC->leadingDimension(blockSize);
this->addResult(blockC);
}
}
}
};


#endif //HH3_MATMUL_CONTIGUOUS_MATRIX_BLOCK_GENERATOR_TASK_H

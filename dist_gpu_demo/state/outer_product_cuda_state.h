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


#ifndef HH3_MATMUL_OUTER_PRODUCT_CUDA_STATE_H
#define HH3_MATMUL_OUTER_PRODUCT_CUDA_STATE_H

#include "../data/cuda_matrix_tile.h"
#include <hedgehog/hedgehog.h>

template<class MatrixType, char InpIdA, char InpIdB, Order Ord>
class OuterProductCudaState:
    public hh::State<2,
        CudaMatrixTile<MatrixType, InpIdA, Ord>,                                                                                        //inp1
        CudaMatrixTile<MatrixType, InpIdB, Ord>,                                                                                        //inp2
        std::pair<std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>>, std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>>>   //out1
    > {
public:
    OuterProductCudaState(uint32_t mTiles, uint32_t kTiles, uint32_t nTiles): mTiles_(mTiles), kTiles_(kTiles), nTiles_(nTiles) {
        gridMatrixA_ = std::vector<std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>>>(mTiles_*kTiles_, nullptr);
        gridMatrixB_ = std::vector<std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>>>(nTiles_*kTiles_, nullptr);
        ttlA_ = std::vector<int32_t>(mTiles_*kTiles_, nTiles_);
        ttlB_ = std::vector<int32_t>(nTiles_*kTiles_, mTiles_);
    }

    virtual ~OuterProductCudaState() = default;

    void execute(std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>> tileA) override {
        tileA->ttl(nTiles_);
        matrixA(tileA);
        uint32_t iA = tileA->rowIdx()*kTiles_ + tileA->colIdx();
        for(size_t jB = 0; jB < nTiles_; ++jB) {
            if(auto tileB = matrixB(tileA->colIdx(), jB); tileB != nullptr) {
                ttlA_[iA]--;
                if(ttlA_[iA] == 0) {
                    gridMatrixA_[iA] = nullptr;
                }
                this->addResult(std::make_shared<std::pair<
                    std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>>,
                    std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>>
                >>(tileA, tileB));
            }
        }
    }

    void execute(std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>> tileB) override {
        tileB->ttl(mTiles_);
        matrixB(tileB);
        uint32_t jB = tileB->rowIdx()*nTiles_ + tileB->colIdx();
        for(size_t iA = 0; iA < mTiles_; ++iA) {
            if(auto tileA = matrixA(iA, tileB->rowIdx()); tileA != nullptr) {
                ttlB_[jB]--;
                if(ttlB_[jB] == 0) {
                    gridMatrixB_[jB] = nullptr;
                }
                this->addResult(std::make_shared<std::pair<
                    std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>>,
                    std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>>
                >>(tileA, tileB));
            }
        }
    }

private:
    [[nodiscard]] std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>> matrixA(size_t i, size_t j) {
        uint32_t idx = i*kTiles_ + j;
        if(auto res = gridMatrixA_[idx]; res != nullptr) {
            ttlA_[idx] = ttlA_[idx] - 1;
            if (ttlA_[idx] == 0) {
                gridMatrixA_[idx] = nullptr;
            }
            return res;
        }

        return nullptr;
    }

    [[nodiscard]] std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>> matrixB(size_t i, size_t j) {
        uint32_t idx = i*nTiles_ + j;
        if(auto res = gridMatrixB_[idx]; res != nullptr) {
            ttlB_[idx] = ttlB_[idx] - 1;
            if(ttlB_[idx] == 0) {
                gridMatrixB_[idx] = nullptr;
            }
            return res;
        }

        return nullptr;
    }

    void matrixA(std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>> tileA) {
        gridMatrixA_[tileA->rowIdx()*kTiles_ + tileA->colIdx()] = tileA;
    }

    void matrixB(std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>> tileB) {
        gridMatrixB_[tileB->rowIdx()*nTiles_ + tileB->colIdx()] = tileB;
    }

private:
    uint32_t mTiles_                                                                   = 0;
    uint32_t kTiles_                                                                   = 0;
    uint32_t nTiles_                                                                   = 0;
    std::vector<std::shared_ptr<CudaMatrixTile<MatrixType, InpIdA, Ord>>> gridMatrixA_ = {};
    std::vector<std::shared_ptr<CudaMatrixTile<MatrixType, InpIdB, Ord>>> gridMatrixB_ = {};
    std::vector<int32_t> ttlA_                                                         = {};
    std::vector<int32_t> ttlB_                                                         = {};
};

#endif //HH3_MATMUL_OUTER_PRODUCT_CUDA_STATE_H

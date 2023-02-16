#include "mmd.h"

template<class MatrixType>
void printMatrix(MatrixType *pData, uint32_t height, uint32_t width) {
    for(uint64_t i = 0; i < height; ++i) {
        for(uint64_t j = 0; j < width; ++j) {
            std::cout << std::setprecision(std::numeric_limits<MatrixType>::digits10 + 1)
               << pData[j*height + i] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    using MatrixType = double;
    constexpr auto Ord = Order::Col;

    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv);
    assert(getNumNodes() == 3);
    uint64_t M = 8, K = 6, N = 8, tileSize = 2;
    std::vector<int32_t> deviceIds = {getNodeId()};

    auto subMatA = std::make_shared<ContiguousSubMatrixContainer<Order::Col, MatrixType, 'a', Ord>>(1, M, K, tileSize, MPI_COMM_WORLD);
    auto subMatB = std::make_shared<ContiguousSubMatrixContainer<Order::Row, MatrixType, 'b', Ord>>(2, K, N, tileSize, MPI_COMM_WORLD);
    auto matrixC = std::make_shared<RedundantMatrixContainer<MatrixType, 'c', Ord>>(3, M, N, tileSize, MPI_COMM_WORLD);

    std::for_each(subMatA->data(), subMatA->data()+subMatA->dataSize(), [](MatrixType &el) { el = getNodeId()+1; });
    std::for_each(subMatB->data(), subMatB->data()+subMatB->dataSize(), [](MatrixType &el) { el = getNodeId()+1; });
    if(isRootNodeId()) {
        std::for_each(matrixC->data(), matrixC->data()+matrixC->dataSize(), [](MatrixType &el) { el = 1; });
    }
    else {
        std::for_each(matrixC->data(), matrixC->data()+matrixC->dataSize(), [](MatrixType &el) { el = 0; });
    }

    {
        MMD_VerifyCublas<MatrixType, 'a', 'b', 'c', Ord>().execute(subMatA, subMatB, matrixC, deviceIds);
    }

    if(isRootNodeId()) {
        std::vector<MatrixType> testA(M * K);
        for (uint64_t j = 0, width, nodeId = 0; j < K; ++nodeId) {
            width = K/getNumNodes();
            width += (nodeId < K%getNumNodes()? 1: 0);
            MatrixType val = nodeId+1;
            for(uint64_t k = 0; k < width; ++k, ++j) {
                for(uint64_t i = 0; i < M; ++i) {
                    testA[j*M + i] = val;
                }
            }
        }
#if not NDEBUG
        printMatrix(testA.data(), M, K);
#endif
        std::vector<MatrixType> testB(K * N);
        for(uint64_t i = 0, height, nodeId = 0; i < K; ++nodeId, i+=height) {
            height = K/getNumNodes();
            height += (nodeId < K%getNumNodes()? 1: 0);
            MatrixType val = nodeId+1;
            for(uint64_t j = 0; j < N; ++j) {
                for(uint64_t k = 0; k < height; ++k) {
                    testB[(j*K + i) + k] = val;
                }
            }
        }
#if not NDEBUG
        printMatrix(testB.data(), K, N);
#endif

        std::vector<MatrixType> testC(M * N, 1);

        for(uint64_t j = 0; j < N; ++j) {
            for(uint64_t i = 0; i < M; ++i) {
                for(uint64_t k = 0; k < K; ++k) {
                    testC[j*M + i] += testA[k*M + i]*testB[j*K + k];
                }
            }
        }
#if not NDEBUG
        printMatrix(testC.data(), M, N);

        if(isRootNodeId()) {
            std::cout << *matrixC;
        }
#endif
        for(uint64_t i = 0; i < testC.size(); ++i) {
            if(0.01 < std::abs(testC[i]-matrixC->data()[i])) {
                throw std::runtime_error("Test cublas failed!\n");
            }
        }
        printf("MMD_VerifyCublas verified!\n");
    }

    return 0;
}

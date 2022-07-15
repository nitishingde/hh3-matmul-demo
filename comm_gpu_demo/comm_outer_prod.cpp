#include <comm/comm.h>
#include <random>
#include <atomic>
#include <thread>

#include "data/matrix_data.h"
#include "mm.h"

#define VERIFY_MM 1

int main([[maybe_unused]]int32_t argc, [[maybe_unused]]char **argv) {
    using MatrixType = double;
    constexpr Order Ord = Order::Column;
    using namespace std::chrono_literals;

    CublasLockGuard cublasLockGuard;

    // initialize comm library
    comm::CommLockGuard commLockGuard(&argc, &argv);

    // A => m x k
    // B => k x n
    // C => m x n
    const size_t m = 4, k = 3, n = 4, blockSize = 2;

    std::vector<MatrixType> subA(m*k), subB(k*n), matC(m*n, 0);
    auto subMatA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(m, k, blockSize, *subA.data());
    auto subMatB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(k, n, blockSize, *subB.data());
    auto matrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(m, n, blockSize, *matC.data());

    // Mersenne Twister Random Generator
    uint64_t timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> (uint64_t) 32)};
    std::mt19937_64 rng(ss);
    // Choose your distribution depending on the type of MatrixType
    std::uniform_real_distribution<MatrixType> unif(0, 10);

    // initialize matrices
    std::for_each(subMatA->data(), subMatA->data() + (m * k), [&unif, &rng](MatrixType &val) { val = comm::getMpiNodeId()+1;(MatrixType) unif(rng); });
    std::for_each(subMatB->data(), subMatB->data() + (k * n), [&unif, &rng](MatrixType &val) { val = comm::getMpiNodeId()+1;(MatrixType) unif(rng); });
    if(comm::isMpiRootPid()) {
        std::for_each(matrixC->data(), matrixC->data() + (m * n), [&unif, &rng](MatrixType &val) { val = 1;(MatrixType) unif(rng); });
    }

    {
        MMCommOuterProduct<MatrixType, Ord> commOuterProduct;
        commOuterProduct.execute(subMatA, subMatB, matrixC);
    }

#if VERIFY_MM
    // initialize matrices for verification
    std::vector<MatrixType> matA, matB, matV;
    if(comm::isMpiRootPid()) {
        matA.resize(m*k*comm::getMpiNumNodes(), 0);
        matB.resize(k*comm::getMpiNumNodes()*n, 0);
        matV.resize(m*n, 1);
    }
    auto matrixA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(m, k*comm::getMpiNumNodes(), blockSize, *matA.data());
    auto matrixB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(k*comm::getMpiNumNodes(), n, blockSize, *matB.data());
    auto testMatrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(m, n, blockSize, *matV.data());

    if(comm::isMpiRootPid()) {
        std::copy_n(subMatA->data(), m*k, matrixA->data());
        for(size_t j = 0; j < subMatB->matrixWidth(); ++j) {
            for(size_t i = 0; i < subMatB->matrixHeight(); ++i) {
                matrixB->data()[j*matrixB->leadingDimension()+i] = subMatB->data()[j*subMatB->leadingDimension()+i];
            }
        }
    }

    // verify code
    if(comm::isMpiRootPid()) {
        std::atomic_uint32_t aCount = comm::getMpiNumNodes()-1, bCount = comm::getMpiNumNodes()-1;
        comm::connectReceiver([&matrixA, &matrixB, &aCount, &bCount](comm::SignalType signalType) {
            auto otherNode = signalType->otherNode;
            if(signalType->id == 1) {
                MatrixData<MatrixType, 'a', Ord> subMatA;
                subMatA.deserialize(std::istringstream(signalType->serializedData));
                std::copy_n(
                    subMatA.data(),
                    m*k,
                    matrixA->data()+otherNode*m*k
                );
                aCount--;
            }
            else if(signalType->id == 2) {
                MatrixData<MatrixType, 'b', Ord> subMatB;
                subMatB.deserialize(std::istringstream(signalType->serializedData));
                for(size_t j = 0; j < subMatB.matrixWidth(); ++j) {
                    for(size_t i = 0; i < subMatB.matrixHeight(); ++i) {
                        matrixB->data()[j*matrixB->leadingDimension()+otherNode*k+i] = subMatB.data()[j*subMatB.leadingDimension()+i];
                    }
                }
                bCount--;
            }
        });

        while (0 < aCount.load() or 0 < bCount.load()) {}

        MMVerification<MatrixType, Ord> mmVerification;
        mmVerification.execute(matrixA, matrixB, testMatrixC);
        std::cout << *matrixA;
        std::cout << *matrixB;
        std::cout << *matrixC;
        std::cout << *testMatrixC;
    }
    else {
        comm::sendMessage(1, subMatA->serialize(), 0);
        comm::sendMessage(2, subMatB->serialize(), 0);
    }
#endif

    return 0;
}

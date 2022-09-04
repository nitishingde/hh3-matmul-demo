#ifndef HH3_MATMUL_MMD_H
#define HH3_MATMUL_MMD_H

#include "utility.h"
#ifdef HH_USE_CUDA
#include <cublasXt.h>
#else
#include <openblas/cblas.h>
#endif
#include "data/contiguous_sub_matrix_container.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"

template<class MatrixType, char IdA, char IdB, char IdC, Order Ord>
class MMD_Strategy {
private:
    virtual void executeImpl(
            std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
            std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
            std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
            const std::vector<int32_t> &deviceIds,
            std::string dotFile
    ) = 0;
    [[nodiscard]] virtual std::string strategy() const = 0;
    [[nodiscard]] virtual std::string matTypeA() const = 0;
    [[nodiscard]] virtual std::string matTypeB() const = 0;
    [[nodiscard]] virtual std::string matTypeC() const = 0;
    [[nodiscard]] virtual std::string className() const = 0;

public:
    void execute(
            std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
            std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
            std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
            const std::vector<int32_t> &deviceIds
    ) {
#if not NDEBUG
        int32_t isMpiInitialized = false;
        if(auto status = MPI_Initialized(&isMpiInitialized); not isMpiInitialized) {
            std::runtime_error("[MPI] not initialized!\n");
        }
#endif

        auto start = std::chrono::high_resolution_clock::now();
        this->executeImpl(matrixA, matrixB, matrixC, deviceIds, className());
        auto end = std::chrono::high_resolution_clock::now();

        if(getNodeId() != 0) return;

        size_t M = matrixC->matrixHeight(), N = matrixC->matrixWidth(), K = matrixA->matrixWidth();
        double time = double(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1.e9;
        //https://forums.developer.nvidia.com/t/how-to-compute-gflops-for-gemm-blas/20218/6
        double gflops = (M*N*(2*K+2))/(1.e9*time);
        printf(
            "[ " GREEN("MMD") " ][ Strategy = " GREEN("%-12s") " ][ A = " GREEN("%-9s") " ][ B = " GREEN("%-9s") " ][ C = " GREEN("%-9s") " ][ " CYAN("%9.3f") " gflops ][ " RED("%8.3f") " secs ]\n",
            strategy().c_str(),
            matTypeA().c_str(),
            matTypeB().c_str(),
            matTypeC().c_str(),
            gflops,
            time
        );
        fflush(stdout);
    }
};

template<class MatrixType, char IdA, char IdB, char IdC, Order Ord>
class MMD_VerifyCublas: public MMD_Strategy<MatrixType, IdA, IdB, IdC, Ord> {
private:
    void executeImpl(
        std::shared_ptr<MatrixContainer<MatrixType, IdA, Ord>> matrixA,
        std::shared_ptr<MatrixContainer<MatrixType, IdB, Ord>> matrixB,
        std::shared_ptr<MatrixContainer<MatrixType, IdC, Ord>> matrixC,
        const std::vector<int32_t> &deviceIds,
        std::string dotFile
    ) override {

        auto subA = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Col, MatrixType, IdA, Ord>>(matrixA);
        auto subB = std::static_pointer_cast<ContiguousSubMatrixContainer<Order::Row, MatrixType, IdB, Ord>>(matrixB);
        auto matC = std::static_pointer_cast<RedundantMatrixContainer<MatrixType, IdC, Ord>>(matrixC);
        size_t m = matC->matrixHeight(), k = subA->subMatrixWidth(), n = matC->matrixWidth();
        size_t tileSize = matC->matrixTileSize();

        cublasXtHandle_t handle;
        checkCudaErrors(cublasXtCreate(&handle));

        checkCudaErrors(cublasXtDeviceSelect(handle, deviceIds.size(), (int*)deviceIds.data()));
        checkCudaErrors(cublasXtSetBlockDim(handle, tileSize));

        MatrixType alpha = 1.0, beta = 1.0;

        if constexpr (std::is_same<MatrixType, double>::value) {
            checkCudaErrors(cublasXtDgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, (double *) (&alpha),
                (double *) subA->data(), subA->leadingDimension(),
                (double *) subB->data(), subB->leadingDimension(),
                (double *) (&beta), (double *) matC->data(), matC->leadingDimension()
            ));
        } else {
            checkCudaErrors(cublasXtSgemm(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, (float *) (&alpha),
                (float *) subA->data(), subA->leadingDimension(),
                (float *) subB->data(), subB->leadingDimension(),
                (float *) (&beta), (float *) matC->data(), matC->leadingDimension()
            ));
        }

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cublasXtDestroy(handle));

        std::vector<MatrixType> tempC((isRootNodeId()? m*n: 0));
        if constexpr(std::is_same_v<MatrixType, double>) {
            MPI_Reduce(matC->data(), tempC.data(), m*n, MPI_DOUBLE, MPI_SUM, 0, matC->mpiComm());
        }
        else if constexpr(std::is_same_v<MatrixType, float>) {
            MPI_Reduce(matC->data(), tempC.data(), m*n, MPI_FLOAT, MPI_SUM, 0, matC->mpiComm());
        }
        else {
            throw std::runtime_error("Type not supported\n");
        }
        std::memcpy(matC->data(), tempC.data(), sizeof(MatrixType)*tempC.size());
    }

    [[nodiscard]] std::string strategy() const override {
        return "CublasXt";
    }

    [[nodiscard]] std::string matTypeA() const override {
        return "Sub";
    }

    [[nodiscard]] std::string matTypeB() const override {
        return "Sub";
    }

    [[nodiscard]] std::string matTypeC() const override {
        return "Redundant";
    }

    [[nodiscard]] std::string className() const override {
        return NAME(MMD_VerifyCublas);
    }
};

#endif //HH3_MATMUL_MMD_H

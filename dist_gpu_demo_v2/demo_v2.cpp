#include "mmd.h"
#include "matrix_utility.h"

#ifndef NDEBUG
template<typename MatrixType, char Id>
MatrixType* getMatrixToRoot(std::shared_ptr<MatrixContainer<MatrixType, Id>> matrix) {
    auto T          = matrix->tileDim();
    auto ld         = matrix->matrixHeight();
    MatrixType *mat = nullptr;
    if(isRootNodeId()) {
        cudaMallocManaged(&mat, matrix->matrixWidth()*matrix->matrixHeight()*sizeof(MatrixType));
    }

    auto copyTile = [](const MatrixType *src, const int64_t tw, const int64_t th, MatrixType *dst, const int64_t ld) {
        for(int64_t i = 0; i < tw; ++i) {
            for(int64_t j = 0; j < th; ++j) {
                dst[i*ld + j] = src[i*th + j];
            }
        }
    };

    auto tempTile = new MatrixType[T*T];
    for(int64_t row = 0; row < matrix->matrixNumRowTiles(); ++row) {
        for(int64_t col = 0; col < matrix->matrixNumRowTiles(); ++col) {
            auto tw = matrix->tileWidth(row, col), th = matrix->tileHeight(row, col);
            if(auto tile = matrix->tile(row, col); !isRootNodeId() and tile != nullptr) {
                checkMpiErrors(MPI_Send(tile->data(), tile->byteSize(), MPI_CHAR, 0, row*100+col, MPI_COMM_WORLD));
            }
            else if(isRootNodeId() and tile == nullptr) {
                MPI_Status mpiStatus;
                checkMpiErrors(MPI_Recv((void*)tempTile, tw*th*sizeof(MatrixType), MPI_CHAR, matrix->owner(row, col), row*100+col, MPI_COMM_WORLD, &mpiStatus));
                copyTile(tempTile, tw, th, &mat[col*ld*T + row*T], ld);
            }
            else if(isRootNodeId()) {
                copyTile((MatrixType*)tile->data(), tw, th, &mat[col*ld*T + row*T], ld);
            }
        }
    }

    return mat;
}

template<typename MatrixType, Major major = Major::COL>
void printMatrix(MatrixType *mat, int64_t height, int64_t width, const char *msgPrefix) {
    assert(major == Major::COL);
    printf("%s\n", msgPrefix);
    fflush(stdout);
    for(int64_t row = 0; row < height; ++row) {
        for(int64_t col = 0; col < width; ++col) {
            printf("%f ", mat[col*height + row]);
        }
        printf("\n");
        fflush(stdout);
    }
    printf("\n");
    fflush(stdout);
}
#endif

int main(int argc, char *argv[]) {
    auto [p, q, M, K, N, T, productThreads, accumulateThreads, windowSize, lookAhead, computeTiles, path, host, resultsFile] = parseArgs(argc, argv);
    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv, p, q);

    using MatrixType = float;

    constexpr char       IdA        = 'a';
    constexpr char       IdB        = 'b';
    constexpr char       IdC        = 'c';
    constexpr MemoryType memoryType = MemoryType::HOST;
    MPI_Comm             mpiComm    = MPI_COMM_WORLD;

    windowSize = genWindowSize<MatrixType>(M, N, T, computeTiles, windowSize);
    printf("[Node %ld][p %ld][q %ld][M %ld][K %ld][N %ld][T %ld][MT/p %ld][KT %ld][NT/q %ld][productThreads %ld][accumulateThreads %ld][windowSize %ld][lookAhead %ld][computeTiles %ld]\n", getNodeId(), p, q, M, K, N, T, (((M+T-1)/T)+p-1)/p, (K+T-1)/T, (((N+T-1)/T)+q-1)/q, productThreads, accumulateThreads, windowSize, lookAhead, computeTiles);
    fflush(stdout);

    int32_t gpuCount = 0;
    checkCudaErrors(cudaGetDeviceCount(&gpuCount));
    auto deviceIds = std::vector<int32_t>();
    deviceIds.reserve(gpuCount);
    for(int32_t i = 0; i < gpuCount; ++i) {
        deviceIds.emplace_back(i);
        cudaDeviceProp cudaDeviceProp{};
        cudaGetDeviceProperties(&cudaDeviceProp, i);
        printf("[Process %ld][GPU %d/%d][%s][RAM = %zuGB][#AyncEngines = %d][Compute capability = %d.%d]\n",
            getNodeId(),
            i, gpuCount,
            cudaDeviceProp.name,
            cudaDeviceProp.totalGlobalMem/(1<<30),
            cudaDeviceProp.asyncEngineCount,
            cudaDeviceProp.major, cudaDeviceProp.minor
        );
    }
    CublasGlobalLockGuard cublasGlobalLockGuard(deviceIds);

    auto matrixA = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdA>>(memoryType, M, K, T, p, q, mpiComm);
    auto matrixB = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdB>>(memoryType, K, N, T, p, q, mpiComm);
    auto matrixC = std::make_shared<TwoDBlockCyclicContiguousSubMatrix<MatrixType, IdC>>(memoryType, M, N, T, p, q, mpiComm);

    if(isRootNodeId()) {
        printDataDistribution<MatrixType, IdA, IdB, IdC>(matrixA, matrixB, matrixC);
        std::filesystem::remove_all(path);
        std::filesystem::create_directory(path);
    }

#ifndef NDEBUG
    auto C = getMatrixToRoot<MatrixType, IdC>(matrixC);
#endif

    auto strategy = MMD_WindowStrategy<MatrixType, IdA, IdB, IdC>();
    auto time = strategy.builder(accumulateThreads, computeTiles, lookAhead, productThreads, windowSize).executeImpl(matrixA, matrixB, matrixC, deviceIds, mpiComm, path + "window_" + std::to_string(getNodeId()) + ".dot");
    if(isRootNodeId()) {
        printf("[ Perf " GREEN("%9.3f") " gflops ][ Time " BLUE("%8.3f") " secs]\n",
            (double(M) * double(K) * double(N) * double(2)) / (1.e9 * time),
            time
        );
        fflush(stdout);
    }

#ifndef NDEBUG
    // verify solution
    MPI_Barrier(mpiComm);
    auto A    = getMatrixToRoot<MatrixType, IdA>(matrixA);
    auto B    = getMatrixToRoot<MatrixType, IdB>(matrixB);
    auto calc = getMatrixToRoot<MatrixType, IdC>(matrixC);

    if(isRootNodeId()) {
        cublasHandle_t handle;
        checkCudaErrors(cublasCreate_v2(&handle));
        MatrixType alpha = 1., beta = 1.;
        checkCudaErrors(cudaMemPrefetchAsync(A, M*K*sizeof(MatrixType), 0));
        checkCudaErrors(cudaMemPrefetchAsync(B, K*N*sizeof(MatrixType), 0));
        checkCudaErrors(cudaMemPrefetchAsync(C, M*N*sizeof(MatrixType), 0));

        printMatrix(A, M, K, "MatrixA");
        printMatrix(B, K, N, "MatrixB");
        printMatrix(calc, M, N, "MatrixC using HH");

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M));
        checkCudaErrors(cudaDeviceSynchronize());
        printMatrix(C, M, N, "MatrixC using cublas");

        int64_t count = 0;
        for(int64_t col = 0; col < N; ++col) {
            for(int64_t row = 0; row < M; ++row) {
                if(0.001 < std::abs(C[col*M + row]-calc[col*M + row])) {
                    count++;
                    fprintf(stderr, "[count = %6ld][row %ld][col %ld] %f <--> %f\n", count, row, col, C[col*M + row], calc[col*M + row]);
                }
            }
        }
        if(count == 0) {
            printf("Verified!\n");
        }
    }
#endif

    return 0;
}

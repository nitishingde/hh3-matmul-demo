#include <print>
#include "common_matrix_utility.h"
#include "utility.h"
#include "tasks.h"
#include "mmd.h"

template<typename MatrixType, char Id>
std::shared_ptr<MatrixType[]> getMatrixToRoot(const std::shared_ptr<MatrixContainer<MatrixType, Id>> &matrix) {
    auto                          T   = matrix->tileDim();
    auto                          ld  = matrix->matrixHeight();
    std::shared_ptr<MatrixType[]> mat = nullptr;
    if(isRootNodeId()) {
        mat = std::make_shared<MatrixType[]>(matrix->matrixWidth()*matrix->matrixHeight());
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
        for(int64_t col = 0; col < matrix->matrixNumColTiles(); ++col) {
            auto tw = matrix->tileWidth(row, col), th = matrix->tileHeight(row, col);
            if(auto tile = matrix->tile(row, col); !isRootNodeId() and tile != nullptr) {
                const auto tagId = row*matrix->matrixNumColTiles() + col;
                checkMpiErrors(MPI_Send(tile->data(), tile->byteSize(), MPI_CHAR, 0, tagId, MPI_COMM_WORLD));
            }
            else if(isRootNodeId() and tile == nullptr) {
                MPI_Status mpiStatus;
                const auto tagId = row*matrix->matrixNumColTiles() + col;
                checkMpiErrors(MPI_Recv(static_cast<void*>(tempTile), tw*th*sizeof(MatrixType), MPI_CHAR, matrix->owner(row, col), tagId, MPI_COMM_WORLD, &mpiStatus));
                copyTile(tempTile, tw, th, &mat[col*ld*T + row*T], ld);
            }
            else if(isRootNodeId()) {
                copyTile(static_cast<MatrixType*>(tile->data()), tw, th, &mat[col*ld*T + row*T], ld);
            }
        }
    }

    return mat;
}

template<Major major = Major::COL>
void printMatrix(const auto &mat, const int64_t height, const int64_t width, const char *msgPrefix) {
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

int main(int argc, char *argv[]) {
    auto [p, q, M, K, N, T, l, _gp, _gq, _wh, _ww, _d, productThreads, verbose, path, resultsFile] = parseArgs(argc, argv);
    auto mpiLg = MpiGlobalLockGuard(&argc, &argv, p, q);

    using MatrixType = float;

    constexpr char IdA        = 'A';
    constexpr char IdB        = 'B';
    constexpr char IdC        = 'C';
    constexpr auto memoryType = MemoryType::HOST;
    constexpr auto mpiComm    = MPI_COMM_WORLD;

    auto           mpiCommA   = MPI_COMM_WORLD;
    auto           mpiCommB   = MPI_COMM_WORLD;
    auto           mpiCommC   = MPI_COMM_WORLD;
    checkMpiErrors(MPI_Comm_dup(mpiComm, &mpiCommB));

    const auto matrixA = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdA>>(memoryType, M, K, T, p, q, mpiCommA);
    const auto matrixB = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdB>>(memoryType, K, N, T, p, q, mpiCommB);
    const auto matrixC = std::make_shared<TwoDBlockCyclicMatrix<MatrixType, IdC>>(memoryType, M, N, T, p, q, mpiCommC);
    auto             C = getMatrixToRoot<MatrixType, IdC>(matrixC);

    auto mmd  = MMD_Simd1<MatrixType, IdA, IdB, IdC>();
    if(isRootNodeId()) {
        printDataDistribution<MatrixType, IdA, IdB, IdC>(matrixA, matrixB, matrixC);
        std::filesystem::remove_all(path);
        std::filesystem::create_directory(path);
    }

    const auto time = mmd.executeImpl(matrixA, matrixB, matrixC, mpiComm, path + "tp" + std::to_string(getNodeId()) + ".dot");
    if(isRootNodeId()) std::printf("[Time %fs]\n", time);

    const auto A    = getMatrixToRoot<MatrixType, IdA>(matrixA);
    const auto B    = getMatrixToRoot<MatrixType, IdB>(matrixB);
    const auto calc = getMatrixToRoot<MatrixType, IdC>(matrixC);

    if(isRootNodeId()) {
        if(2 <= verbose) printMatrix(A, M, K, "MatrixA");
        if(2 <= verbose) printMatrix(B, K, N, "MatrixB");
        if(2 <= verbose) printMatrix(calc, M, N, "MatrixC using HH");

        if constexpr(std::is_same_v<MatrixType, float>) {
            constexpr MatrixType alpha = 1;
            constexpr MatrixType beta  = 1;
            cblas_sgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                alpha,
                static_cast<float*>(A.get()), static_cast<int>(M),
                static_cast<float*>(B.get()), static_cast<int>(K),
                beta,
                static_cast<float*>(C.get()), static_cast<int>(M)
            );
        }
        // else if constexpr(std::is_same_v<MatrixType, double>) {
        //     cblas_dgemm(
        //         CblasColMajor, CblasNoTrans, CblasNoTrans,
        //         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        //         alpha,
        //         reinterpret_cast<double*>(A.get()), static_cast<int>(M),
        //         reinterpret_cast<double*>(B.get()), static_cast<int>(K),
        //         beta,
        //         reinterpret_cast<double*>(C.get()), static_cast<int>(M)
        //     );
        // }

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
            printf(GREEN("Verified!\n"));
        }
    }

    checkMpiErrors(MPI_Comm_free(&mpiCommB));

    return 0;
}

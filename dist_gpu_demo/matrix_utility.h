#ifndef HH3_MATMUL_MATRIX_UTILITY_H
#define HH3_MATMUL_MATRIX_UTILITY_H

#include "data/contiguous_sub_matrix_container.h"
#include "data/cyclic2d_matrix_container.h"
#include "data/redundant_matrix_container.h"
#include "mmd.h"

template<Order SubMatrixOrder, class MatrixType, char Id, Order Ord>
void init(std::shared_ptr<ContiguousSubMatrixContainer<SubMatrixOrder, MatrixType, Id, Ord>> subMat) {
    std::for_each(
        subMat->data(),
        subMat->data() + subMat->subMatrixHeight()*subMat->subMatrixWidth(),
        [](MatrixType &val) { val = fastrand()%10; }
    );
}

template<class MatrixType, char Id, Order Ord>
void init(std::shared_ptr<Cyclic2dMatrixContainer<MatrixType, Id, Ord>> cyclic2dMatrix) {
    for(uint32_t i = 0; i < cyclic2dMatrix->matrixNumColTiles(); ++i) {
        for(uint32_t j = 0; j < cyclic2dMatrix->matrixNumRowTiles(); ++j) {
            if(auto tile = cyclic2dMatrix->getTile(i, j); tile != nullptr) {
                if(tile->sourceNodeId() == getNodeId()) {
                    std::for_each(tile->data(), tile->data()+tile->dataSize(), [](MatrixType &val) { val = getNodeId(); });
                }
                else {
                    std::for_each(tile->data(), tile->data()+tile->dataSize(), [](MatrixType &val) { val = 0; });
                }
            }
        }
    }
}

template<class MatrixType, char Id, Order Ord>
void init(std::shared_ptr<RedundantMatrixContainer<MatrixType, Id, Ord>> redundantMatrix) {
    if(isRootNodeId()) {
        for(uint32_t idx = 0; idx < redundantMatrix->matrixNumRowTiles()*redundantMatrix->matrixNumColTiles(); ++idx) {
            uint32_t rowIdx = idx/redundantMatrix->matrixNumColTiles(), colIdx = idx%redundantMatrix->matrixNumColTiles();
            if(auto tile = redundantMatrix->getTile(rowIdx, colIdx); tile != nullptr) {
                if constexpr(Ord == Order::Col) {
                    uint32_t value = idx%getNumNodes();
                    for(uint32_t jj = 0; jj < tile->width(); ++jj) {
                        std::for_each(
                            &tile->data()[jj*tile->leadingDimension()],
                            &tile->data()[jj*tile->leadingDimension()+tile->height()],
                            [value](MatrixType &val) { val = value; }
                        );
                    }
                }
                else {
                    throw;
                }
            }
        }
    }
    else {
        std::memset(redundantMatrix->data(), 0, redundantMatrix->dataSize()*sizeof(MatrixType));
    }
}

template<class MatrixType, char IdA, char IdB, char IdC, Order Ord>
bool verifySolution(
        std::shared_ptr<ContiguousSubMatrixContainer<Order::Col, MatrixType, IdA, Ord>> &&subMatA,
        std::shared_ptr<ContiguousSubMatrixContainer<Order::Row, MatrixType, IdB, Ord>> &&subMatB,
        std::shared_ptr<Cyclic2dMatrixContainer<MatrixType, IdC, Ord>> matrixC,
        const std::vector<int32_t> &deviceIds,
        MPI_Comm matrixComm
    ) {
    uint64_t M = matrixC->matrixHeight(), N = matrixC->matrixWidth(), tileSize = matrixC->matrixTileSize();
    auto redundantMatrixC = std::make_shared<RedundantMatrixContainer<MatrixType, IdC, Ord>>(3, M, N, tileSize, matrixComm, isRootNodeId());
    init(redundantMatrixC);

    {
        MPI_Barrier(MPI_COMM_WORLD);
        MMD_VerifyCublas<MatrixType, IdA, IdB, IdC, Ord>().execute(std::move(subMatA), std::move(subMatB), redundantMatrixC, deviceIds);
        subMatA = nullptr;
        subMatB = nullptr;
    }

    MPI_Datatype datatype;
    MPI_Type_contiguous(int32_t(M), std::is_same_v<MatrixType, double>? MPI_DOUBLE: MPI_FLOAT, &datatype);
    MPI_Type_commit(&datatype);
    MPI_Bcast(redundantMatrixC->data(), int32_t(N), datatype, 0, redundantMatrixC->mpiComm());
    MPI_Barrier(MPI_COMM_WORLD);
    if(isRootNodeId()) printf("Verifying solution.\n");

    bool isCorrect = true;
    for(uint64_t i = 0; i < redundantMatrixC->matrixNumRowTiles(); ++i) {
        for(uint64_t j = 0; j < redundantMatrixC->matrixNumColTiles(); ++j) {
            if(auto tile = matrixC->getTile(i, j); tile == nullptr or tile->sourceNodeId() != getNodeId()) continue;
            else if(*tile != *redundantMatrixC->getTile(i, j)) {
                isCorrect = false;
                std::cerr << "[Error] tile @[" + std::to_string(i) + ", " + std::to_string(j) + "] don't match.\n";
#if not NDEBUG
                std::cout << GREEN("Actual  : ") << *redundantMatrixC->getTile(i, j);
                std::cout <<   RED("Computed: ") << *tile;
#endif
            }
        }
    }

    return isCorrect;
}

template<class MatrixType, char Id, Order Ord>
void dumpData(std::shared_ptr<Cyclic2dMatrixContainer<MatrixType, Id, Ord>> matrixC, const std::string &path) {
    for(uint64_t i = 0; i < matrixC->matrixNumColTiles(); ++i) {
        for(uint64_t j = 0; j < matrixC->matrixNumRowTiles(); ++j) {
            if(auto tile = matrixC->getTile(i, j); tile != nullptr and tile->sourceNodeId() == getNodeId()) {
                std::string fileName = path+"/matrix_tile_"+std::to_string(i)+"_"+std::to_string(j)+".dat";
                printf("[Process %d] Writing to file %s\n", getNodeId(), fileName.c_str());
                auto pFile = fopen(fileName.c_str(), "w");
                fwrite(tile->data(), sizeof(MatrixType), tile->dataSize(), pFile);
                fclose(pFile);
            }
        }
    }
}

#endif //HH3_MATMUL_MATRIX_UTILITY_H

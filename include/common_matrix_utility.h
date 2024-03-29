#ifndef HH3_MATMUL_COMMON_MATRIX_UTILITY_H
#define HH3_MATMUL_COMMON_MATRIX_UTILITY_H

#include "common_data.h"

template<class MatrixType, char IdA, char IdB, char IdC,
    class MatrixA = MatrixContainer<MatrixType, IdA>,
    class MatrixB = MatrixContainer<MatrixType, IdA>,
    class MatrixC = MatrixContainer<MatrixType, IdA>
>
void printDataDistribution(std::shared_ptr<MatrixA> matrixA, std::shared_ptr<MatrixB> matrixB, std::shared_ptr<MatrixC> matrixC) {
    printf("Data distribution:\n");
    fflush(stdout);
    for(int64_t row = 0; row < matrixB->matrixNumRowTiles(); ++row) {
        for(int64_t col = 0; col < matrixA->matrixNumColTiles(); ++col) {
            printf("   ");
        }
        printf("  ");
        for(int64_t col = 0; col < matrixB->matrixNumColTiles(); ++col) {
            printf("%2ld ", matrixB->owner(row, col));
        }
        printf("\n");
        fflush(stdout);
    }
    printf("\n");
    fflush(stdout);

    for(int64_t row = 0; row < matrixA->matrixNumRowTiles(); ++row) {
        for(int64_t col = 0; col < matrixA->matrixNumColTiles(); ++col) {
            printf("%2ld ", matrixA->owner(row, col));
        }
        printf("  ");
        for(int64_t col = 0; col < matrixC->matrixNumColTiles(); ++col) {
            printf("%2ld ", matrixC->owner(row, col));
        }
        printf("\n");
        fflush(stdout);
    }
    printf("\n");
    fflush(stdout);
}

template<class MatrixType, char Id, class Matrix = MatrixContainer<MatrixType, Id>>
void printDataDistribution(std::shared_ptr<Matrix> matrix) {
    printf("Data distribution for matrix-%c:\n", Id);
    fflush(stdout);
    for(int64_t row = 0; row < matrix->matrixNumRowTiles(); ++row) {
        for(int64_t col = 0; col < matrix->matrixNumColTiles(); ++col) {
            printf("%2ld ", matrix->owner(row, col));
        }
        printf("\n");
        fflush(stdout);
    }
    printf("\n");
    fflush(stdout);
}

#endif //HH3_MATMUL_COMMON_MATRIX_UTILITY_H

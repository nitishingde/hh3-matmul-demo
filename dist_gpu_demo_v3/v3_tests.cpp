#include <hedgehog/hedgehog.h>
#include "utility.h"

int main(int argc, char *argv[]) {
    auto [p, q, M, K, N, T, l, gp, gq, wh, ww, d, productThreads, verbose, path, resultsFile] = parseArgs(argc, argv);
    MpiGlobalLockGuard mpiGlobalLockGuard(&argc, &argv, p, q, MPI_THREAD_SERIALIZED);
    using MatrixType = float;

    auto [wh1, ww1] = getWindowSize<MatrixType>(M, N, T, gp, gq, d);

    int64_t MT = (M+T-1)/T, NT = (N+T-1)/T;
    if(isRootNodeId()) printf("[MT/p %ld][NT/q %ld][wh ww %ld %ld]\n", (MT+p-1)/p, (NT+q-1)/q, wh1, ww1);

    return 0;
}
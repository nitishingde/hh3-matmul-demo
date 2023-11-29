#ifndef HH3_MATMUL_UTILITY_H
#define HH3_MATMUL_UTILITY_H

#include <tclap/CmdLine.h>
#include "common_utility.h"

// Argument parser -------------------------------------------------------------------------------------------------- //

auto parseArgs(int argc, char **argv) {
    int64_t prodThreads = 4;
#if HH_USE_CUDA
    cudaDeviceProp cudaDeviceProp{};
    cudaGetDeviceProperties(&cudaDeviceProp, 0);
    prodThreads = cudaDeviceProp.asyncEngineCount;
#endif

    try {
        TCLAP::CmdLine cmd("Command description message", ' ', "1.0");

        TCLAP::ValueArg<int64_t> argP("p", "p-dim", "height of grid of nodes", true, 1, "non negative integer value");
        cmd.add(argP);
        TCLAP::ValueArg<int64_t> argQ("q", "q-dim", "width of grid of nodes", true, 1, "non negative integer value");
        cmd.add(argQ);
        TCLAP::ValueArg<int64_t> argM("M", "m-dim", "height of matrix A / height of matrix C", false, 32768, "non negative integer value");
        cmd.add(argM);
        TCLAP::ValueArg<int64_t> argK("K", "k-dim", "width of matrix A / height of matrix B", false, 32768, "non negative integer value");
        cmd.add(argK);
        TCLAP::ValueArg<int64_t> argN("N", "n-dim", "width of matrix B / width of matrix C", false, 32768, "non negative integer value");
        cmd.add(argN);
        TCLAP::ValueArg<int64_t> argT("T", "tileSize", "tile size", false, 8192, "non negative integer value");
        cmd.add(argT);
        TCLAP::ValueArg<int64_t> argProductThreads("t", "prod", "product threads", false, prodThreads, "non negative integer value");
        cmd.add(argProductThreads);
        TCLAP::ValueArg<int64_t> argAccumulateThreads("a", "acc", "accumulate threads", false, 4, "non negative integer value");
        cmd.add(argAccumulateThreads);
        TCLAP::ValueArg<int64_t> argWindowSize("W", "windowSize", "window size", false, -1, "non negative integer value");
        cmd.add(argWindowSize);
        TCLAP::ValueArg<int64_t> argLookAhead("l", "lookAhead", "look ahead factor", false, 2, "non negative integer value");
        cmd.add(argLookAhead);
        TCLAP::ValueArg<int64_t> argComputeTiles("c", "computeTiles", "computation tiles", false, 32, "non negative integer value");
        cmd.add(argComputeTiles);
        TCLAP::ValueArg<std::string> argPath("P", "path", "scratch/tmp dir path", false, "./dots/", "dir path");
        cmd.add(argPath);
        TCLAP::ValueArg<std::string> argHostFile("H", "hostfile", "path to hostfile", false, "", "file path");
        cmd.add(argHostFile);
        TCLAP::ValueArg<std::string> results("R", "results", "store results in csv format", false, "./results.csv", "file path");
        cmd.add(results);

        cmd.parse(argc, argv);
        return std::make_tuple(
            argP.getValue(),
            argQ.getValue(),
            argM.getValue(),
            argK.getValue(),
            argN.getValue(),
            argT.getValue(),
            argProductThreads.getValue(),
            argAccumulateThreads.getValue(),
            argWindowSize.getValue(),
            argLookAhead.getValue(),
            argComputeTiles.getValue(),
            argPath.getValue(),
            argHostFile.getValue(),
            results.getValue()
        );
    }
    catch (TCLAP::ArgException &e) {
        fprintf(stderr, "[Error] %s for arg %s\n", e.error().c_str(), e.argId().c_str());
        return std::make_tuple(
            int64_t(1),
            int64_t(1),
            int64_t(32768),
            int64_t(32768),
            int64_t(32768),
            int64_t(32768),
            int64_t(prodThreads),
            int64_t(4),
            int64_t(-1),
            int64_t(2),
            int64_t(32),
            std::string("./"),
            std::string(""),
            std::string("./results.csv")
        );
    }
}

// MM --------------------------------------------------------------------------------------------------------------- //

template<typename MatrixType>
int64_t genWindowSize(const int64_t M, const int64_t N, const int64_t tileSize, const int64_t prodTilesPerDev, const int64_t suggestedWindowSize = -1) {
    int64_t tilesInCol            = (M+tileSize-1)/tileSize;
    int64_t tilesInRow            = (N+tileSize-1)/tileSize;
    int64_t tilesInColPerNode     = (tilesInCol+sGridP-1)/sGridP;
    int64_t tilesInRowPerNode     = (tilesInRow+sGridQ-1)/sGridQ;
    int64_t maxTilesPerDimPerNode = std::max(tilesInColPerNode, tilesInRowPerNode);

    cudaDeviceProp cudaDeviceProp = {};
    checkCudaErrors(cudaGetDeviceProperties(&cudaDeviceProp, 0));
    int64_t tilesPerDev   = int64_t(cudaDeviceProp.totalGlobalMem)/int64_t(tileSize*tileSize*sizeof(MatrixType));
    int64_t maxWindowSize = (tilesPerDev-prodTilesPerDev)/2;

    // wh + ww + prodTilesPerDev = tilesPerDev, let wh = ww
    if(suggestedWindowSize <= 0) {
        return std::min(maxTilesPerDimPerNode, maxWindowSize);
    }

    return std::min(suggestedWindowSize, maxWindowSize);
}

#endif //HH3_MATMUL_UTILITY_H

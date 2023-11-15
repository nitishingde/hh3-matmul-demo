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
        TCLAP::ValueArg<int64_t> argT("T", "tileSize", "tile size", false, 1024, "non negative integer value");
        cmd.add(argT);
        TCLAP::ValueArg<int64_t> argL("l", "lookAhead", "look ahead", false, 2, "non negative integer value");
        cmd.add(argL);

        TCLAP::ValueArg<int64_t> argGp("", "gp", "height of grid of gpus", true, -1, "non negative integer value");
        cmd.add(argGp);
        TCLAP::ValueArg<int64_t> argGq("", "gq", "width of grid of gpus", true, -1, "non negative integer value");
        cmd.add(argGq);
        TCLAP::ValueArg<int64_t> argD("d", "depth", "depth", false, 2, "non negative integer value");
        cmd.add(argD);
        TCLAP::ValueArg<int64_t> argPt("t", "prod", "product threads", false, prodThreads, "non negative integer value");
        cmd.add(argPt);

        TCLAP::ValueArg<std::string> argPath("P", "path", "scratch/tmp dir path", false, "./dots/", "dir path");
        cmd.add(argPath);
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
            argL.getValue(),
            argGp.getValue(),
            argGq.getValue(),
            argD.getValue(),
            argPt.getValue(),
            argPath.getValue(),
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
            int64_t(1024),
            int64_t(2),
            int64_t(2),
            int64_t(2),
            int64_t(2),
            int64_t(4),
            std::string("./"),
            std::string("./results.csv")
        );
    }
}

// MM --------------------------------------------------------------------------------------------------------------- //

template<typename MatrixType>
auto getWindowSize(const int64_t M, const int64_t N, const int64_t tileSize, const int64_t gp, const int64_t gq, const int64_t depth) {
    cudaDeviceProp cudaDeviceProp = {};
    checkCudaErrors(cudaGetDeviceProperties(&cudaDeviceProp, 0));
    int64_t tilesPerDev   = int64_t(cudaDeviceProp.totalGlobalMem)/int64_t(tileSize*tileSize*sizeof(MatrixType));

    int64_t maxRows = (M+tileSize-1)/tileSize;
    int64_t maxCols = (N+tileSize-1)/tileSize;

    int64_t maxRowsPerNode = (maxRows+sGridP-1)/sGridP;
    int64_t maxColsPerNode = (maxCols+sGridQ-1)/sGridQ;

    int32_t gpuCount = -1;
    checkCudaErrors(cudaGetDeviceCount(&gpuCount));
    assert(gp*gq == gpuCount);

    int64_t maxRowsPerDev  = (maxRowsPerNode+gp-1)/gp;
    int64_t maxColsPerDev  = (maxColsPerNode+gq-1)/gq;
    printf("[Node %ld][GPU -> %ld x %ld][tilesPerDev %ld][maxRowsPerDev %ld][maxColsPerDev %ld]\n", getNodeId(), gp, gq, tilesPerDev, maxRowsPerDev, maxColsPerDev);

    // FIXME: search isn't exhaustive
    // for example
    // fh = 1, fw = 1 Nope
    // fh = 2, fh = 1 Nope
    // fh = 2, fh = 2 Yes
    // but i didn't look for fh = 1, fw = 2
    for(int64_t windowHeight = maxRowsPerDev, windowWidth = maxColsPerDev, fh = 1, fw = 1;;) {
        printf("[possible windowHeight %ld][possible windowWidth %ld][fh %ld][fw %ld][calc required tiles %ld/%ld]\n", windowHeight, windowWidth, fh, fw, windowHeight*windowWidth + depth*(windowHeight+windowWidth), tilesPerDev);
        if(windowHeight*windowWidth + depth*(windowHeight+windowWidth) < tilesPerDev) {
            return std::make_tuple(windowHeight, windowWidth);
        }

        if(windowWidth < windowHeight or (windowWidth == windowHeight and fw < fh)) {
            fh++;
            windowHeight = (maxRowsPerDev+fh-1)/fh;
        }
        else {
            fw++;
            windowWidth = (maxColsPerDev+fw-1)/fw;
        }
    }
}

#endif //HH3_MATMUL_UTILITY_H

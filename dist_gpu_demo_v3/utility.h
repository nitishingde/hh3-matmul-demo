#ifndef HH3_MATMUL_UTILITY_H
#define HH3_MATMUL_UTILITY_H

#include <tclap/CmdLine.h>
#include "common_utility.h"

// Argument parser -------------------------------------------------------------------------------------------------- //

static struct Args {
    int64_t p     = 1;               // MPI grid dimension
    int64_t q     = 1;               // MPI grid dimension
    int64_t M     = 32768;           // matrix dimension
    int64_t K     = 32768;           // matrix dimension
    int64_t N     = 32768;           // matrix dimension
    int64_t T     = 8192;            // tile dimension
    int64_t l     = 2;               // lookahead parameter
    int64_t gp    = 2;               // GPU grid dimension
    int64_t gq    = 2;               // GPU grid dimension
    int64_t wh    = 2;               // suggested window height
    int64_t ww    = 2;               // suggested window width
    int64_t d     = 2;               // depth parameter
    int64_t t     = 4;               // product thread count
    int64_t v     = 0;               // verbosity level
    std::string P = "./dots/";       // path to scratch/tmp dir path
    std::string R = "./results.csv"; // path to csv file to store the results
} args;

auto parseArgs(int argc, char **argv) {
    int64_t prodThreads = 4;
#if HH_USE_CUDA
    cudaDeviceProp cudaDeviceProp{};
    cudaGetDeviceProperties(&cudaDeviceProp, 0);
    prodThreads = cudaDeviceProp.asyncEngineCount;
#endif
    TCLAP::CmdLine cmd("Command description message", ' ', "1.0");

    TCLAP::ValueArg<int64_t> argP("p", "p-dim", "height of grid of nodes", true, args.p, "non negative integer value");
    cmd.add(argP);
    TCLAP::ValueArg<int64_t> argQ("q", "q-dim", "width of grid of nodes", true, args.q, "non negative integer value");
    cmd.add(argQ);

    TCLAP::ValueArg<int64_t> argM("M", "m-dim", "height of matrix A / height of matrix C", false, args.M, "non negative integer value");
    cmd.add(argM);
    TCLAP::ValueArg<int64_t> argK("K", "k-dim", "width of matrix A / height of matrix B", false, args.K, "non negative integer value");
    cmd.add(argK);
    TCLAP::ValueArg<int64_t> argN("N", "n-dim", "width of matrix B / width of matrix C", false, args.N, "non negative integer value");
    cmd.add(argN);
    TCLAP::ValueArg<int64_t> argT("T", "tileSize", "tile size", false, args.T, "non negative integer value");
    cmd.add(argT);

    TCLAP::ValueArg<int64_t> argL("l", "lookAhead", "look ahead", false, args.l, "non negative integer value");
    cmd.add(argL);
    TCLAP::ValueArg<int64_t> argGp("", "gp", "height of grid of gpus", true, -1, "non negative integer value");
    cmd.add(argGp);
    TCLAP::ValueArg<int64_t> argGq("", "gq", "width of grid of gpus", true, -1, "non negative integer value");
    cmd.add(argGq);
    TCLAP::ValueArg<int64_t> argWh("", "wh", "height of grid of gpus", false, args.wh, "non negative integer value");
    cmd.add(argWh);
    TCLAP::ValueArg<int64_t> argWw("", "ww", "width of grid of gpus", false, args.ww, "non negative integer value");
    cmd.add(argWw);
    TCLAP::ValueArg<int64_t> argD("d", "depth", "depth", false, args.d, "non negative integer value");
    cmd.add(argD);
    TCLAP::ValueArg<int64_t> argPt("t", "prod", "product threads", false, prodThreads, "non negative integer value");
    cmd.add(argPt);

    TCLAP::ValueArg<int64_t> argV("v", "verbose", "verbose level (0/1/2)", false, args.v, "non negative integer value");
    cmd.add(argV);

    TCLAP::ValueArg<std::string> argPath("P", "path", "scratch/tmp dir path", false, args.P, "dir path");
    cmd.add(argPath);
    TCLAP::ValueArg<std::string> argR("R", "results", "store results in csv format", false, args.R, "file path");
    cmd.add(argR);

    cmd.parse(argc, argv);
    args.p = argP.getValue();
    args.q = argQ.getValue();
    args.M = argM.getValue();
    args.K = argK.getValue();
    args.N = argN.getValue();
    args.T = argT.getValue();
    args.l = argL.getValue();
    args.gp = argGp.getValue();
    args.gq = argGq.getValue();
    args.wh = argWh.getValue();
    args.ww = argWw.getValue();
    args.d = argD.getValue();
    args.t = argPt.getValue();
    args.v = argV.getValue();
    args.P = argPath.getValue();
    args.R = argR.getValue();

    return std::make_tuple(
        args.p,
        args.q,
        args.M,
        args.K,
        args.N,
        args.T,
        args.l,
        args.gp,
        args.gq,
        args.wh,
        args.ww,
        args.d,
        args.t,
        args.v,
        args.P,
        args.R
    );
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

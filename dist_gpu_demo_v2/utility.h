#ifndef HH3_MATMUL_UTILITY_H
#define HH3_MATMUL_UTILITY_H

#include <mpi.h>
#include <string>
#include <tclap/CmdLine.h>

// LOG -------------------------------------------------------------------------------------------------------------- //

#define NAME(x) #x

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_CLEAR_LINE    "\33[2K"

#define RED(x) ANSI_COLOR_RED x ANSI_COLOR_RESET
#define GREEN(x) ANSI_COLOR_GREEN x ANSI_COLOR_RESET
#define YELLOW(x) ANSI_COLOR_YELLOW x ANSI_COLOR_RESET
#define BLUE(x) ANSI_COLOR_BLUE x ANSI_COLOR_RESET
#define MAGENTA(x) ANSI_COLOR_MAGENTA x ANSI_COLOR_RESET
#define CYAN(x) ANSI_COLOR_CYAN x ANSI_COLOR_RESET

#ifndef NDEBUG
#define LOG() printf("[Node %ld][%s:%d]\n", getNodeId(), __FILE__, __LINE__)
#else
#define LOG()
#endif

// MPI Related ------------------------------------------------------------------------------------------------------ //

static int32_t sMpiNodeId    = -1;
static int32_t sMpiNumNodes  = -1;
static int64_t sGridP        = -1;
static int64_t sGridQ        = -1;
static std::string sHostName = {};

[[nodiscard]] int64_t getNodeId() {
    return sMpiNodeId;
}

[[nodiscard]] int64_t getNumNodes() {
    return sMpiNumNodes;
}

[[nodiscard]] std::tuple<int64_t, int64_t> getGridDim() {
    return {sGridP, sGridQ};
}

[[nodiscard]] bool isRootNodeId() {
    return sMpiNodeId == 0;
}

[[nodiscard]] std::string getHostName() {
    return sHostName;
}

#ifndef checkMpiErrors
void __checkMpiErrors(const int errorCode, const char *file, const int line) {
    if(errorCode == MPI_SUCCESS) return;
    char msg[MPI_MAX_ERROR_STRING];
    int length;
    MPI_Error_string(errorCode, msg, &length);
    fprintf(stderr, "[Process %d] %s:%d {%s}\n", sMpiNodeId, file, line, msg);
    MPI_Abort(MPI_COMM_WORLD, errorCode);
}

#if MMD_ENABLE_CHECK_MPI
//#define checkMpiErrors(err) if(int errorCode = err; errorCode != MPI_SUCCESS) __checkMpiErrors(errorCode, __FILE__, __LINE__)
#define checkMpiErrors(err) __checkMpiErrors(err, __FILE__, __LINE__)
#else
#define checkMpiErrors(err) err
#endif

#endif //checkMpiErrors

class MpiGlobalLockGuard {
public:
    explicit MpiGlobalLockGuard(int *argc, char ***argv, const int64_t p, const int64_t q) {
        sGridP = p;
        sGridQ = q;

        int32_t provided;
        checkMpiErrors(MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided));
        init();
        int32_t len = -1;
        sHostName.resize(MPI_MAX_PROCESSOR_NAME);
        checkMpiErrors(MPI_Get_processor_name(sHostName.data(), &len));
        sHostName.resize(len);
    }

    ~MpiGlobalLockGuard() {
        checkMpiErrors(MPI_Finalize());
    }

    static void init(MPI_Comm mpiComm = MPI_COMM_WORLD) {
        checkMpiErrors(MPI_Comm_rank(mpiComm, &sMpiNodeId));
        checkMpiErrors(MPI_Comm_size(mpiComm, &sMpiNumNodes));
        assert(sGridP*sGridQ == int64_t(sMpiNumNodes));
    }
};

// Cublas related --------------------------------------------------------------------------------------------------- //

class CublasGlobalLockGuard {
public:
    explicit CublasGlobalLockGuard(const std::vector<int32_t> &deviceIds): deviceIds_(deviceIds) {
        for(auto deviceId: deviceIds_) {
            checkCudaErrors(cudaSetDevice(deviceId));
            checkCudaErrors(cublasInit());
        }
    }

    ~CublasGlobalLockGuard() {
        for(auto deviceId: deviceIds_) {
            checkCudaErrors(cudaSetDevice(deviceId));
            checkCudaErrors(cublasShutdown());
        }
    }

private:
    std::vector<int32_t> deviceIds_ = {};
};

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

// Random number generator ------------------------------------------------------------------------------------------ //

static unsigned int g_seed = 1;
[[maybe_unused]] inline unsigned int fast_rand() {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

static int32_t sTagCounter = 16384;
inline int32_t tagGenerator() {
    sTagCounter = 16384 + (sTagCounter+1)%16384;
    return sTagCounter;
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

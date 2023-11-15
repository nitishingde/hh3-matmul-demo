#ifndef HH3_MATMUL_UTILITY_H
#define HH3_MATMUL_UTILITY_H

#include <mpi.h>
#include <string>
#include <tclap/CmdLine.h>
#include <unistd.h>

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
#define LOG() printf("[Node %ld][%s:%d]\n", getNodeId(), __FILE__, __LINE__); fflush(stdout)
#define LOG_THREAD_COUNT() printf("[Node %ld][%s:%d][ThreadCount %d]\n", getNodeId(), __FILE__, __LINE__, getThreadCount())
#else
#define LOG()
#define LOG_THREAD_COUNT()
#endif

// MPI Related ------------------------------------------------------------------------------------------------------ //

static int32_t sMpiNodeId    = -1;
static int32_t sMpiNumNodes  = -1;
static int64_t sGridP        = -1;
static int64_t sGridQ        = -1;
static std::string sHostName = {};
std::mutex         mpiMutex  = {};

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
    explicit MpiGlobalLockGuard(int *argc, char ***argv, const int64_t p, const int64_t q, const int32_t flag = MPI_THREAD_MULTIPLE) {
        sGridP = p;
        sGridQ = q;

        int32_t provided;
        checkMpiErrors(MPI_Init_thread(argc, argv, flag, &provided));
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

int getThreadCount() {
    auto pid = getpid();
    int count = 0;
    for(const auto &dirEntry: std::filesystem::directory_iterator("/proc/"+std::to_string(pid)+"/task/")) {
        if(dirEntry.is_directory()) {
            count++;
        }
    }
    return count;
}

#endif //HH3_MATMUL_UTILITY_H

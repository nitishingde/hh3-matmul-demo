#ifndef HH3_MATMUL_COMMON_UTILITY_H
#define HH3_MATMUL_COMMON_UTILITY_H

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
#define LOG() printf("[Node %ld][%s:%d]\n", getNodeId(), __FILE__, __LINE__)
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
    if(errorCode == MPI_SUCCESS or (errorCode < 0 or MPI_ERR_LASTCODE <= errorCode)) return;
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

// Number Generator ------------------------------------------------------------------------------------------------- //

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

// Thread Counter --------------------------------------------------------------------------------------------------- //

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

#endif //HH3_MATMUL_COMMON_UTILITY_H

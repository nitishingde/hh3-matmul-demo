#ifndef HH3_MATMUL_UTILITY_H
#define HH3_MATMUL_UTILITY_H

#include <mpi.h>
#include <string>
#include <tclap/CmdLine.h>

// MPI Related ------------------------------------------------------------------------------------------------------ //

static int32_t sMpiNodeId    = -1;
static int32_t sMpiNumNodes  = -1;
static std::string sHostName = {};

[[nodiscard]] int64_t getNodeId() {
    return sMpiNodeId;
}

[[nodiscard]] int64_t getNumNodes() {
    return sMpiNumNodes;
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
    explicit MpiGlobalLockGuard(int *argc, char ***argv) {
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

    void init(MPI_Comm mpiComm = MPI_COMM_WORLD) {
        checkMpiErrors(MPI_Comm_rank(mpiComm, &sMpiNodeId));
        checkMpiErrors(MPI_Comm_size(mpiComm, &sMpiNumNodes));
    }
};

// Argument parser -------------------------------------------------------------------------------------------------- //

auto parseArgs(int argc, char **argv) {
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
        TCLAP::ValueArg<int64_t> argProdThreads("t", "prod", "product threads", false, 4, "non negative integer value");
        cmd.add(argProdThreads);
        TCLAP::ValueArg<std::string> argPath("P", "path", "scratch/tmp dir path", false, "./", "dir path");
        cmd.add(argPath);
        TCLAP::ValueArg<std::string> argHostFile("H", "hostfile", "path to hostfile", false, "", "file path");
        cmd.add(argHostFile);

        cmd.parse(argc, argv);
        return std::make_tuple(argP.getValue(), argQ.getValue(), argM.getValue(), argK.getValue(), argN.getValue(), argT.getValue(), argProdThreads.getValue(), argPath.getValue(), argHostFile.getValue());
    }
    catch (TCLAP::ArgException &e) {
        fprintf(stderr, "[Error] %s for arg %s\n", e.error().c_str(), e.argId().c_str());
        return std::make_tuple(int64_t(1), int64_t(1), int64_t(32768), int64_t(32768), int64_t(32768), int64_t(32768), int64_t(4), std::string("./"), std::string(""));
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


#endif //HH3_MATMUL_UTILITY_H

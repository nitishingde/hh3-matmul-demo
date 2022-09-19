#ifndef HH3_MATMUL_UTILITY_H
#define HH3_MATMUL_UTILITY_H

#include <hedgehog/hedgehog.h>
#include <random>

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

#include <mpi.h>
#include <tclap/CmdLine.h>

// MPI Related ------------------------------------------------------------------------------------------------------ //

static int32_t sMpiNodeId = 0;
static int32_t sMpiNumNodes = 1;

[[nodiscard]] int32_t getNodeId() {
    return sMpiNodeId;
}

[[nodiscard]] int32_t getNumNodes() {
    return sMpiNumNodes;
}

[[nodiscard]] bool isRootNodeId() {
    return sMpiNodeId == 0;
}

class MpiGlobalLockGuard {
public:
    explicit MpiGlobalLockGuard(int *argc, char ***argv) {
        int32_t mpiNodeId = -1, mpiNumNodes = -1;
        int32_t provided;
        if(MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided) == MPI_SUCCESS) {
            int32_t flag = false;
            if(auto status = MPI_Initialized(&flag); status == MPI_SUCCESS and flag) {
                MPI_Comm_rank(MPI_COMM_WORLD, &sMpiNodeId);
                MPI_Comm_size(MPI_COMM_WORLD, &sMpiNumNodes);
            } else {
                printf("[MPI ERROR] %s:%d", __FILE__, __LINE__);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }

    ~MpiGlobalLockGuard() {
        if(MPI_Finalize() != MPI_SUCCESS) {
            printf("[MPI ERROR] %s:%d", __FILE__, __LINE__);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
};

// Cublas related --------------------------------------------------------------------------------------------------- //

class CublasGlobalLockGuard {
public:
    explicit CublasGlobalLockGuard() {
        checkCudaErrors(cudaGetDeviceCount(&deviceCount_));
        for(int deviceId = 0; deviceId < deviceCount_; ++deviceId) {
            checkCudaErrors(cudaSetDevice(deviceId));
            checkCudaErrors(cublasInit());
        }
    }

    ~CublasGlobalLockGuard() {
        for(int deviceId = 0; deviceId < deviceCount_; ++deviceId) {
            checkCudaErrors(cudaSetDevice(deviceId));
            checkCudaErrors(cublasShutdown());
        }
    }

private:
    int32_t deviceCount_ = 0;
};

// Argument parser -------------------------------------------------------------------------------------------------- //

std::tuple<uint64_t, uint64_t, uint64_t, uint64_t, std::string> parseArgs(int argc, char **argv) {
    try {
        TCLAP::CmdLine cmd("Command description message", ' ', "1.0");

        TCLAP::ValueArg<uint64_t> argM("M", "mdim", "height of matrix A / height of matrix C", false, 32768, "non negative integer value");
        cmd.add(argM);
        TCLAP::ValueArg<uint64_t> argK("K", "kdim", "width of matrix A / height of matrix B",  false, 32768, "non negative integer value");
        cmd.add(argK);
        TCLAP::ValueArg<uint64_t> argN("N", "ndim", "width of matrix B / width of matrix C",   false, 32768, "non negative integer value");
        cmd.add(argN);
        TCLAP::ValueArg<uint64_t> argT("T", "tileSize", "tile size",                           false,  8192, "non negative integer value");
        cmd.add(argT);
        TCLAP::ValueArg<std::string> argP("P", "path", "scratch/tmp dir path",                false,  "./", "dir path");
        cmd.add(argP);

        cmd.parse(argc, argv);
        return {argM.getValue(), argK.getValue(), argN.getValue(), argT.getValue(), argP.getValue()};
    }
    catch (TCLAP::ArgException e) {
        fprintf(stderr, "[Error] %s for arg %s\n", e.error().c_str(), e.argId().c_str());
        return {32768, 32768, 32768, 8192, "./"};
    }
}

// Random number generator ------------------------------------------------------------------------------------------ //

static unsigned int g_seed = 1;
inline int fastrand() {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

template <class Type>
std::tuple<std::uniform_real_distribution<Type>, std::mt19937_64>
MersenneTwisterRandomGenerator(Type start = 0, Type end = 10) {
    // Mersenne Twister Random Generator
    uint64_t timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> (uint64_t) 32)};
    std::mt19937_64 rng(ss);
    // Choose your distribution depending on the type of MatrixType
    std::uniform_real_distribution<Type> unif(start, end);

    return std::make_tuple(unif, rng);
}

// Type Context Manager --------------------------------------------------------------------------------------------- //

namespace Internal_ {
    class TypeContextManager;
    static std::shared_ptr<TypeContextManager> sInstance = nullptr;

    struct BaseType {
    public:
        uint32_t id;
        explicit BaseType(uint32_t id): id(id) {}
        [[nodiscard]] virtual uint32_t getId() { return id; }
    };

    template<class Type>
    struct MappedType: public BaseType {
    public:
        explicit MappedType(uint32_t id, const Type &data): BaseType(id), data_(data) {}
        [[nodiscard]] const Type& get() const { return data_; }

    private:
        Type data_ = {};
    };

    class TypeContextManager {
    public:
        explicit TypeContextManager() = default;
        ~TypeContextManager() = default;

        static std::shared_ptr<TypeContextManager> getInstance() {
            if(sInstance == nullptr) {
                sInstance = std::make_shared<TypeContextManager>();
            }

            return sInstance;
        }

        template<class Type>
        void emplace(uint32_t id, const Type &data) {
            map_.emplace_back(new MappedType<Type>(id, data));
        }

        template<class Type>
        [[nodiscard]] const Type& get(uint32_t id) {
            for(BaseType *baseType: map_) {
                if(baseType->id == id) {
                    auto typedData = static_cast<MappedType<Type>*>(baseType);
                    return typedData->get();
                }
            }

            throw std::runtime_error("Type is not registered for the given contextId.");
        }

    private:
        std::vector<BaseType*> map_;
    };
}

template<class Type>
void registerContext(uint32_t contextId, const Type &data) {
    Internal_::TypeContextManager::getInstance()->template emplace(contextId, data);
}

template<class Type>
const Type& getContext(uint32_t contextId) {
    return Internal_::TypeContextManager::getInstance()->template get<Type>(contextId);
}

#endif //HH3_MATMUL_UTILITY_H

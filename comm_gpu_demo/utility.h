#ifndef HH3_MATMUL_UTILITY_H
#define HH3_MATMUL_UTILITY_H

#include <hedgehog/hedgehog.h>
#include <random>

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

class CublasLockGuard {
private:
    int32_t deviceCount_ = 0;
public:
    explicit CublasLockGuard() {
        checkCudaErrors(cudaGetDeviceCount(&deviceCount_));
        for(int deviceId = 0; deviceId < deviceCount_; ++deviceId) {
            checkCudaErrors(cudaSetDevice(deviceId));
            checkCudaErrors(cublasInit());
        }
    }

    ~CublasLockGuard() {
        for(int deviceId = 0; deviceId < deviceCount_; ++deviceId) {
            checkCudaErrors(cudaSetDevice(deviceId));
            checkCudaErrors(cublasShutdown());
        }
    }
};

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

#endif //HH3_MATMUL_UTILITY_H

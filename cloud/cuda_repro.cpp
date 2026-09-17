// Minimal reproducer for Thunder Compute: which CUDA driver API calls around host-mapped (pinned) memory fail.
// Builds with only g++ and libcuda.so (no toolkit):  g++ -O1 -o cuda_repro cuda_repro.cpp -ldl && ./cuda_repro
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef int CUresult; typedef int CUdevice; typedef void* CUcontext; typedef unsigned long long CUdeviceptr;
#define CU_MEMHOSTALLOC_PORTABLE 1
#define CU_MEMHOSTALLOC_DEVICEMAP 2
#define CU_MEMHOSTREGISTER_DEVICEMAP 2

int main() {
    void* lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) { printf("dlopen libcuda.so.1 failed: %s\n", dlerror()); return 1; }
#define F(name, ...) auto name = (CUresult(*)(__VA_ARGS__))dlsym(lib, #name); if (!name) { printf("missing %s\n", #name); return 1; }
    F(cuInit, unsigned)
    F(cuDeviceGet, CUdevice*, int)
    F(cuDevicePrimaryCtxRetain, CUcontext*, CUdevice)
    F(cuCtxSetCurrent, CUcontext)
    F(cuMemHostAlloc, void**, size_t, unsigned)
    F(cuMemHostGetDevicePointer_v2, CUdeviceptr*, void*, unsigned)
    F(cuMemHostRegister_v2, void*, size_t, unsigned)
    F(cuMemHostUnregister, void*)
    F(cuMemAlloc_v2, CUdeviceptr*, size_t)
    F(cuMemcpyHtoD_v2, CUdeviceptr, const void*, size_t)
    F(cuMemcpyDtoH_v2, void*, CUdeviceptr, size_t)
    F(cuMemFreeHost, void*)
    auto cuGetErrorName = (CUresult(*)(CUresult, const char**))dlsym(lib, "cuGetErrorName");
    auto name = [&](CUresult r) { const char* s = "?"; if (cuGetErrorName) cuGetErrorName(r, &s); return s; };
#define CHECK(call) do { CUresult r = (call); printf("%-70s -> %d (%s)\n", #call, r, name(r)); } while (0)

    CUdevice dev; CUcontext ctx;
    CHECK(cuInit(0));
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK(cuCtxSetCurrent(ctx));

    const size_t n = 1 << 20;
    // 1. plain device memory + copies: the baseline everything else relies on
    CUdeviceptr d = 0; char* h = (char*)malloc(n); memset(h, 7, n);
    CHECK(cuMemAlloc_v2(&d, n));
    CHECK(cuMemcpyHtoD_v2(d, h, n));
    CHECK(cuMemcpyDtoH_v2(h, d, n));

    // 2. pinned host memory mapped into the device address space (what TensorRT's autotuner does)
    void* pinned = nullptr; CUdeviceptr dp = 0;
    CHECK(cuMemHostAlloc(&pinned, n, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
    CHECK(cuMemHostGetDevicePointer_v2(&dp, pinned, 0));   // <- fails on Thunder Compute
    if (dp) CHECK(cuMemcpyDtoH_v2(h, dp, n));               // reading through the mapped pointer
    if (pinned) CHECK(cuMemFreeHost(pinned));

    // 3. the same via registering ordinary malloc'd memory
    void* reg = malloc(n); CUdeviceptr rp = 0;
    CHECK(cuMemHostRegister_v2(reg, n, CU_MEMHOSTREGISTER_DEVICEMAP));
    CHECK(cuMemHostGetDevicePointer_v2(&rp, reg, 0));       // <- fails on Thunder Compute
    CHECK(cuMemHostUnregister(reg));
    return 0;
}

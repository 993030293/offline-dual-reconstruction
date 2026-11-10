/* Count successful native runtime payloads without changing their calls. */
#define _GNU_SOURCE
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

struct owned_transfer {
    uint64_t bytes;
    int32_t kind;
    int32_t result;
    int32_t api;
    int32_t reserved;
};

static pthread_mutex_t owned_lock = PTHREAD_MUTEX_INITIALIZER;
static struct owned_transfer owned_rows[64];
static uint64_t owned_count = 0;
static int owned_active = 0;
static __thread int owned_depth = 0;

int owned_transfer_begin(void) {
    pthread_mutex_lock(&owned_lock);
    if (owned_active) {
        pthread_mutex_unlock(&owned_lock);
        return 1;
    }
    owned_count = 0;
    memset(owned_rows, 0, sizeof(owned_rows));
    owned_active = 1;
    pthread_mutex_unlock(&owned_lock);
    return 0;
}

uint64_t owned_transfer_end(struct owned_transfer *rows, uint64_t capacity) {
    pthread_mutex_lock(&owned_lock);
    owned_active = 0;
    uint64_t count = owned_count;
    if (capacity >= count && count <= 64) {
        memcpy(rows, owned_rows, count * sizeof(*rows));
    }
    pthread_mutex_unlock(&owned_lock);
    return count;
}

static void owned_record(size_t bytes, enum cudaMemcpyKind kind, cudaError_t result, int api) {
    pthread_mutex_lock(&owned_lock);
    if (owned_active) {
        uint64_t index = owned_count++;
        if (index < 64) {
            owned_rows[index] = (struct owned_transfer){bytes, kind, result, api, 0};
        }
    }
    pthread_mutex_unlock(&owned_lock);
}

const char *owned_transfer_backend(void) {
    static Dl_info info;
    void *symbol = dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return symbol && dladdr(symbol, &info) ? info.dli_fname : "";
}

cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    typedef cudaError_t (*function)(void *, const void *, size_t, enum cudaMemcpyKind);
    function native = (function)dlsym(RTLD_NEXT, "cudaMemcpy");
    if (!native) return cudaErrorSharedObjectSymbolNotFound;
    int outer = owned_depth++ == 0;
    cudaError_t result = native(dst, src, count, kind);
    --owned_depth;
    if (outer) owned_record(count, kind, result, 1);
    return result;
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    typedef cudaError_t (*function)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    function native = (function)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    if (!native) return cudaErrorSharedObjectSymbolNotFound;
    int outer = owned_depth++ == 0;
    cudaError_t result = native(dst, src, count, kind, stream);
    --owned_depth;
    if (outer) owned_record(count, kind, result, 2);
    return result;
}

cudaError_t CUDARTAPI cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind) {
    typedef cudaError_t (*function)(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind);
    function native = (function)dlsym(RTLD_NEXT, "cudaMemcpy2D");
    if (!native) return cudaErrorSharedObjectSymbolNotFound;
    int outer = owned_depth++ == 0;
    cudaError_t result = native(dst, dpitch, src, spitch, width, height, kind);
    --owned_depth;
    if (outer) owned_record(width * height, kind, result, 3);
    return result;
}

cudaError_t CUDARTAPI cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream) {
    typedef cudaError_t (*function)(void *, size_t, const void *, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
    function native = (function)dlsym(RTLD_NEXT, "cudaMemcpy2DAsync");
    if (!native) return cudaErrorSharedObjectSymbolNotFound;
    int outer = owned_depth++ == 0;
    cudaError_t result = native(dst, dpitch, src, spitch, width, height, kind, stream);
    --owned_depth;
    if (outer) owned_record(width * height, kind, result, 4);
    return result;
}

cudaError_t CUDARTAPI cudaMemcpyAsync_ptsz(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
    typedef cudaError_t (*function)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t);
    function native = (function)dlsym(RTLD_NEXT, "cudaMemcpyAsync_ptsz");
    if (!native) return cudaErrorSharedObjectSymbolNotFound;
    int outer = owned_depth++ == 0;
    cudaError_t result = native(dst, src, count, kind, stream);
    --owned_depth;
    if (outer) owned_record(count, kind, result, 5);
    return result;
}

cudaError_t CUDARTAPI cudaMemcpy_ptds(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    typedef cudaError_t (*function)(void *, const void *, size_t, enum cudaMemcpyKind);
    function native = (function)dlsym(RTLD_NEXT, "cudaMemcpy_ptds");
    if (!native) return cudaErrorSharedObjectSymbolNotFound;
    int outer = owned_depth++ == 0;
    cudaError_t result = native(dst, src, count, kind);
    --owned_depth;
    if (outer) owned_record(count, kind, result, 6);
    return result;
}

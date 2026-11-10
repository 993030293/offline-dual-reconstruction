/* Fuse exact interleaved uint32 layout reads with bounded standard SHA-256. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <openssl/evp.h>
#include <immintrin.h>
#include <omp.h>

typedef struct {
    const uint32_t *data;
    uint64_t pixels;
    uint32_t channels;
    uint32_t reserved;
    uint64_t available_bytes;
} ImageInput;

enum { BLOCK_WORDS = 8192, MAX_IMAGES = 4096 };

int fused_digest_abi(void) { return 1; }
int fused_digest_avx2(void) { __builtin_cpu_init(); return __builtin_cpu_supports("avx2") != 0; }
uint32_t fused_digest_descriptor_bytes(void) { return sizeof(ImageInput); }
uint64_t fused_digest_scratch_bytes(void) { return BLOCK_WORDS * sizeof(uint32_t); }

__attribute__((target("avx2")))
static size_t gather_avx2(uint32_t *out, const uint32_t *data, size_t count, uint32_t channels) {
    const __m256i offset = _mm256_set_epi32(7*channels, 6*channels, 5*channels,
        4*channels, 3*channels, 2*channels, channels, 0);
    size_t j = 0;
    for (; j + 8 <= count; j += 8) {
        __m256i values = _mm256_i32gather_epi32((const int *)(data + j*channels), offset, 4);
        _mm256_storeu_si256((__m256i *)(out + j), values);
    }
    return j;
}

static int digest_one(const ImageInput *row, unsigned char *out, int vectorized) {
    uint32_t buffer[BLOCK_WORDS];
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    if (ctx == NULL) return -2;
    int ok = EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    for (uint32_t channel = 0; ok == 1 && channel < row->channels; ++channel) {
        for (uint64_t start = 0; ok == 1 && start < row->pixels; start += BLOCK_WORDS) {
            size_t count = (size_t)(row->pixels - start);
            if (count > BLOCK_WORDS) count = BLOCK_WORDS;
            const uint32_t *data = row->data + start*row->channels + channel;
            size_t j = vectorized ? gather_avx2(buffer, data, count, row->channels) : 0;
            for (; j < count; ++j) memcpy(buffer+j, data+j*row->channels, sizeof(uint32_t));
            ok = EVP_DigestUpdate(ctx, buffer, count*sizeof(uint32_t));
        }
    }
    unsigned int size = 0;
    if (ok == 1) ok = EVP_DigestFinal_ex(ctx, out, &size);
    EVP_MD_CTX_free(ctx);
    return ok == 1 && size == 32 ? 0 : -3;
}

int fused_digest_batch(const ImageInput *rows, uint32_t count, unsigned char *out,
                       int *statuses, uint32_t workers, uint32_t vectorized) {
    if (rows == NULL || out == NULL || statuses == NULL || count == 0 || count > MAX_IMAGES
        || (workers != 1 && workers != 4) || vectorized > 1) return -1;
    if (vectorized && !fused_digest_avx2()) return -4;
    for (uint32_t i = 0; i < count; ++i) {
        const ImageInput *row = rows+i;
        if (row->data == NULL || row->reserved != 0 || row->channels == 0 || row->channels > 4
            || row->pixels == 0 || row->pixels > 67108864ULL
            || row->pixels*row->channels*sizeof(uint32_t) > 268435456ULL
            || row->available_bytes < row->pixels*row->channels*sizeof(uint32_t)) return -1;
    }
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (uint32_t i = 0; i < count; ++i) statuses[i] = digest_one(rows+i, out+32*i, vectorized);
    for (uint32_t i = 0; i < count; ++i) if (statuses[i] != 0) return statuses[i];
    return 0;
}

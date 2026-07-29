#ifndef SRO_SPILL_IO
#define SRO_SPILL_IO

#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <malloc.h>
#include <string>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>

// cuFile transport mechanism
constexpr size_t CUFILE_CHUNK_MAX = 16 * 1024 * 1024;

inline std::string spill_directory() {
    if (const char* env = std::getenv("GPUDECIDE_SPILL_DIR")) return std::string(env);
    return "/tmp";
}

// Write all bytes via cuFile, looping to handle partial writes / chunk size cap.
inline void cuFile_write_all(CUfileHandle_t handle, void* device_ptr,
                             size_t byte_count, off_t file_offset = 0) {
    size_t written = 0;
    while (written < byte_count) {
        size_t chunk = std::min(byte_count - written, CUFILE_CHUNK_MAX);
        ssize_t w = cuFileWrite(handle, device_ptr, chunk,
                                file_offset + static_cast<off_t>(written),
                                static_cast<off_t>(written));
        if (w < 0) {
            throw std::runtime_error("cuFileWrite returned error code "
                                     + std::to_string(w));
        }
        if (w == 0) {
            throw std::runtime_error("cuFileWrite made no progress");
        }
        written += static_cast<size_t>(w);
    }
}

// Read all bytes via cuFile, looping to handle partial reads / chunk size cap.
inline void cuFile_read_all(CUfileHandle_t handle, void* device_ptr,
                            size_t byte_count, off_t file_offset = 0) {
    size_t read_total = 0;
    while (read_total < byte_count) {
        size_t chunk = std::min(byte_count - read_total, CUFILE_CHUNK_MAX);
        ssize_t r = cuFileRead(handle, device_ptr, chunk,
                               file_offset + static_cast<off_t>(read_total),
                               static_cast<off_t>(read_total));
        if (r < 0) {
            throw std::runtime_error("cuFileRead returned error code "
                                     + std::to_string(r));
        }
        if (r == 0) {
            throw std::runtime_error("cuFileRead made no progress (unexpected EOF)");
        }
        read_total += static_cast<size_t>(r);
    }
}

inline void reclaim_device_memory() {
    cudaDeviceSynchronize();
    cudaMemPool_t mp;
    if (cudaDeviceGetDefaultMemPool(&mp, 0) == cudaSuccess) cudaMemPoolTrimTo(mp, 0);
    malloc_trim(0);
    cudaDeviceSynchronize();
}

#endif
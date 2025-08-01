// common_cuda.h
#pragma once
#include <ATen/cuda/CUDAContext.h>

template <typename T>
__forceinline__ __device__ T gpuAtomicAdd(T* address, T val) {
    if constexpr (std::is_same_v<T, float>) {
        return atomicAdd(address, val);
    } else if constexpr (std::is_same_v<T, double>) {
        unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(address);
        unsigned long long int old = *address_as_ull, assumed;
        
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                           __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old);
        
        return __longlong_as_double(old);
    } else if constexpr (std::is_same_v<T, int>) {
        return atomicAdd(address, val);
    } else {
        // static_assert(false, "Unsupported type for gpuAtomicAdd");
        return T(0); // 临时返回默认值
    }
}

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err)
            << " in " << std::string(file)
            << " at line " << line << "\n";
        throw std::runtime_error("cuda error");
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__device__ uint32_t offset(uint32_t i){                     // смещение для избежание конфликтов банков разделяемой памяти
    return i + (i / 32);
}

__device__ void sweep_down(uint32_t* shmem, int32_t tid, uint32_t stride){ // фаза спускания сканирования
    uint32_t r = offset(stride * tid + stride - 1);
    uint32_t l = offset(stride * tid + (stride >> 1) - 1);
    shmem[r] += shmem[l];
}

__device__ void sweep_up(uint32_t* shmem, int32_t tid, uint32_t stride){ // фаза подьема сканирования
    uint32_t r = offset(stride * tid + stride - 1);
    uint32_t l = offset(stride * tid + (stride >> 1) - 1);
    shmem[r] += shmem[l];
    shmem[l] = shmem[r] - shmem[l];
}


__global__ void scan_kern_bs128(uint32_t* hist) {                   // ядро сканирования (exclusive) - строит профиксные суммы

    __shared__ uint32_t shmem[512];                                 // разделяемая память (немного больше тк используются смещения для избежания конфликтов банков памяти)
    shmem[offset(threadIdx.x      )] = hist[threadIdx.x      ];
    shmem[offset(threadIdx.x + 128)] = hist[threadIdx.x + 128];
    __syncthreads();

    for(uint32_t i = 2; i <= 256; i <<= 1){                         // фаза спуска
        if (i * threadIdx.x < 256) {
            sweep_down(shmem, threadIdx.x, i);
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) shmem[offset(255)] = 0; __syncthreads();

    for(uint32_t i = 256; i >= 2; i >>= 1){                         // фаза подьема
        if (i * threadIdx.x < 256) {
            sweep_up(shmem, threadIdx.x, i);
        }
        __syncthreads();
    }

    hist[threadIdx.x      ] = shmem[offset(threadIdx.x      )];
    hist[threadIdx.x + 128] = shmem[offset(threadIdx.x + 128)];     // пишем обратно в DRAM из разделяемой памяти
    __syncthreads();
}




__global__ void hist256_kern(uint8_t* arr, int32_t n,              // ядро гистограммы (подсчет встречаемости каждого числа в массиве)
                                        uint32_t* hist) {
    const int32_t s = gridDim.x * blockDim.x;
    for (int32_t i = blockIdx.x * blockDim.x
                + threadIdx.x; i < n; i += s) {
        atomicAdd(&hist[arr[i]], 1u);
    }
}


__global__ void sort256_kern(uint8_t* arr, uint8_t* res,            // ядро сортировки массива (сортировка подсчетом)
                             int32_t n, uint32_t* hist) {           // использует префиксные суммы которые сформировали ранее
    const int32_t s = gridDim.x * blockDim.x;
    for (int32_t i = blockIdx.x * blockDim.x
                + threadIdx.x; i < n; i += s) {
        uint8_t num = arr[i];
        const int32_t t = atomicAdd(&hist[num], 1);
        res[t] = num;
    }

}

const int32_t hist_size = 256;

int main()
{

    uint32_t arr_size = 0;
    std::cin.read(reinterpret_cast<char*>(&arr_size), sizeof(uint32_t));


    std::vector<uint8_t> arr(arr_size);
    std::cin.read(reinterpret_cast<char*>(arr.data()), arr_size * sizeof(uint8_t));        // считываем размер массива и сам массив

    uint8_t* dev_arr = nullptr;
    HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(
                &dev_arr), arr_size * sizeof(uint8_t)));
    HANDLE_ERROR(cudaMemcpy(dev_arr, arr.data(), arr_size *
                sizeof(uint8_t), cudaMemcpyHostToDevice));

    uint8_t* dev_res = nullptr;
    HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(
                &dev_res), arr_size * sizeof(uint8_t)));


    uint32_t* dev_hist = nullptr;                                                            // память в DRAM для гистограммы
    HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(
                &dev_hist), hist_size * sizeof(uint32_t)));

    HANDLE_ERROR(cudaMemset(dev_hist, 0,
                            hist_size * sizeof(uint32_t)));                                  // обнуляем все счетчики гистограммы

    float total_duration = 0, duration = 0;
	cudaEvent_t start_event, stop_event;
	HANDLE_ERROR(cudaEventCreate(&start_event));
	HANDLE_ERROR(cudaEventCreate(&stop_event));


    HANDLE_ERROR(cudaEventRecord(start_event));                                             // строим гистограмму на GPU
    hist256_kern <<< 32, 256 >>> (
                        dev_arr, arr_size, dev_hist);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaEventRecord(stop_event));
    HANDLE_ERROR(cudaEventSynchronize(stop_event));
    HANDLE_ERROR(cudaEventElapsedTime(
        &duration, start_event, stop_event));
    total_duration += duration;

    HANDLE_ERROR(cudaEventRecord(start_event));
    scan_kern_bs128 <<< 1, 128 >>> (dev_hist);                                              //  считаем префиксные суммы
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaEventRecord(stop_event));
    HANDLE_ERROR(cudaEventSynchronize(stop_event));
    HANDLE_ERROR(cudaEventElapsedTime(
        &duration, start_event, stop_event));
    total_duration += duration;


    HANDLE_ERROR(cudaEventRecord(start_event));
    sort256_kern <<< 32, 256 >>> (                                                          // сортируем подсчетом на GPU
            dev_arr, dev_res, arr_size, dev_hist);
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaEventRecord(stop_event));
    HANDLE_ERROR(cudaEventSynchronize(stop_event));
    HANDLE_ERROR(cudaEventElapsedTime(
        &duration, start_event, stop_event));
    total_duration += duration;

    HANDLE_ERROR(cudaEventDestroy(start_event));
	HANDLE_ERROR(cudaEventDestroy(stop_event));

    HANDLE_ERROR(cudaMemcpy(arr.data(), dev_res,                                            // копируем отсортированный массив
                        arr_size * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(dev_arr));
	HANDLE_ERROR(cudaFree(dev_hist));
    HANDLE_ERROR(cudaFree(dev_res));

    std::cout.write(reinterpret_cast<char*>(arr.data()), arr_size * sizeof(uint8_t));       // пишем результат
    return 0;
}

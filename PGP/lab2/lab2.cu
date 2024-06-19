#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err)
            << " in " << std::string(file)
            << " at line " << line << "\n";
        throw std::runtime_error("cuda exception");
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


struct ssaa_context_t {   // аргументы ядра
    uchar4* out;          // сглаженное изображение
    int32_t width;        // ширина сглаженного изображение
    int32_t height;       // высота сглаженного изображение
    int32_t height_ratio; // коэффициент сглаживания по высоте
    int32_t width_ratio;  // коэффициент сглаживания по ширине
};

texture<uchar4, cudaTextureType2D, cudaReadModeElementType> t_ref; // текстурная ссылка

__global__ void ssaa_kernel(ssaa_context_t context) {

    const int32_t block_index = gridDim.x * blockIdx.y + blockIdx.x;        // линейный индекс блока относительно сетки
    const int32_t thread_index = blockDim.x * threadIdx.y + threadIdx.x;    // линейный индекс потока относительно блока
    const int32_t block_size = blockDim.x * blockDim.y;                     // размер блока
    const int32_t ssaa_volume = context.width_ratio * context.height_ratio; // размер блока сглаживания
    const int32_t grid_size = gridDim.x * gridDim.y * block_size;           // размер сетки в потоках
    const int32_t image_size = context.width * context.height;              // размер изображения


    for(int32_t k = block_index * block_size + thread_index;
                            k < image_size; k += grid_size)  {
        int32_t x = (k % context.width) * context.width_ratio;               // координаты пикселя
        int32_t y = (k / context.width) * context.height_ratio;              // координаты пикселя
        int32_t sx = 0, sy = 0, sz = 0, sw = 0;                              // суммы по значениям каналов сглаживаемого блока

        for(int32_t i = 0; i < context.width_ratio; ++i){                    // суммируем каналы в блоке сглаживания
            for(int32_t j = 0; j < context.height_ratio; ++j){
                uchar4 p = tex2D(t_ref, x + i, y + j);
                sx += static_cast<unsigned int>(p.x);
                sy += static_cast<unsigned int>(p.y);
                sz += static_cast<unsigned int>(p.z);
            }
        }
        context.out[k].x = static_cast<unsigned char>(sx / ssaa_volume);    // усредняем эти значения и записываем
        context.out[k].y = static_cast<unsigned char>(sy / ssaa_volume);
        context.out[k].z = static_cast<unsigned char>(sz / ssaa_volume);
        context.out[k].w = static_cast<unsigned char>(sw / ssaa_volume);
    }
}
int main() {

    std::string in_fn; std::getline(std::cin, in_fn);                       // считываем входное изображение
    std::fstream in_bytes(in_fn, std::ios::binary | std::ios::in);
    if (!in_bytes) {
        throw std::runtime_error("failed to open input file");
    }
    int32_t origin_width  = 0, origin_height = 0;
    in_bytes.read(reinterpret_cast<char*>(&origin_width),  sizeof(int32_t)); // размеры изображения записаны в первых байтах
    in_bytes.read(reinterpret_cast<char*>(&origin_height), sizeof(int32_t));

    std::cerr << "origin dimensions: " << origin_width
             << " x " << origin_height << std::endl;

    const int32_t origin_pixels_count = origin_height * origin_width;
    auto origin_bytes = new uchar4[origin_pixels_count];   // создаем массив под входное изображение
    for (int32_t i = 0; i < origin_pixels_count; ++i) {
        in_bytes.read(reinterpret_cast<char*>(&(origin_bytes[i].x)), sizeof(uchar4::x));
        in_bytes.read(reinterpret_cast<char*>(&(origin_bytes[i].y)), sizeof(uchar4::y));
        in_bytes.read(reinterpret_cast<char*>(&(origin_bytes[i].z)), sizeof(uchar4::z));
        in_bytes.read(reinterpret_cast<char*>(&(origin_bytes[i].w)), sizeof(uchar4::w));
    }
    std::string out_fn; std::getline(std::cin, out_fn);                    // считываем имя выходного файла

    int32_t target_width = 0, target_height = 0;
    std::cin >> target_width >> target_height;                            // считываем результирующие размеры
    const int32_t target_pixels_count = target_height * target_width;




    cudaArray_t dev_origin_bytes;
    cudaChannelFormatDesc cuda_chf = cudaCreateChannelDesc<uchar4>();
    HANDLE_ERROR(cudaMallocArray(&dev_origin_bytes, &cuda_chf,
                                origin_width, origin_height));            // создаем cuda массив для текстурной ссылки


    HANDLE_ERROR(cudaMemcpyToArray(dev_origin_bytes, 0, 0,
                                origin_bytes,
                                origin_pixels_count * sizeof(uchar4),
                                cudaMemcpyHostToDevice));                 // копируем изображение в cuda массив

    uchar4* dev_target_bytes = nullptr;
    HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&dev_target_bytes),  // выделяем память под результирующее изображение
                            target_pixels_count * sizeof(uchar4)));

    t_ref.normalized = false;
    t_ref.filterMode = cudaFilterModePoint;
    t_ref.addressMode[0] = cudaAddressModeClamp;
    t_ref.addressMode[1] = cudaAddressModeClamp;

    HANDLE_ERROR(cudaBindTextureToArray(&t_ref, dev_origin_bytes, &cuda_chf)); // подключаем cuda массив к текстурной ссылке

    const dim3 grid_dim(128, 128, 1), block_dim(32, 32, 1);

    cudaEvent_t start_event, stop_event;
    HANDLE_ERROR(cudaEventCreate(&start_event));
    HANDLE_ERROR(cudaEventCreate(&stop_event));                               // создаем и регистрируем события запуска чтобы подчитать затраченное время
    ssaa_context_t context;
    context.width = target_width;
    context.height = target_height;
    context.width_ratio = origin_width / target_width;
    context.height_ratio = origin_height / target_height;
    context.out = dev_target_bytes;
    HANDLE_ERROR(cudaEventRecord(start_event));
    ssaa_kernel<<<grid_dim, block_dim>>>(context);                            // запускаем ядро сглаживания
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaEventRecord(stop_event));
    HANDLE_ERROR(cudaEventSynchronize(stop_event));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start_event, stop_event)); // считаем затраченное время
    HANDLE_ERROR(cudaEventDestroy(start_event));
    HANDLE_ERROR(cudaEventDestroy(stop_event));

    std::cerr << "passed " << elapsedTime << "\n";

    HANDLE_ERROR(cudaMemcpy(origin_bytes, dev_target_bytes,             // копируем результат на RAM
                            target_pixels_count * sizeof(uchar4),
                            cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaUnbindTexture(t_ref));                                   // освобождаем ресурсы
    HANDLE_ERROR(cudaFreeArray(dev_origin_bytes));
    HANDLE_ERROR(cudaFree(dev_target_bytes));



    std::fstream out_bytes(out_fn, std::ios::binary | std::ios::out);        // записываем файл с результатом
    if (!out_bytes) {
        throw std::runtime_error("failed to create output file");
    }

    out_bytes.write(reinterpret_cast<char*>(&target_width), sizeof(int32_t));
    out_bytes.write(reinterpret_cast<char*>(&target_height), sizeof(int32_t));
    for (int32_t i = 0; i < target_pixels_count; ++i) {
        out_bytes.write(reinterpret_cast<char*>(&(origin_bytes[i].x)), sizeof(uchar4::x));
        out_bytes.write(reinterpret_cast<char*>(&(origin_bytes[i].y)), sizeof(uchar4::y));
        out_bytes.write(reinterpret_cast<char*>(&(origin_bytes[i].z)), sizeof(uchar4::z));
        out_bytes.write(reinterpret_cast<char*>(&(origin_bytes[i].w)), sizeof(uchar4::w));
    }
    delete [](origin_bytes);
    return 0;
}

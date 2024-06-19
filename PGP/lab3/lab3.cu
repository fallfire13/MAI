#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err)
            << " in " << std::string(file)
            << " at line " << line << "\n";
        throw std::runtime_error("cuda error");
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct cvals_t {
    double avg[3];
    double cov[9];
    double logd;
};                                                                         // стат данные


__device__ double eval(const uchar4& pixel, const cvals_t& c) {            // функция с формулой
    double a[3] = {
        pixel.x - c.avg[0],
        pixel.y - c.avg[1],
        pixel.z - c.avg[2],
    };

    const double* cov = c.cov;
    double b[3] = {
        -a[0] * cov[0] - a[1] * cov[3] - a[2] * cov[6],
        -a[0] * cov[1] - a[1] * cov[4] - a[2] * cov[7],
        -a[0] * cov[2] - a[1] * cov[5] - a[2] * cov[8],
    };


    double res = c.logd;
    for(int32_t i = 0; i < 3; ++i){
        res += a[i] * b[i];
    }
    return res;
}



__constant__ cvals_t dev_cvals[32];

__global__ void kern(uchar4* const image_bytes,
    const int32_t pixel_count, const int32_t nc) {

    const int32_t block_index = gridDim.x * blockIdx.y + blockIdx.x;        // линейный индекс блока относительно сетки
    const int32_t thread_index = blockDim.x * threadIdx.y + threadIdx.x;    // линейный индекс потока относительно блока
    const int32_t block_size = blockDim.x * blockDim.y;                     // размер блока
    const int32_t grid_size = gridDim.x * gridDim.y * block_size;           // размер сетки в потоках


    for(int32_t k = block_index * block_size + thread_index;                // считаем номер класса для пикселей
                            k < pixel_count; k += grid_size)  {
        double max_val = eval(image_bytes[k], dev_cvals[0]);
        unsigned char& max_idx = image_bytes[k].w; max_idx = 0;
        for(int32_t j = 1; j < nc; ++j){
            double val = eval(image_bytes[k], dev_cvals[j]);
            if (val > max_val) {max_idx = j; max_val = val;}
        }
    }
}



double invm(double* m) {                                                        // считаем обратную матрицу
    double d = m[0] * m[4] * m[8]
            + m[1] * m[5] * m[6]
            + m[3] * m[7] * m[2]
            - m[2] * m[4] * m[6]
            - m[0] * m[5] * m[7]
            - m[1] * m[3] * m[8];
    double t[9] = {
        (m[4] * m[8] - m[5] * m[7]),
        -(m[1] * m[8] - m[2] * m[7]),
        (m[1] * m[5] - m[2] * m[4]),
        -(m[3] * m[8] - m[5] * m[6]),
        (m[0] * m[8] - m[2] * m[6]),
        -(m[0] * m[5] - m[2] * m[3]),
        (m[3] * m[7] - m[4] * m[6]),
        -(m[0] * m[7] - m[1] * m[6]),
        (m[0] * m[4] - m[1] * m[3])
    };
    for(int32_t i = 0; i < 9; ++i){
        m[i] = t[i] / d;
    }
    return -std::log(d);
}


int main() {

    std::string in_fn, out_fn;                                                      // считываем входные данные
    int32_t image_width = 0, image_height = 0, pixels_count = 0;
    uchar4* image_bytes = nullptr;
    {
        std::getline(std::cin, in_fn);
        std::fstream in_stream(in_fn, std::ios::binary | std::ios::in);
        if (!in_stream) {
            throw std::runtime_error("failed to open input file");
        }
        std::getline(std::cin, out_fn);
        in_stream.read(
            reinterpret_cast<char*>(&image_width), sizeof(int32_t));
        in_stream.read(
            reinterpret_cast<char*>(&image_height), sizeof(int32_t));
        pixels_count = image_width * image_height;
        image_bytes = new uchar4[pixels_count];
        for (int32_t i = 0; i < pixels_count; ++i) {
            in_stream.read(reinterpret_cast<char*>(
                            &image_bytes[i].x), sizeof(uchar4::x));
            in_stream.read(reinterpret_cast<char*>(
                            &image_bytes[i].y), sizeof(uchar4::y));
            in_stream.read(reinterpret_cast<char*>(
                            &image_bytes[i].z), sizeof(uchar4::z));
            in_stream.read(reinterpret_cast<char*>(
                            &image_bytes[i].w), sizeof(uchar4::w));
        }
    }

    int32_t nc = 0; std::cin >> nc;
    std::vector<std::vector<int32_t>> ps(nc);
    std::vector<cvals_t> cvals(nc);

    {                                                                               // считаем матрицу ковариации и детерминант
        for (int32_t j = 0; j < nc; ++j) {
            int32_t npj = 0; std::cin >> npj;
            ps[j].reserve(npj);
            for (int32_t t = 0; t < npj; ++t) {
                int32_t pi = 0; std::cin >> pi;
                int32_t pj = 0; std::cin >> pj;
                int32_t pk = pj * image_width + pi;
                ps[j].push_back(pk);
            }
        }
        for (int32_t j = 0; j <  nc; ++j) {
            const int32_t npj = ps[j].size();
            for(int32_t t = 0; t < 3; ++t){
                cvals[j].avg[t] = 0.0;
            }
            for (int32_t t = 0; t < npj; ++t) {
                cvals[j].avg[0] += image_bytes[ps[j][t]].x;
                cvals[j].avg[1] += image_bytes[ps[j][t]].y;
                cvals[j].avg[2] += image_bytes[ps[j][t]].z;
            }
            for(int32_t t = 0; t < 3; ++t){
                cvals[j].avg[t] /= npj;
            }
        }
        for (int32_t j = 0; j < nc; ++j) {
            const int32_t npj = ps[j].size();
            for(int32_t t = 0; t < 9; ++t){
                cvals[j].cov[t] = 0.0;
            }
            for (int32_t t = 0; t < npj; ++t) {
                uchar4 px = image_bytes[ps[j][t]];
                double x = px.x - cvals[j].avg[0];
                double y = px.y - cvals[j].avg[1];
                double z = px.z - cvals[j].avg[2];
                cvals[j].cov[0] += x * x;
                cvals[j].cov[1] += x * y;
                cvals[j].cov[2] += x * z;
                cvals[j].cov[3] += x * y;
                cvals[j].cov[4] += y * y;
                cvals[j].cov[5] += y * z;
                cvals[j].cov[6] += x * z;
                cvals[j].cov[7] += y * z;
                cvals[j].cov[8] += z * z;
            }
            for(int32_t i = 0; i < 9; ++i){
                cvals[j].cov[i] /= npj - 1;
            }
            cvals[j].logd = invm(cvals[j].cov);
        }
    }



    uchar4* dev_image_bytes = nullptr;
    HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&dev_image_bytes),             // аллоцируем память под изображение
                                        pixels_count * sizeof(uchar4)));

    HANDLE_ERROR(cudaMemcpy(dev_image_bytes, image_bytes, pixels_count
                            * sizeof(uchar4), cudaMemcpyHostToDevice));              // копируем изображение в DRAM

    HANDLE_ERROR(cudaMemcpyToSymbol(dev_cvals, cvals.data(), nc * sizeof(cvals_t))); // копируем стат данные в константную память

    dim3 block_dim(32, 1, 1);
    dim3 grid_dim(32, 32, 1);

    cudaEvent_t start_event, stop_event;
    HANDLE_ERROR(cudaEventCreate(&start_event));                                   // создаем и регистрируем события запуска чтобы подсчитать затраченное время
    HANDLE_ERROR(cudaEventCreate(&stop_event));
    HANDLE_ERROR(cudaEventRecord(start_event));
    kern<<<grid_dim, block_dim >>> (dev_image_bytes, pixels_count, nc);             // запускаем ядро классификации
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaEventRecord(stop_event));
    HANDLE_ERROR(cudaEventSynchronize(stop_event));
    float elapsed_time;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));     // считаем затраченное время
    HANDLE_ERROR(cudaEventDestroy(start_event));
    HANDLE_ERROR(cudaEventDestroy(stop_event));
    std::cerr << "duration " << elapsed_time << "\n";


    HANDLE_ERROR(cudaMemcpy(image_bytes, dev_image_bytes, pixels_count              // копируем изображение из DRAM в RAM
                            * sizeof(uchar4), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_image_bytes));                                        // освобождаем память на GPU

                                                                                    // записываем результат обратно в файл
    {
        std::fstream out_stream(out_fn, std::ios::binary | std::ios::out);
        if (!out_stream) {
            throw std::runtime_error("failed to open output fil");
        }
        out_stream.write(
            reinterpret_cast<char*>(&image_width), sizeof(int32_t));
        out_stream.write(
            reinterpret_cast<char*>(&image_height), sizeof(int32_t));

        for (int i = 0; i < pixels_count; ++i) {
            out_stream.write(reinterpret_cast<char*>(
                        &(image_bytes[i].x)), sizeof(uchar4::x));
            out_stream.write(reinterpret_cast<char*>(
                        &(image_bytes[i].y)), sizeof(uchar4::y));
            out_stream.write(reinterpret_cast<char*>(
                        &(image_bytes[i].z)), sizeof(uchar4::z));
            out_stream.write(reinterpret_cast<char*>(
                        &(image_bytes[i].w)), sizeof(uchar4::w));
        }
    }
    delete[](image_bytes);
    return 0;
}

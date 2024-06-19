#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err)
            << " in " << std::string(file)
            << " at line " << line << "\n";
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__global__ void sum_kernel(double* a, double* b, int n) {          // ядро складывающее два вектора на GPU
	for(int i =  blockDim.x * blockIdx.x + threadIdx.x;
                    i < n; i += blockDim.x * gridDim.x){
		a[i] += b[i];
	}
}


int main() {
	std::cout.tie(nullptr);
	std::cin.tie(nullptr);
	std::ios::sync_with_stdio(false);  // ускоряем ввод

	int n; std::cin >> n;

	double* host_arr = nullptr;        // выделяем память под массив на RAM
	host_arr = new double[n];


	for(int i = 0; i < n; ++i){
		std::cin >> host_arr[i];      // считываем элементы первого массива
	}
	double* dev_arr1 = nullptr;       // выделяем память под первый вектор на GPU
	HANDLE_ERROR(cudaMalloc(&dev_arr1, n * sizeof(double)));
	HANDLE_ERROR(cudaMemcpy(
		reinterpret_cast<void*>(dev_arr1),
		reinterpret_cast<void*>(host_arr),
		n * sizeof(double), cudaMemcpyHostToDevice
	));                              // копируем элементы первого вектора на GPU

	for(int i = 0; i < n; ++i){
		std::cin >> host_arr[i];    // считываем элементы второго массива
	}
	double* dev_arr2 = nullptr;     // выделяем память под второй вектор на GPU
	HANDLE_ERROR(cudaMalloc(&dev_arr2, n * sizeof(double)));
	HANDLE_ERROR(cudaMemcpy(
		reinterpret_cast<void*>(dev_arr2),
		reinterpret_cast<void*>(host_arr),
		n * sizeof(double), cudaMemcpyHostToDevice
	));                              // копируем элементы второго вектора на GPU

	dim3 blocks(256), grid(256);
	sum_kernel<<<grid, blocks>>>(dev_arr1, dev_arr2, n); // запускаем сложение на GPU
	HANDLE_ERROR(cudaDeviceSynchronize());               // ждем завершения операции
	HANDLE_ERROR(cudaGetLastError());

	HANDLE_ERROR(cudaMemcpy(                             // копируем результат на хост RAM
		reinterpret_cast<void*>(host_arr),
		reinterpret_cast<void*>(dev_arr1),
		n * sizeof(double), cudaMemcpyDeviceToHost
	));

	HANDLE_ERROR(cudaFree(dev_arr1));                    //  освобождаем память на GPU
	HANDLE_ERROR(cudaFree(dev_arr2));

	std::cout << std::scientific << std::setprecision(10);

	for(int i = 0; i < n; ++i){
		std::cout << host_arr[i] << " ";                // выводим результат на экран
	}
	delete[](host_arr);                                 // освобождаем память на хосте
	std::cout << std::endl;
	return 0;
}

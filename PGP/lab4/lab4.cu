#include <string>
#include <stdio.h>
#include <iomanip>
#include <vector>
#include <cuda.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>


__device__ double atomic_add(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    } while (assumed != old);

    return __longlong_as_double(old);
}


const double EPS = 1e-7;
const dim3 sub_threads(8,  4), sub_blocks(16, 16),
		   div_threads(32   ), div_blocks(128   ),
		   swp_threads(32   ), swp_blocks(128   ),
		   slv_threads(32   ), slv_blocks(128   );

static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		std::cerr << cudaGetErrorString(err)
			<< " in " << std::string(file)
			<< " at line " << line << "\n";
		throw std::runtime_error("cuda error");
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ void sub_kern(double* m, int32_t k, int32_t n) {         // ядро вычитания строк под опорной строкой
	const int32_t sy = gridDim.y * blockDim.y;
	const int32_t sx = gridDim.x * blockDim.x;
	for (int i = blockIdx.y * blockDim.y +
	threadIdx.y + k + 1;  i <= n; i += sy) {
		for(int j = blockIdx.x * blockDim.x +
		threadIdx.x + k + 1; j < n; j += sx) {
			m[i * n + j] -= m[k * n + j] * m[i * n + k];
		}
	}
}

__global__ void swp_kern(double* m, int32_t i1, int32_t i2, int32_t n) {  // ядро перестановки (обмена) двух строк
	const int32_t s = gridDim.x * blockDim.x;
	double tmp;
	for (int32_t i = blockIdx.x * blockDim.x
				+ threadIdx.x; i <= n; i += s) {
		tmp = m[i * n + i1];
		m[i * n + i1] = m[i * n + i2];
		m[i * n + i2] = tmp;
	}



}

__global__ void div_kern(double* m, int32_t pi, int32_t n) {             // делим ведущую строку на диаг элемент
	const int32_t s = gridDim.x * blockDim.x;
	const double div = m[pi * n + pi];
	for (int32_t i = blockIdx.x * blockDim.x
	+ threadIdx.x + pi + 1; i <= n; i += s) {
		m[i * n + pi] /= div;
	}
}


void sweep_up(double* m,  int32_t k, int32_t n) {				// обратный проход, находим неизвестные
	double& res = m[n * n + k];
	for (int32_t i = n - 1; i > k; i -= 1) {
		res -= m[n * n + i] * m[i * n + k] ;
	}
}
struct abs_max_t														// функция для абсолютного максимума
{																		// нужна для поиска опорной строки
	__host__ __device__
		bool operator()(double lhs, double rhs)
	{
		return fabs(lhs) < fabs(rhs);
	}
};


int main(){
	int32_t n = 0; std::cin >> n;										// вводим входные данные
	const int32_t ns = n * n + n;
	std::vector<double> matrix(ns);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			scanf("%lf", &matrix[j * n + i]);
		}
	}

	for(int32_t i = 0; i < n; ++i){
		scanf("%lf", &matrix[n * n + i]);
	}

	double* dev_matrix = nullptr;
	HANDLE_ERROR(cudaMalloc(reinterpret_cast<void**>(&dev_matrix), 		//  копируем данные в DRAM
                                            ns * sizeof(double)));
	HANDLE_ERROR(cudaMemcpy(dev_matrix, matrix.data(),
                            ns * sizeof(double),
                            cudaMemcpyHostToDevice));

	thrust::device_ptr<double> thrust_matrix_ptr = 						// thrust указатель на матрицу в GPU
                            thrust::device_pointer_cast(dev_matrix);

	float duration = 0.0, duration_local;

	cudaEvent_t start_event, stop_event;								//  готовим события старта остановки для ядер
	HANDLE_ERROR(cudaEventCreate(&start_event));
	HANDLE_ERROR(cudaEventCreate(&stop_event));

	for (int32_t i = 0; i < n; ++i) {
		auto thrust_row_ptr = thrust_matrix_ptr + i * n;
		auto thrust_max_ptr = thrust::max_element(
									thrust_row_ptr + i,
									thrust_row_ptr + n,
									abs_max_t());

		if (std::abs(*thrust_max_ptr) <= EPS) {
			HANDLE_ERROR(cudaFree(dev_matrix));
			throw std::runtime_error("error while procesing");			// опорный элемент равен 0 - вырожденный случай
		}
		int32_t pi = thrust_max_ptr - thrust_row_ptr;					// нашли опорную строку

		if (pi != i) {												// меняем строки по необходимости
			HANDLE_ERROR(cudaEventRecord(start_event));
			swp_kern <<< swp_blocks, swp_threads >>>(
				          dev_matrix, i, pi, n);
			HANDLE_ERROR(cudaGetLastError());
			HANDLE_ERROR(cudaEventRecord(stop_event));
			HANDLE_ERROR(cudaEventSynchronize(stop_event));
			HANDLE_ERROR(cudaEventElapsedTime(
                &duration_local, start_event, stop_event));
			duration += duration_local;
		}

		HANDLE_ERROR(cudaEventRecord(start_event));						// записываем коэффициенты с которыми будем вычитать
		div_kern <<< div_blocks, div_threads >>> (
                    			dev_matrix, i, n);
		HANDLE_ERROR(cudaGetLastError());
		HANDLE_ERROR(cudaEventRecord(stop_event));
		HANDLE_ERROR(cudaEventSynchronize(stop_event));
		HANDLE_ERROR(cudaEventElapsedTime(
            &duration_local, start_event, stop_event));
		duration += duration_local;


		HANDLE_ERROR(cudaEventRecord(start_event));						// вычитаем строки с ранее вычисленными коэффициентами
		sub_kern <<< sub_blocks, sub_threads >>> (
			                	dev_matrix, i, n);
		HANDLE_ERROR(cudaGetLastError());
		HANDLE_ERROR(cudaEventRecord(stop_event));
		HANDLE_ERROR(cudaEventSynchronize(stop_event));
		HANDLE_ERROR(cudaEventElapsedTime(
			&duration_local, start_event, stop_event));
		duration += duration_local;
	}

	HANDLE_ERROR(cudaEventDestroy(start_event));					   // копируем результат на RAM и освобождаем ресурсы
	HANDLE_ERROR(cudaEventDestroy(stop_event));
	HANDLE_ERROR(cudaMemcpy(matrix.data(), dev_matrix,
							ns * sizeof(double),
							cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(dev_matrix));

	for(int32_t i = n - 2; i > -1; --i){
		sweep_up(matrix.data(), i, n);                  					// делаем обратный проход
	}
	for (int j = 0; j < n; j++) {
		printf("%.10lf ", matrix[n * n + j]);
	}
	//std::cerr << "solved in: "  << duration  << "\n";
	return 0;
}

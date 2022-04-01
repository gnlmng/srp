// FFT functions (GPU)

#include "../../util/timer.h"

#include <vector>
#include <complex>
#include <cassert>
#include <cuComplex.h>

extern bool MODULE_TIMED_OUT[];
extern size_t CURRENT_MODULE;

namespace gpu_dft {
	
	namespace _cuda__global__ {
		
		__global__ void dft(const size_t size, cuDoubleComplex* buffer, cuDoubleComplex* input) {
			__shared__ cuDoubleComplex cache[1024];
			
			cache[threadIdx.x] = make_cuDoubleComplex(0.0, 0.0);
			
			size_t i = blockIdx.y * gridDim.x + blockIdx.x;
			size_t j = threadIdx.x;
			
			if (i > size || j > size) {
				return;
			}
			
			cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
			
			for (; j < size; j += 1024) {
				double angle = 2.0 * std::acos(-1.0) * (double) j * (double) i / (double) size;
				cuDoubleComplex ep = make_cuDoubleComplex(cos(angle), sin(angle));
				
				sum = cuCadd(sum, cuCmul(input[j], ep));
			}
			
			cache[threadIdx.x] = sum;
			
			__syncthreads();
			
			size_t k = threadIdx.x;
			
			for (size_t it = 1; it < 1024 && !(k & 1); it <<= 1, k >>= 1) {
				cache[threadIdx.x] = cuCadd(cache[threadIdx.x], cache[threadIdx.x + it]);
				__syncthreads();
			}
			
			if (!threadIdx.x) {
				buffer[blockIdx.y * gridDim.x + blockIdx.x] = cache[0];
			}
		}
		
	} /// namespace _cuda__global__
	
	timestamp _run(std::vector <std::complex <double>>& input) {
		if (input.size() > (size_t) 65536) {
			MODULE_TIMED_OUT[CURRENT_MODULE] = true;
			return (timestamp) 8'500'000;
		}
		
		static_assert(sizeof(cuDoubleComplex) == sizeof(std::complex <double>));
		
		const size_t size = input.size();
		
		cuDoubleComplex* in_buffer;
		cuDoubleComplex* out_buffer;
		
		cudaMallocManaged(&in_buffer, sizeof(cuDoubleComplex) * size);
		cudaMallocManaged(&out_buffer, sizeof(cuDoubleComplex) * size);
		
		cudaMemcpy(in_buffer, input.data(), sizeof(std::complex <double>) * size, cudaMemcpyHostToDevice);
		cudaMemset(out_buffer, 0, sizeof(cuDoubleComplex) * size);
		
		Timer timer;
		
		size_t block_size = 1024;
		size_t grid_count = size;
		dim3 grid_size(grid_count);
		if (grid_count > 65535) {
			size_t value = ceil(sqrt(grid_count));
			grid_size = dim3(value, value);
		}
		
		_cuda__global__::dft <<<grid_size, block_size>>> (size, out_buffer, in_buffer);
		cudaDeviceSynchronize();
		
		timestamp result_time = timer.current();
		
		cudaMemcpy(input.data(), out_buffer, sizeof(std::complex <double>) * size, cudaMemcpyDeviceToHost);
		
		return result_time;
	}
	
} /// namespace gpu_dft

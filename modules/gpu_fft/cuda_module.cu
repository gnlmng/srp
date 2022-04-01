// FFT functions (GPU)

#include "../../util/timer.h"

#include <vector>
#include <complex>
#include <cassert>
#include <cuComplex.h>

namespace gpu_fft {
	
	namespace _cuda__global__ {
		
		__global__ void rearrange(const size_t size, cuDoubleComplex* buffer, size_t* reverse) {
			size_t x = blockIdx.x * blockDim.x + threadIdx.x;
			size_t y = blockIdx.y * blockDim.y + threadIdx.y;
			size_t sx = blockDim.x * gridDim.x;
			size_t index = y * sx + x;
			
			if (index >= size) {
				return;
			}
			
			if (index < reverse[index]) {
				auto tmp = buffer[index];
				buffer[index] = buffer[reverse[index]];
				buffer[reverse[index]] = tmp;
			}
		}
		
		__global__ void fft(const size_t size, cuDoubleComplex* buffer,
		cuDoubleComplex* roots, const size_t sub_size) {
			size_t x = blockIdx.x * blockDim.x + threadIdx.x;
			size_t y = blockIdx.y * blockDim.y + threadIdx.y;
			size_t sx = blockDim.x * gridDim.x;
			size_t index = y * sx + x;
			
			if (index >= (size >> 1)) {
				return;
			}
			
			size_t block = (index / sub_size) * (sub_size << 1);
			size_t offset = index % sub_size;
			
			cuDoubleComplex delta = cuCmul(roots[sub_size + offset], buffer[sub_size + block + offset]);
			
			buffer[sub_size + block + offset] = cuCsub(buffer[block + offset], delta);
			buffer[block + offset] = cuCadd(buffer[block + offset], delta);
		}
		
	} /// namespace _cuda__global__
	
	class FFT {
		
	public:
		
		FFT(const size_t _size) :
		m_size(_size)
		{
			assert(!_size || !(_size ^ (_size & ~(_size - 1))));
			
			cudaMallocManaged(&m_roots, sizeof(cuDoubleComplex) * _size);
			cudaMallocManaged(&m_reverse, sizeof(size_t) * _size);
			cudaMallocManaged(&m_buffer, sizeof(cuDoubleComplex) * _size);
			
			m_roots[0] = make_cuDoubleComplex(1.0, 0.0);
			m_roots[1] = make_cuDoubleComplex(1.0, 0.0);
			
			for (size_t i = 2; i < _size; i <<= 1) {
				static const double pi = std::acos(-1.0);
				double inv = pi / (double) i;
				cuDoubleComplex angle = make_cuDoubleComplex(std::cos(inv), std::sin(inv));
				for (size_t j = i; j < (i << 1); j++) {
					m_roots[j] = cuCmul(m_roots[j >> 1], (j & 1) ? angle : make_cuDoubleComplex(1.0, 0.0));
				}
			}
			
			const int leading = 63 - __builtin_clzll(_size);
			for (size_t i = 0; i < _size; i++) {
				m_reverse[i] = (m_reverse[i >> 1] | ((i & 1) << leading)) >> 1;
			}
		}
		
		~FFT() {
			if (m_roots) {
				cudaFree(m_roots);
			}
			if (m_reverse) {
				cudaFree(m_reverse);
			}
			if (m_buffer) {
				cudaFree(m_buffer);
			}
		}
		
		void set_memory(std::vector <std::complex <double>>& p) {
			cudaMemcpy(m_buffer, p.data(), sizeof(std::complex <double>) * m_size, cudaMemcpyHostToDevice);
		}
		
		void get_memory(std::vector <std::complex <double>>& p) {
			cudaMemcpy(p.data(), m_buffer, sizeof(std::complex <double>) * m_size, cudaMemcpyDeviceToHost);
		}
		
		timestamp fft() {
			Timer timer;
			
			static_assert(sizeof(cuDoubleComplex) == sizeof(std::complex <double>));
			
			size_t block_size = 1024;
			size_t grid_count = (m_size + block_size - 1) / block_size;
			dim3 grid_size(grid_count);
			if (grid_count > 65535) {
				size_t value = ceil(sqrt(grid_count));
				grid_size = dim3(value, value);
			}
			
			_cuda__global__::rearrange <<<grid_size, block_size>>> (m_size, m_buffer, m_reverse);
			cudaDeviceSynchronize();
			
			grid_count = ((m_size >> 1) + block_size - 1) / block_size;
			grid_size = dim3(grid_count);
			if (grid_count > 65535) {
				size_t value = ceil(sqrt(grid_count));
				assert(value <= (size_t) 65535);
				grid_size = dim3(value, value);
			}
			
			for (size_t sub_size = 1; sub_size < m_size; sub_size <<= 1) {
				_cuda__global__::fft <<<grid_size, block_size>>> (m_size, m_buffer, m_roots, sub_size);
				cudaDeviceSynchronize();
			}
			
			return timer.current();
		}
		
	private:
		
		size_t m_size;
		
		cuDoubleComplex* m_roots;
		size_t* m_reverse;
		
		cuDoubleComplex* m_buffer;
		
	};
	
	timestamp _run(std::vector <std::complex <double>>& input) {
		FFT fft(input.size());
		fft.set_memory(input);
		timestamp result_time = fft.fft();
		fft.get_memory(input);
		return result_time;
	}
	
} /// namespace gpu_fft

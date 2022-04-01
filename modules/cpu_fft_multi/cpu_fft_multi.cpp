#include "cpu_fft_multi.h"

#include "../../util/thread_pool.h"

extern Thread_pool thread_pool;

namespace cpu_fft_multi {
	
	timestamp run(std::vector <std::complex <double>>& input) {
		const size_t size = input.size();
		
		std::vector <std::complex <double>> roots(size, 1);
		for (size_t i = 2; i < size; i <<= 1) {
			std::complex <double> angle = std::polar(1.0, std::acos(-1.0) / (double) i);
			for (size_t j = i; j < (i << 1); j++) {
				roots[j] = roots[j >> 1] * ((j & 1) ? angle : 1.0);
			}
		}
		
		const int leading = 63 - __builtin_clzll(size);
		
		std::vector <size_t> reverse(size, 0);
		for (size_t i = 0; i < size; i++) {
			reverse[i] = (reverse[i >> 1] | ((i & 1) << leading)) >> 1;
		}
		
		Timer timer;
		
		const size_t thread_count = thread_pool.size();
		
		auto rearrange = [&] (int id, const size_t l, const size_t r) -> void {
			for (size_t i = l; i < r; i++) {
				if (i < reverse[i]) {
					std::swap(input[i], input[reverse[i]]);
				}
			}
		};
		
		const size_t block_size = (size + thread_count - 1) / thread_count;
		const size_t trail_block_size = size - (block_size * (thread_count - 1));
		
		for (size_t i = 0; i < thread_count; i++) {
			size_t l = block_size * i;
			size_t r = l + (i + 1 == thread_count ? trail_block_size : block_size);
			thread_pool.push(rearrange, l, r);
		}
		thread_pool.wait_all();
		
		auto job = [&] (int, const size_t sub_size, const size_t l, const size_t r) -> void {
			for (size_t i = l; i < r; i++) {
				size_t block = (i / sub_size) * (sub_size << 1);
				size_t offset = i % sub_size;
				std::complex <double> delta = roots[sub_size + offset] * input[sub_size + block + offset];
				input[sub_size + block + offset] = input[block + offset] - delta;
				input[block + offset] += delta;
			}
		};
		
		const size_t hblock_size = ((size >> 1) + thread_count - 1) / thread_count;
		const size_t htrail_block_size = (size >> 1) - (hblock_size * (thread_count - 1));
		
		for (size_t sub_size = 1; sub_size < size; sub_size <<= 1) {
			for (size_t i = 0; i < thread_count; i++) {
				size_t l = hblock_size * i;
				size_t r = l + (i + 1 == thread_count ? htrail_block_size : hblock_size);
				thread_pool.push(job, sub_size, l, r);
			}
			thread_pool.wait_all();
		}
		
		return timer.current();
	}
	
} /// namespace cpu_fft_multi

#include "cpu_fft_single.h"

#include "../../util/thread_pool.h"

extern Thread_pool thread_pool;

namespace cpu_fft_single {
	
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
		
		for (size_t i = 0; i < size; i++) {
			if (i < reverse[i]) {
				std::swap(input[i], input[reverse[i]]);
			}
		}
		
		for (size_t i = 1; i < size; i <<= 1) {
			for (size_t j = 0; j < size; j += (i << 1)) {
				for (size_t k = 0; k < i; k++) {
					std::complex <double> delta = roots[i + k] * input[i + j + k];
					input[i + j + k] = input[j + k] - delta;
					input[j + k] += delta;
				}
			}
		}
		
		return timer.current();
	}
	
} /// namespace cpu_fft_single

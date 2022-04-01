#include "cpu_dft_single.h"

extern const timestamp MAX_ALLOWED_TIME_MICROS;
extern bool MODULE_TIMED_OUT[];
extern size_t CURRENT_MODULE;

namespace cpu_dft_single {
	
	timestamp run(std::vector <std::complex <double>>& input) {
		const size_t size = input.size();
		
		std::vector <std::complex <double>> result(size, 0);
		
		Timer timer;
		
		for (size_t i = 0; i < size && timer.current() < MAX_ALLOWED_TIME_MICROS; i++) {
			for (size_t j = 0; j < size; j++) {
				double angle = 2.0 * std::acos(-1.0) * (double) j * (double) i / (double) size;
				result[i] += input[j] * std::exp(std::complex <double> (0.0, angle));
			}
		}
		
		MODULE_TIMED_OUT[CURRENT_MODULE] = timer.current() > MAX_ALLOWED_TIME_MICROS;
		
		input = result;
		
		return timer.current();
	}
	
} /// namespace cpu_fft_single

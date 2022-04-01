#include "cpu_dft_multi.h"

#include "../../util/thread_pool.h"

extern const timestamp MAX_ALLOWED_TIME_MICROS;
extern bool MODULE_TIMED_OUT[];
extern size_t CURRENT_MODULE;

extern Thread_pool thread_pool;

namespace cpu_dft_multi {
	
	timestamp run(std::vector <std::complex <double>>& input) {
		const size_t size = input.size();
		
		std::vector <std::complex <double>> result(size, 0);
		
		const size_t thread_count = thread_pool.size();
		
		Timer timer;
		
		auto job = [&] (int, size_t i) -> void {
			for (; i < size && timer.current() < MAX_ALLOWED_TIME_MICROS; i += thread_count) {
				for (size_t j = 0; j < size; j++) {
					double angle = 2.0 * std::acos(-1.0) * (double) j * (double) i / (double) size;
					result[i] += input[j] * std::exp(std::complex <double> (0.0, angle));
				}
			}
		};
		
		for (size_t i = 0; i < thread_count; i++) {
			thread_pool.push(job, i);
		}
		thread_pool.wait_all();
		
		MODULE_TIMED_OUT[CURRENT_MODULE] = timer.current() > MAX_ALLOWED_TIME_MICROS;
		
		input = result;
		
		return timer.current();
	}
	
} /// namespace cpu_dft_multi

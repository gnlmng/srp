#include "modules/cpu_dft_single/cpu_dft_single.h"
#include "modules/cpu_dft_multi/cpu_dft_multi.h"
#include "modules/gpu_dft/gpu_dft.h"
#include "modules/cpu_fft_single/cpu_fft_single.h"
#include "modules/cpu_fft_multi/cpu_fft_multi.h"
#include "modules/gpu_fft/gpu_fft.h"

#include "util/thread_pool.h"
#include "util/random.h"
#include "util/timer.h"

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <complex>
#include <sstream>
#include <cstring>

// a set of input sizes to test with
const std::vector <size_t> SIZES = {
	1 << 8,  // 2^8
	1 << 10, // 2^10
	1 << 14, // 2^14
	1 << 16, // 2^16
	1 << 20, // 2^20
	1 << 24, // 2^24
	1 << 26, // 2^26
};

// number of iterations to run each module on each input size
constexpr const size_t ITERATIONS = 4;

// number of modules, must satisfy 0 < MODULES <= module_functions.size()
constexpr const size_t MODULES = 6;

// function pointers to the modules being tested
// each function is of the form:
// timestamp f(std::vector <std::complex <double>>&);
const std::vector <std::function <timestamp (std::vector <std::complex <double>>&)>> module_functions = {
	cpu_dft_single::run,
	cpu_dft_multi::run,
	gpu_dft::run,
	cpu_fft_single::run,
	cpu_fft_multi::run,
	gpu_fft::run
};

// string representation of each module - for printing
std::string module_strings[MODULES] = {
	"[CPU DFT single-threaded]: ",
	"[CPU DFT multi-threaded].: ",
	"[GPU DFT]................: ",
	"[CPU FFT single-threaded]: ",
	"[CPU FFT multi-threaded].: ",
	"[GPU FFT]................: "
};

// the maximum number of microseconds allowed for a module to finish
// some modules use this variable to exit prematurely if the process takes too long
// external modules will access it through:
// 'extern const timestamp MAX_ALLOWED_TIME_MICROS;'
extern timestamp constexpr const MAX_ALLOWED_TIME_MICROS = 8'500'000; // 8.5 seconds
// additionally, this table is used to mark if a module fails to finish
// if marked, the half-finished output from that module will not be considered
// external modules will access it through: 'extern bool MODULE_TIMED_OUT[];'
bool MODULE_TIMED_OUT[MODULES];
// variable allowing modules know which index to mark
// external modules will access it through: 'extern size_t CURRENT_MODULE;'
size_t CURRENT_MODULE;

// thread pool for global access
// external modules will access it through 'extern Thread_pool thread_pool;'
// default Thread_pool constructor sets number of
// software threads used equal to number of hardware threads available
Thread_pool thread_pool;

int main() {
	// timestamps for each input size for each module
	// averaged for number of iterations
	timestamp ts[SIZES.size()][MODULES];
	memset(ts, 0, sizeof(timestamp) * SIZES.size() * MODULES);
	
	// max abs error from the average output
	double max_error[SIZES.size()][MODULES];
	memset(ts, 0, sizeof(double) * SIZES.size() * MODULES);
	
	// run tests for each size
	for (size_t size_ptr = 0; size_ptr < SIZES.size(); size_ptr++) {
		const size_t size = SIZES[size_ptr];
		
		// run some number of iterations for each size
		for (size_t iteration = 0; iteration < ITERATIONS; iteration++) {
			std::cerr << "size: " << size << " (iteration " << iteration << ")" << std::endl;
			
			// reset the table telling whether a certain module has finished
			memset(MODULE_TIMED_OUT, 0, sizeof(bool) * MODULES);
			
			// query a random input
			std::vector <std::complex <double>> input = Random::query(size, thread_pool);
			
			// assign an equivilant set of input data for each module
			std::vector <std::complex <double>> module_input[MODULES];
			auto assign_module_input = [&] (int, size_t id) -> void {
				module_input[id] = input;
			};
			for (size_t module = 0; module < MODULES; module++) {
				thread_pool.push(assign_module_input, module);
			}
			thread_pool.wait_all();
			
			// run each module
			for (size_t module = 0; module < MODULES; module++) {
				//std::cerr << "  module " << module << " '" << module_strings[module] << "'" << std::endl;
				CURRENT_MODULE = module;
				ts[size_ptr][module] += module_functions[module](module_input[module]);
			}
			
			// update absolute errors
			// get the number of modules that finished
			size_t cnt_finished = 0;
			for (size_t module = 0; module < MODULES; module++) {
				cnt_finished += !MODULE_TIMED_OUT[module];
			}
			std::vector <std::complex <double>> avg(size, 0);
			for (size_t i = 0; i < size; i++) {
				// compute the average for the current index
				for (size_t module = 0; module < MODULES; module++) {
					// if the current module did not finish, do not consider its output
					if (MODULE_TIMED_OUT[module]) {
						continue;
					}
					avg[i].real(avg[i].real() + module_input[module][i].real() / (double) cnt_finished);
					avg[i].imag(avg[i].imag() + module_input[module][i].imag() / (double) cnt_finished);
				}
				// use the average to update the output of each module
				for (size_t module = 0; module < MODULES; module++) {
					// if the current module did not finish, do not update its error
					if (MODULE_TIMED_OUT[module]) {
						continue;
					}
					max_error[size_ptr][module] = std::max(max_error[size_ptr][module],
					fabs(avg[i].real() - module_input[module][i].real()));
					max_error[size_ptr][module] = std::max(max_error[size_ptr][module],
					fabs(avg[i].imag() - module_input[module][i].imag()));
				}
			}
		}
		
		// convert summed execution times to an average over the number of iterations
		for (size_t module = 0; module < MODULES; module++) {
			ts[size_ptr][module] /= ITERATIONS;
		}
	}
	
	std::cerr << "OK\n" << std::endl;
	
	/// EXECUTION TIME OUTPUT ///
	auto format_timestamp = [] (timestamp stamp) -> std::string {
		bool is_greater_8 = stamp > (timestamp) 1000 * 1000 * 8;
		stamp = std::min(stamp, (timestamp) 1000 * 1000 * 8);
		std::stringstream stream;
		stream << std::fixed << std::setprecision(3);
		stream << (double) stamp / 1000.0;
		std::string result = stream.str();
		int before_sep = 0;
		for (size_t i = 0; i < result.size() && result[i] != '.'; i++) {
			before_sep++;
		}
		result = std::string(5 - is_greater_8 - before_sep, ' ') + result + "ms";
		if (is_greater_8) {
			result = ">" + result;
		}
		return result;
	};
	std::cout << "execution time" << std::endl;
	std::cout << std::string(27, ' ');
	std::cout << "2^8     ";
	std::cout << "     2^10    ";
	std::cout << "     2^14    ";
	std::cout << "     2^16    ";
	std::cout << "     2^20    ";
	std::cout << "     2^24    ";
	std::cout << "     2^26\n";
	for (size_t module = 0; module < MODULES; module++) {
		std::cout << module_strings[module];
		for (size_t size_ptr = 0; size_ptr < SIZES.size(); size_ptr++) {
			std::cout << format_timestamp(ts[size_ptr][module]);
			if (size_ptr + 1 != SIZES.size()) {
				std::cout << ", ";
			}
		}
		std::cout << "\n";
	}
	std::cout.flush();
	
	std::cout << "\n\n";
	
	/// MAX ABS ERROR OUTPUT ///
	auto format_error = [] (double error) -> std::string {
		if (fabs(error) < 1e-50) {
			error = 0;
		}
		std::stringstream stream;
		stream << std::fixed << std::setprecision(3) << std::scientific;
		stream << error;
		std::string result = stream.str();
		return result;
	};
	std::cout << "max abs error from the average output" << std::endl;
	std::cout << std::string(27, ' ');
	std::cout << "2^8     ";
	std::cout << "   2^10    ";
	std::cout << "   2^14    ";
	std::cout << "   2^16    ";
	std::cout << "   2^20    ";
	std::cout << "   2^24    ";
	std::cout << "   2^26\n";
	for (size_t module = 0; module < MODULES; module++) {
		std::cout << module_strings[module];
		for (size_t size_ptr = 0; size_ptr < SIZES.size(); size_ptr++) {
			std::cout << format_error(max_error[size_ptr][module]);
			if (size_ptr + 1 != SIZES.size()) {
				std::cout << ", ";
			}
		}
		std::cout << "\n";
	}
	std::cout.flush();
}

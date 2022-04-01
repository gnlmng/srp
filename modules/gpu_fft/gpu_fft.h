#pragma once

#include <vector>
#include <complex>

namespace gpu_fft {
	
	extern timestamp _run(std::vector <std::complex <double>>&);
	
	timestamp run(std::vector <std::complex <double>>& input) {
		return _run(input);
	}
	
} /// namespace gpu_fft

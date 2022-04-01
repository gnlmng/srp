#pragma once

#include <vector>
#include <complex>

namespace gpu_dft {
	
	extern timestamp _run(std::vector <std::complex <double>>&);
	
	timestamp run(std::vector <std::complex <double>>& input) {
		return _run(input);
	}
	
} /// namespace gpu_dft

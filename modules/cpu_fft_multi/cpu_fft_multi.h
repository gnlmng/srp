#pragma once

#include "../../util/timer.h"

#include <vector>
#include <complex>

namespace cpu_fft_multi {
	
	timestamp run(std::vector <std::complex <double>>& input);
	
} /// namespace cpu_fft_multi

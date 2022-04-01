#pragma once

#include "../../util/timer.h"

#include <vector>
#include <complex>

namespace cpu_dft_multi {
	
	timestamp run(std::vector <std::complex <double>>& input);
	
} /// namespace cpu_dft_multi

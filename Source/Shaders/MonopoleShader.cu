/** (c) 2024 Kangrui Xue
 *
 * \file MonopoleShader.cu
 * \brief Implements Monopole acoustic shader for testing
 */

#include "Shaders.h"


#ifdef USE_CUDA

/** */
__global__ void Ker_monopole(REAL* d_vb, int global_bid, const REAL* d_B, const REAL* d_BN, 
	REAL freqHz, REAL speed, REAL C, int srate, int N_points, int N_samples, int start_step)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int k = threadIdx.y + blockDim.y * blockIdx.y;
	if (i >= N_points || k >= N_samples) { return; }

	REAL x_src = 0.; REAL y_src = 0.; REAL z_src = 0.;
	x_src = (start_step >= N_samples - 1) ? -speed + speed * (start_step + k) / (N_samples - 1) : 0.;

	REAL x = d_B[3 * i]; REAL y = d_B[3 * i + 1]; REAL z = d_B[3 * i + 2];
	REAL xn = d_BN[3 * i]; REAL yn = d_BN[3 * i + 1]; REAL zn = d_BN[3 * i + 2];

	REAL time = ((REAL) (start_step + k) / srate);

	// Eq. (D-5b), Blackstock pg. 358 
	REAL r = sqrt((x - x_src) * (x - x_src) + (y - y_src) * (y - y_src) + (z - z_src) * (z - z_src));
	REAL vb = (cos(2*M_PI * freqHz * (time - r / C)) / (r * r * 2*M_PI * freqHz)
		- sin(2*M_PI * freqHz * (time - r / C)) / (r * C)) * (xn + yn + zn);

	if (time - r / C < 0.f) { vb = 0.f; }
	d_vb[(global_bid + i) * N_samples + k] = vb;
}
#else

// Not implemented

#endif

/** */
void Monopole::compute(REAL* d_vb, int global_bid)
{
	const dim3 threads(16, 16);
	dim3 blocks((_N_points + 15) / 16, (_N_samples + 15) / 16);

	Ker_monopole<<<blocks, threads>>>(d_vb, global_bid, d_B, d_BN, _freqHz, _speed, 343.2, _srate, _N_points, _N_samples, _step);
	_step += (_N_samples - 1);

	_V1 = _V2;
	_V2.rowwise() += Eigen::RowVector3<REAL>({ _speed, 0., 0. });

	if (_step > 0 && _speed == 0.) { _changed = false; }
}
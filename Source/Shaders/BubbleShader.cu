/** (c) 2024 Kangrui Xue
 *
 * \file BubbleShader.cu
 * \brief Implements Bubbles acoustic shader for bubble-based water sound
 * 
 * References:
 *   [Xue et al. 2023] Improved Water Sound Synthesis using Coupled Bubbles
 *	 [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains 
 */

#include "Shaders.h"


__device__ static REAL _abs(REAL val) { return (val < 0.f) ? -val : val; }

/** \brief Constructs (N_points x N_bubs) "bubble-to-boundary" transfer matrix (Eq. 11 from [Xue et al. 2024]) */
__global__ void Ker_bubToBoundary(REAL *d_bubToBoundary, const REAL *d_B, const REAL *d_BN, int N_points,
	const REAL *d_bubData, int N_bubs)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i >= N_points || j >= N_bubs) { return; }

	REAL x = d_B[3*i]; REAL y = d_B[3*i + 1]; REAL z = d_B[3*i + 2];
	REAL xn = d_BN[3*i]; REAL yn = d_BN[3*i + 1]; REAL zn = d_BN[3*i + 2];

	REAL r = d_bubData[4*j];
	REAL xbub = d_bubData[4*j + 1]; REAL ybub = d_bubData[4*j + 2]; REAL zbub = d_bubData[4*j + 3];

	REAL dist = sqrt((x - xbub) * (x - xbub) + (y - ybub) * (y - ybub) + (z - zbub) * (z - zbub));
	REAL A_bub = ((x - xbub) * _abs(xn) + (y - ybub) * _abs(yn) + (z - zbub) * _abs(zn)) / (dist * 4.f * M_PI);
	
	// Special handling for d < r case (avoid d = 0 blowing up, etc.)
	if (A_bub < 0.f) { A_bub = 0.f; }
	else if (dist < r) { A_bub *= -3.f * dist * dist / (r * r * r * r) + 4.f * dist / (r * r * r); }
	else { A_bub *= 1.f / (dist * dist); }

	d_bubToBoundary[N_points * j + i] = A_bub;
}


/** \brief Multiplies "bubble-to-boundary" transfer matrix with bubble volume velocities to compute vb, */
// TODO: port to cublas 
// This is a brute-force implementation, but it's by no means the bottleneck
__global__ void Ker_bubMatmul(REAL* d_vb, int global_bid, const REAL* d_bubToBoundary, const REAL* d_bubVels, const REAL* d_flux,
	int N_points, int N_bubs, int N_samples, int start, int stop)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int k = threadIdx.y + blockDim.y * blockIdx.y;
	if (i >= N_points || k >= stop - start) { return; }

	REAL vb = 0.f;
	for (int j = 0; j < N_bubs; j++)
	{
		REAL flux = (d_flux[j] > 1e-6) ? d_flux[j] : 1.f;
		vb += d_bubToBoundary[N_points * j + i] * d_bubVels[N_bubs * k + j] / flux;
	}
	d_vb[(global_bid + i) * N_samples + (start + k)] = vb;
}


/** */
void Bubbles::_readFluidMesh()
{
	int lookahead = _N_samples - 1;
	
	// Bunch of filename logic
	std::string filetag = "tmpMesh-00";
	filetag += std::to_string((_step + lookahead + int(_srate * _ts)) / _srate) + ".";
	
	int dec = (_step + lookahead + int(_srate * _ts)) % _srate / (_srate / 1000);
	std::stringstream ss; ss << std::setw(3) << std::setfill('0') << dec;
	filetag += ss.str() + "000";

	std::cout << "Reading mesh from " << _fluidMeshDir + filetag << std::endl;
	_V1 = _V2;
	_changed = _readObj(_fluidMeshDir + filetag + ".obj", _V2, _F);  // true if _readObj() successful, false otherwise
}


/** */
// TODO: probably will not work on batch lengths that are non-integer multiples of 1 ms
void Bubbles::compute(REAL* d_vb, int global_bid)
{
	int minibatchsize = std::min((_srate / 1000), (_N_samples - 1));  // 1 ms
	int N_MINIBATCHES = ((_N_samples - 1) + (minibatchsize - 1)) / minibatchsize;  // ceiling


	// Determine active oscillators during current batch
	_activeOscIDs.clear();
	double t1 = _step * _dt + _ts; double t2 = (_step + _N_samples - 1) * _dt + _ts;
	for (int osID = 0; osID < _solver.oscillators().size(); osID++)
	{
		const FluidSound::Oscillator<double>& osc = _solver.oscillators()[osID];

		if (osc.startTime <= t1 && osc.endTime > t1) // coupled oscillators
			_activeOscIDs.push_back(osID);
		else if (osc.endTime <= t1 && !osc.is_dead()) // uncoupled oscillators
			_activeOscIDs.push_back(osID);
		else if (t1 < osc.startTime && osc.startTime < t2) // osc will be added during current batch
			_activeOscIDs.push_back(osID);
	}

	// Initialize a bunch of buffers
	_bubData.resize(_activeOscIDs.size() * 4);
	_bubVels = Eigen::MatrixX<REAL>::Zero(_activeOscIDs.size(), (minibatchsize + 1));

	const dim3 threads(16, 16);
	dim3 blocks((_N_points + 15) / 16, (_activeOscIDs.size() + 15) / 16);
	dim3 matmul_blocks((_N_points + 15) / 16, ((minibatchsize + 1) + 15) / 16);

	if (_activeOscIDs.size() > _max_N_osc) 
	{
		_max_N_osc = _activeOscIDs.size();
		cudaFree(d_bubData); cudaMalloc((void **) &d_bubData, _bubData.size() * sizeof(REAL));
		cudaFree(d_bubVels); cudaMalloc((void **) &d_bubVels, _bubVels.size() * sizeof(REAL));
		cudaFree(d_flux); cudaMalloc((void **) &d_flux, _activeOscIDs.size() * sizeof(REAL));
	}
	cudaFree(d_bubToBoundary); cudaMalloc((void **) &d_bubToBoundary, _N_points * _activeOscIDs.size() * sizeof(REAL));

	Eigen::VectorX<REAL> ones = Eigen::VectorX<REAL>::Ones(_N_points);
	cudaFree(d_ones); cudaMalloc((void**) &d_ones, _N_points * sizeof(REAL));
	cudaMemcpy(d_ones, ones.data(), _N_points * sizeof(REAL), cudaMemcpyHostToDevice);
	
	for (int m = 0; m < N_MINIBATCHES; m++)
	{
		// Construct bubble-to-boundary projection matrix (N_points x N_bubs) 
		for (int j = 0; j < _activeOscIDs.size(); j++)
		{
			int osID = _activeOscIDs[j];
			FluidSound::Oscillator<double>& osc = _solver.oscillators()[osID];

			if ((_step * _dt + _ts) > osc.endTime && osc.is_dead()) { continue; }
			Eigen::ArrayXd Data = osc.interp(_step * _dt + _ts);
			_bubData[4 * j] = Data[0];      // r
			_bubData[4 * j + 1] = Data[2];  // xbub
			_bubData[4 * j + 2] = Data[3];  // ybub
			_bubData[4 * j + 3] = Data[4];  // zbub
		}
		cudaMemcpy(d_bubData, _bubData.data(), _bubData.size() * sizeof(REAL), cudaMemcpyHostToDevice);
		Ker_bubToBoundary<<<blocks, threads>>>(d_bubToBoundary, d_B, d_BN, _N_points, d_bubData, _activeOscIDs.size());
		cudaDeviceSynchronize();


		const REAL alpha = _dx * _dx;  // cell face area
		const REAL beta = 0.f;
#ifdef USE_FLOAT64
		cublasDgemv(_handle, CUBLAS_OP_T, _N_points, (int)_activeOscIDs.size(), &alpha, d_bubToBoundary, _N_points, d_ones, 1, &beta, d_flux, 1);
#else
		cublasSgemv(_handle, CUBLAS_OP_T, _N_points, (int)_activeOscIDs.size(), &alpha, d_bubToBoundary, _N_points, d_ones, 1, &beta, d_flux, 1);
#endif
		// Package oscillator velocities (over minibatch) into matrix (N_bubs x T_steps)
		int start = m * minibatchsize; int stop = (m + 1) * minibatchsize;
		for (int k = start; k < stop; k++)
		{
			for (int j = 0; j < _activeOscIDs.size(); j++)
			{
				int osID = _activeOscIDs[j];
				FluidSound::Oscillator<double>& osc = _solver.oscillators()[osID];

				_bubVels(j, k % minibatchsize) = osc.state[1];
			}
			double out = _solver.step();
			_step += 1;
		}

		// Extend last minibatch for interpolation
		if (m == N_MINIBATCHES - 1)
		{
			for (int j = 0; j < _activeOscIDs.size(); j++)
			{
				int osID = _activeOscIDs[j];
				FluidSound::Oscillator<double>& osc = _solver.oscillators()[osID];

				_bubVels(j, minibatchsize) = osc.state[1];
			}
			stop += 1;
		}
		// Matrix multiply to get boundary velocity
		cudaMemcpy(d_bubVels, _bubVels.data(), _bubVels.size() * sizeof(REAL), cudaMemcpyHostToDevice);
		Ker_bubMatmul<<<matmul_blocks, threads>>>(d_vb, global_bid, d_bubToBoundary, d_bubVels, d_flux,
			_N_points, _activeOscIDs.size(), _N_samples, start, stop);
	}

	_readFluidMesh();  // read next mesh
}
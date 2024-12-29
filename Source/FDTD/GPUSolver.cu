/** (c) 2024 Kangrui Xue
 *
 * \file GPUSolver.cu
 * 
 * Thanks to Paulius Micivekius and Guillaume Thomas Collignon at NVIDIA for GPU optimization assistance.
 * 
 * References:
 *   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains.
 */

#include "GPUSolver.h"


/** \brief Pressure update kernel (Eq. 8a from [Xue et al. 2024]) */
__global__ void Ker_stepPressure(REAL* __restrict__ d_p, REAL* __restrict__ d_px, REAL* __restrict__ d_py, REAL* __restrict__ d_pz,
	const REAL* __restrict__ d_vx, const REAL* __restrict__ d_vy, const REAL* __restrict__ d_vz, REAL* __restrict__ d_beta, const int* __restrict__ d_cell, 
	REAL tb, const REAL* __restrict__ d_pmlN, const REAL* __restrict__ d_pmlD, REAL RHO_CC_dt, REAL inv_dx, int Nx, int Ny, int Nz)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	if (i >= Nx || j >= Ny || k >= Nz) { return; }
	int cid = CID(i, j, k, Nx, Ny);

	// Compute distance to closest domain boundary
	int min_xdist = min(i, Nx - 1 - i);
	int min_ydist = min(j, Ny - 1 - j);
	int min_zdist = min(k, Nz - 1 - k);

	// Compute individual components of divergence
	REAL vx = d_vx[cid];
	REAL vx_L = (i > 0) ? d_vx[CID(i - 1, j, k, Nx, Ny)] : 0.f;  // LEFT
	REAL vy = d_vy[cid];
	REAL vy_D = (j > 0) ? d_vy[CID(i, j - 1, k, Nx, Ny)] : 0.f;  // DOWN
	REAL vz = d_vz[cid];
	REAL vz_F = (k > 0) ? d_vz[CID(i, j, k - 1, Nx, Ny)] : 0.f;  // FRONT

	REAL divx = (vx - vx_L) * inv_dx;
	REAL divy = (vy - vy_D) * inv_dx;
	REAL divz = (vz - vz_F) * inv_dx;

	// If inside PML, split pressure update
	if (min_xdist < PML_WIDTH || min_ydist < PML_WIDTH || min_zdist < PML_WIDTH)  
	{
		REAL px = d_px[cid]; REAL py = d_py[cid]; REAL pz = d_pz[cid];

		REAL pmlNx = d_pmlN[min_xdist]; REAL pmlDx = d_pmlD[min_xdist];
		REAL pmlNy = d_pmlN[min_ydist]; REAL pmlDy = d_pmlD[min_ydist];
		REAL pmlNz = d_pmlN[min_zdist]; REAL pmlDz = d_pmlD[min_zdist];

		px = (pmlNx * px - RHO_CC_dt * divx) * pmlDx;
		py = (pmlNy * py - RHO_CC_dt * divy) * pmlDy;
		pz = (pmlNz * pz - RHO_CC_dt * divz) * pmlDz;

		d_px[cid] = px; d_py[cid] = py; d_pz[cid] = pz;
		d_p[cid] = px + py + pz;
	}
	else  // Otherwise, regular pressure update
	{
		d_p[cid] = d_p[cid] - RHO_CC_dt * (divx + divy + divz);
	}

	// Update beta (cubic smoothstep function)
	if (d_cell[cid] > 0 && d_beta[cid] < 1.f)
	{
		d_beta[cid] = 3.f * tb * tb - 2.f * tb * tb * tb;
	}
	else if (d_cell[cid] == 0 && d_beta[cid] > 0.f)
	{
		REAL _tb = 1.f - tb;
		d_beta[cid] = 3.f * _tb * _tb - 2.f * _tb * _tb * _tb;
	}
}


/** \brief Blended velocity update kernel: boundary conditions term (Eq. 8b from [Xue et al. 2024]) */
__global__ void Ker_applyShader(REAL* __restrict__ d_vx, REAL* __restrict__ d_vy, REAL* __restrict__ d_vz, const REAL* __restrict__ d_beta, 
	const REAL* __restrict__ d_shaderData, const int* d_shaderMap, int N_shader_samples, int N_shader_points, 
	REAL inv_RHO_dt, int Nx, int Ny, int Nz, REAL ss)
{
	int global_bid = threadIdx.x + blockDim.x * blockIdx.x;
	if (global_bid >= N_shader_points) { return; }

	// Decode shader map
	bool isForce = d_shaderMap[global_bid] < 0;		// hack to encode isForce via the sign of d_shaderMap[cid]
	int dir = (!isForce) ? d_shaderMap[global_bid] % 3 : (-d_shaderMap[global_bid]) % 3;	// direction
	int cid = (!isForce) ? d_shaderMap[global_bid] / 3 : (-d_shaderMap[global_bid]) / 3;	// cell index

	int i = cid % Nx;
	int j = (cid / Nx) % Ny;
	int k = (cid / Nx) / Ny;

	// Decode shader values
	REAL frac = (global_bid * N_shader_samples) + ss;	// fractional index in d_shaderData memory (to allow for interpolation)
	int floor = int(frac); int ceil = floor + 1;
	REAL val = (ceil - frac) * d_shaderData[floor] + (frac - floor) * d_shaderData[ceil];  // interpolate

	// Update velocity
	if (dir == X_DIR)
	{ 
		REAL betax = max(d_beta[cid], d_beta[CID(i + 1, j, k, Nx, Ny)]);
		if (!isForce) { d_vx[cid] += betax * (val - d_vx[cid]); }
		else { d_vx[cid] += (1.f - betax) * val * inv_RHO_dt; }
	}
	else if (dir == Y_DIR)
	{ 
		REAL betay = max(d_beta[cid], d_beta[CID(i, j + 1, k, Nx, Ny)]);
		if (!isForce) { d_vy[cid] += betay * (val - d_vy[cid]); }
		else { d_vy[cid] += (1.f - betay) * val * inv_RHO_dt; }
	}
	else if (dir == Z_DIR) 
	{ 
		REAL betaz = max(d_beta[cid], d_beta[CID(i, j, k + 1, Nx, Ny)]);
		if (!isForce) { d_vz[cid] += betaz * (val - d_vz[cid]); }
		else { d_vz[cid] += (1.f - betaz) * val * inv_RHO_dt; }
	}
}


/** \brief Blended velocity update kernel: pressure gradient term (Eq. 8b from [Xue et al. 2024]) */
__global__ void Ker_stepVelocity(REAL* __restrict__ d_vx, REAL* __restrict__ d_vy, REAL* __restrict__ d_vz, const REAL* __restrict__ d_p,
	REAL* __restrict__ d_beta, const REAL* __restrict__ d_pmlN, const REAL* __restrict__ d_pmlD, REAL inv_RHO_dt, REAL inv_dx, int Nx, int Ny, int Nz)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	int k = threadIdx.z + blockDim.z * blockIdx.z;

	if (i >= Nx || j >= Ny || k >= Nz) { return; }
	int cid = CID(i, j, k, Nx, Ny);

	// Distance to closest domain boundary
	int min_xdist = min(i + 1, Nx - i - 1);
	int min_ydist = min(j + 1, Ny - j - 1);
	int min_zdist = min(k + 1, Nz - k - 1);

	REAL p = d_p[cid];
	REAL p_R = (i < Nx - 1) ? d_p[CID(i + 1, j, k, Nx, Ny)] : 0.f;  // RIGHT
	REAL p_U = (j < Ny - 1) ? d_p[CID(i, j + 1, k, Nx, Ny)] : 0.f;  // UP
	REAL p_B = (k < Nz - 1) ? d_p[CID(i, j, k + 1, Nx, Ny)] : 0.f;  // BACK

	REAL beta = d_beta[cid];
	REAL beta_R = (i < Nx - 1) ? d_beta[CID(i + 1, j, k, Nx, Ny)] : 1.f;  // RIGHT
	REAL beta_U = (j < Ny - 1) ? d_beta[CID(i, j + 1, k, Nx, Ny)] : 1.f;  // UP
	REAL beta_B = (k < Nz - 1) ? d_beta[CID(i, j, k + 1, Nx, Ny)] : 1.f;  // BACK

	REAL gradx = (p_R - p) * inv_dx; 
	REAL grady = (p_U - p) * inv_dx;
	REAL gradz = (p_B - p) * inv_dx;

	REAL betax = max(beta, beta_R);
	REAL betay = max(beta, beta_U);
	REAL betaz = max(beta, beta_B);

	// pmlN = 1. and pmlD = 1. / (1. + damping) if outside PML layer
	REAL pmlNx = d_pmlN[min_xdist]; REAL pmlDx = d_pmlD[min_xdist];
	REAL pmlNy = d_pmlN[min_ydist]; REAL pmlDy = d_pmlD[min_ydist];
	REAL pmlNz = d_pmlN[min_zdist]; REAL pmlDz = d_pmlD[min_zdist];

	d_vx[cid] = (pmlNx * d_vx[cid] - (1.f - betax) * inv_RHO_dt * gradx) * pmlDx;
	d_vy[cid] = (pmlNy * d_vy[cid] - (1.f - betay) * inv_RHO_dt * grady) * pmlDy;
	d_vz[cid] = (pmlNz * d_vz[cid] - (1.f - betaz) * inv_RHO_dt * gradz) * pmlDz;
}


/** */
void FDTDSolver::runFDTD()
{
	const dim3 FDTD_threads(8, 8, 8);
	dim3 FDTD_blocks((_simParams.Nx + 7) / 8, (_simParams.Ny + 7) / 8, (_simParams.Nz + 7) / 8);

	const dim3 shader_threads(32);
	dim3 shader_blocks((_N_shader_points + 31) / 32);


	for (int s = 0; s < _N_FDTD_samples; s++)
	{
		REAL t = (s + 1 == _N_FDTD_samples) ? 1. : (REAL) (s + 1) / _N_FDTD_samples;  // normalized blending time (0, 1] 
		REAL ss = t * (_N_shader_samples - 1);  // shader sample index (fractional to support interpolation)
		switch (_simParams.scheme)
		{
			case SCHEME::NO_BLEND: t = (s + 1 == _N_FDTD_samples) ? 1. : 0.; break;
			default: break;
		}
		Ker_stepPressure<<<FDTD_blocks, FDTD_threads>>>(d_p, d_px, d_py, d_pz, d_vx, d_vy, d_vz, d_beta, d_cell, t,
			d_pmlNp, d_pmlDp, _RHO_CC_dt, _inv_dx, _simParams.Nx, _simParams.Ny, _simParams.Nz);
		Ker_applyShader<<<shader_blocks, shader_threads>>>(d_vx, d_vy, d_vz, d_beta, d_shaderData, d_shaderMap,
			_N_shader_samples, _N_shader_points, _inv_RHO_dt, _simParams.Nx, _simParams.Ny, _simParams.Nz, ss);
		Ker_stepVelocity<<<FDTD_blocks, FDTD_threads>>>(d_vx, d_vy, d_vz, d_p, d_beta, 
			d_pmlNv, d_pmlDv, _inv_RHO_dt, _inv_dx, _simParams.Nx, _simParams.Ny, _simParams.Nz);

		//if (_step % 882 == 0) { logZSlice("logs/log-" + std::to_string(_step / 882) + ".bin"); }
		for (GPUListener* listener : _listeners) { listener->addSample(d_p, s); }
		_step += 1;
	}
	for (GPUListener* listener : _listeners) { listener->write(); }
}


/** */
FDTDSolver::FDTDSolver(const SimParams& params) : _simParams(params)
{
	_gridSize = _simParams.Nx * _simParams.Ny * _simParams.Nz;
	_N_FDTD_samples = _simParams.FDTD_srate / _simParams.blendrate;
	_N_shader_samples = _simParams.shader_srate / _simParams.blendrate + 1;

	cudaMalloc((void**)&d_p, _gridSize * sizeof(REAL));
	cudaMalloc((void**)&d_vx, _gridSize * sizeof(REAL));
	cudaMalloc((void**)&d_vy, _gridSize * sizeof(REAL));
	cudaMalloc((void**)&d_vz, _gridSize * sizeof(REAL));

	cudaMalloc((void**)&d_beta, _gridSize * sizeof(REAL));
	cudaMalloc((void**)&d_cell, _gridSize * sizeof(int));

	_RHO_CC_dt = _simParams.RHO * _simParams.C * _simParams.C * _simParams.dt;
	_inv_dx = 1. / _simParams.dx;
	_inv_RHO_dt = 1. / _simParams.RHO * _simParams.dt;

	_damping = _simParams.damping;
	_initializePML();
}


/** */
void FDTDSolver::addListener(const std::string& format, const std::vector<REAL>& listenerP, const std::string& output_name)
{
	int i = (listenerP[0] / _simParams.dx) + (_simParams.Nx - 1) / 2.;
	int j = (listenerP[1] / _simParams.dx) + (_simParams.Ny - 1) / 2.;
	int k = (listenerP[2] / _simParams.dx) + (_simParams.Nz - 1) / 2.;

	if (format == "Mono") 
	{ 
		_listeners.push_back(new MonoListener(_cid(i, j, k), _N_FDTD_samples, output_name));
	}
	else { throw std::runtime_error("Invalid listener format: " + format); }
}


/** */
void FDTDSolver::logZSlice(const std::string& filetag)
{
	int offset = _cid(0, 0, _simParams.Nz / 2);  // z-slice
	std::vector<REAL> buf(_simParams.Nx * _simParams.Ny);
	std::ofstream logfile(filetag, std::ofstream::binary);

	// Log pressure
	cudaMemcpy(buf.data(), d_p + offset, _simParams.Nx * _simParams.Ny * sizeof(REAL), cudaMemcpyDeviceToHost);
	logfile.write((char*)buf.data(), buf.size() * sizeof(REAL));

	// Log beta
	cudaMemcpy(buf.data(), d_beta + offset, _simParams.Nx * _simParams.Ny * sizeof(REAL), cudaMemcpyDeviceToHost);
	logfile.write((char*)buf.data(), buf.size() * sizeof(REAL));

	logfile.close();
}


/** */
// TODO: switch to better PML (e.g., Convolutional-PML)
void FDTDSolver::_initializePML()
{
	// Allocate split pressure field
	cudaMalloc((void**)&d_px, _gridSize * sizeof(REAL));
	cudaMalloc((void**)&d_py, _gridSize * sizeof(REAL));
	cudaMalloc((void**)&d_pz, _gridSize * sizeof(REAL));

	int max_halfGridLength = (std::max({ _simParams.Nx, _simParams.Ny, _simParams.Nz }) + 1) / 2;

	// Allocate velocity PML weights: numerator and denominator
	std::vector<REAL> pmlNv((max_halfGridLength + 1), 1.), pmlDv((max_halfGridLength + 1), 1. / (1. + _damping));
	cudaMalloc((void**)&d_pmlNv, (max_halfGridLength + 1) * sizeof(REAL));
	cudaMalloc((void**)&d_pmlDv, (max_halfGridLength + 1) * sizeof(REAL));

	// Allocate pressure PML weights: numerator and denominator
	std::vector<REAL> pmlNp(max_halfGridLength, 1.), pmlDp(max_halfGridLength, 1.);
	cudaMalloc((void**)&d_pmlNp, max_halfGridLength * sizeof(REAL));
	cudaMalloc((void**)&d_pmlDp, max_halfGridLength * sizeof(REAL));

	// Precompute PML weights
	REAL pmlWeight;
	for (int dist = 0; dist < PML_WIDTH; dist++)
	{
		pmlWeight = ((REAL) PML_WIDTH - dist) / PML_WIDTH;   // velocity
		pmlWeight = 0.5 * pmlWeight * pmlWeight;
		pmlNv[dist] = (1. - pmlWeight); pmlDv[dist] = 1. / (1. + pmlWeight);

		pmlWeight = ((REAL) PML_WIDTH - dist - 0.5) / PML_WIDTH;  // pressure
		pmlWeight = 0.5 * pmlWeight * pmlWeight;
		pmlNp[dist] = (1. - pmlWeight); pmlDp[dist] = 1. / (1. + pmlWeight);
	}
	cudaMemcpy(d_pmlNv, pmlNv.data(), (max_halfGridLength + 1) * sizeof(REAL), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlDv, pmlDv.data(), (max_halfGridLength + 1) * sizeof(REAL), cudaMemcpyHostToDevice);

	cudaMemcpy(d_pmlNp, pmlNp.data(), max_halfGridLength * sizeof(REAL), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pmlDp, pmlDp.data(), max_halfGridLength * sizeof(REAL), cudaMemcpyHostToDevice);
}


/** */
FDTDSolver::~FDTDSolver()
{
	cudaFree(d_p); cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
	cudaFree(d_beta); cudaFree(d_cell);

	cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
	cudaFree(d_pmlNp); cudaFree(d_pmlDp); cudaFree(d_pmlNv); cudaFree(d_pmlDv);

	cudaFree(d_shaderMap); cudaFree(d_shaderData);

	for (GPUListener* listener : _listeners) { delete listener; }
}
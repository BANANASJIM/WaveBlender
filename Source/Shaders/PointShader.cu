/** (c) 2024 Kangrui Xue
 *
 * \file PointShader.cu
 * \brief Implements Point force acoustic shader for impulse-related "clicks"
 * 
 * TODO: cleanup needed
 */

#include "Shaders.h"


__device__ REAL _abs(REAL val) { return (val < 0.f) ? -val : val; }

/** \brief Constructs (N_points x N_impulses) "impulse-to-boundary" (particle-to-grid) transfer matrix */
__global__ void Ker_impulseToBoundary(REAL* d_impulseToBoundary, const REAL* d_B, const REAL* d_BN, int N_points,
	const REAL* d_impulseData, int N_impulses, REAL dx)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;
	if (i >= N_points || j >= N_impulses) { return; }

	REAL x = d_B[3*i]; REAL y = d_B[3*i + 1]; REAL z = d_B[3*i + 2];
	REAL xn = d_BN[3*i]; REAL yn = d_BN[3*i + 1]; REAL zn = d_BN[3*i + 2];

	REAL xpt = d_impulseData[6*j]; REAL ypt = d_impulseData[6*j + 1]; REAL zpt = d_impulseData[6*j + 2];
	REAL xpt_dir = d_impulseData[6*j + 3]; REAL ypt_dir = d_impulseData[6*j + 4]; REAL zpt_dir = d_impulseData[6*j + 5];

	REAL distx = _abs(x - xpt); REAL disty = _abs(y - ypt); REAL distz = _abs(z - zpt);

	REAL weight = 0.f;
	if (distx < dx && disty < dx && distz < dx) { weight = (1.f - distx / dx) * (1.f - disty / dx) * (1.f - distz / dx); }

	if (xn != 0.f) { weight *= xpt_dir; }
	else if (yn != 0.f) { weight *= ypt_dir; }
	else if (zn != 0.f) { weight *= zpt_dir; }

	d_impulseToBoundary[N_points * j + i] = weight;
}


/** */
// TODO: port to cublas
// This is a brute-force implementation, but it's by no means the bottleneck
__global__ void Ker_impulseMatmul(REAL* d_force, int global_bid, const REAL* d_impulseToBoundary, const REAL* d_impulseVels, int N_points, 
	int N_impulses, int N_samples)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int k = threadIdx.y + blockDim.y * blockIdx.y;
	if (i >= N_points || k >= N_samples) { return; }

	REAL vb = 0.f;
	for (int j = 0; j < N_impulses; j++)
	{
		vb += d_impulseToBoundary[N_points * j + i] * d_impulseVels[N_impulses * k + j];
	}
	d_force[(global_bid + i) * N_samples + k] = vb;
}


/** */
void Point::_readImpulses(const std::string &filename)
{
	std::ifstream inFile(filename);
	std::string line;
	std::getline(inFile, line);  // Skip first line
	while (!line.empty() && inFile.good())
	{
		std::getline(inFile, line);
		std::istringstream is(line);

		double time, impulse, tau, dvx, dvy, dvz, Px, Py, Pz; int id;
		is >> time >> impulse >> tau >> dvx >> dvy >> dvz >> Px >> Py >> Pz >> id;

		_times.push_back(time);
		_tau.push_back(tau);
		_DV.push_back(Eigen::Vector3<REAL>(dvx, dvy, dvz));
		_P.push_back(Eigen::Vector3<REAL>(Px, Py, Pz));
	}
	std::cout << "No. impulses: " << _times.size() << std::endl;
}


/** */
void Point::_getActiveImpulseList()
{
	_activeImpulseIDs.clear();
	for (int k = 0; k < _N_samples; k++)
	{
		double time = (_step + k) * _dt;
		for (int idx = 0; idx < _times.size(); idx++)
		{
			if (time >= _times[idx] && time < _times[idx] + _tau[idx])
				_activeImpulseIDs.insert(idx);
		}
	}
	_V2 = Eigen::MatrixX<REAL>::Zero(_activeImpulseIDs.size(), 3);
	int r = 0; for (int idx : _activeImpulseIDs)
	{
		_V2.row(r) = _P[idx]; r += 1;
	}
	_changed = true;
}


/** */
void Point::compute(REAL* d_force, int global_bid)
{
	const dim3 threads(16, 16);
	dim3 blocks((_N_points + 15) / 16, (_activeImpulseIDs.size() + 15) / 16);
	dim3 matmul_blocks((_N_points + 15) / 16, (_N_samples + 15) / 16);


	_impulseData.resize(_activeImpulseIDs.size() * 6);
	_impulseVels = Eigen::MatrixX<REAL>::Zero(_activeImpulseIDs.size(), _N_samples);
	int j = 0; for (int idx : _activeImpulseIDs)
	{
		const Eigen::Vector3<REAL>& J = _DV[idx];
		const Eigen::Vector3<REAL>& pos = _P[idx];
		const Eigen::Vector3<REAL> dir = J / J.norm();
		for (int k = 0; k < _N_samples; k++)
		{
			double time = (_step + k) * _dt;
			if (time < _times[idx] || time > _times[idx] + _tau[idx]) { continue; }

			REAL S = std::sin(M_PI * (time - _times[idx]) / _tau[idx]);

			// translational acceleration
			Eigen::Vector3<REAL> accelPulse = M_PI / (2. * _tau[idx]) * J * S;
			_impulseVels(j, k) = accelPulse.norm();  // TODO: pre-compute
		}
		_impulseData[6*j] = pos[0]; _impulseData[6*j + 1] = pos[1]; _impulseData[6*j + 2] = pos[2];  // positions
		_impulseData[6*j + 3] = dir[0]; _impulseData[6*j + 4] = dir[1]; _impulseData[6*j + 5] = dir[2];  // directions

		j += 1;
	}
	_step += (_N_samples - 1);

	cudaFree(d_impulseData); cudaMalloc((void **) &d_impulseData, _activeImpulseIDs.size() * 6 * sizeof(REAL));
	cudaFree(d_impulseToBoundary); cudaMalloc((void **) &d_impulseToBoundary, _activeImpulseIDs.size() * _N_points * sizeof(REAL));
	cudaFree(d_impulseVels); cudaMalloc((void **) &d_impulseVels, _impulseVels.size() * sizeof(REAL));


	cudaMemcpy(d_impulseData, _impulseData.data(), _impulseData.size() * sizeof(REAL), cudaMemcpyHostToDevice);
	Ker_impulseToBoundary<<<blocks, threads>>>(d_impulseToBoundary, d_B, d_BN, _N_points, d_impulseData, _activeImpulseIDs.size(), _dx);

	cudaMemcpy(d_impulseVels, _impulseVels.data(), _impulseVels.size() * sizeof(REAL), cudaMemcpyHostToDevice);
	Ker_impulseMatmul<<<matmul_blocks, threads>>>(d_force, global_bid, d_impulseToBoundary, d_impulseVels, _N_points, _activeImpulseIDs.size(), _N_samples);

	
	_getActiveImpulseList();
}
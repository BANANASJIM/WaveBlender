/** (c) 2024 Kangrui Xue
 *
 * \file ModalShader.cu
 * \brief Implements Modal acoustic shader
 * 
 * TODO: cleanup needed
 */

#include "Shaders.h"


/** */
// TODO: port to cublas
// This is a brute-force implementation, but it's by no means the bottleneck
__global__ void Ker_modeMatmul(REAL* d_vb, int global_bid, const REAL* d_modeToBoundary1, const REAL* d_modeToBoundary2,
	const REAL* d_modeVels, const REAL* d_accelNoise, int N_points, int N_modes, int N_samples)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int k = threadIdx.y + blockDim.y * blockIdx.y;
	if (i >= N_points || k >= N_samples) { return; }

	REAL vb = 0.f;
	REAL alpha = ((REAL) k) / (N_samples - 1.f);
	for (int j = 0; j < N_modes; j++)
	{
		vb += (1.f - alpha) * d_modeToBoundary1[N_points * j + i] * d_modeVels[N_modes * k + j];
		vb += alpha * d_modeToBoundary2[N_points * j + i] * d_modeVels[N_modes * k + j];
	}
	d_vb[(global_bid + i) * N_samples + k] = vb + d_accelNoise[N_points * k + i];
}


/** */
void Modal::_setModeToBoundary(Eigen::MatrixX<REAL>& modeToBoundary)
{
	Eigen::Quaterniond quat1(_rotation1[0], _rotation1[1], _rotation1[2], _rotation1[3]);
	Eigen::Quaterniond quat2(_rotation2[0], _rotation2[1], _rotation2[2], _rotation2[3]);

	const Eigen::MatrixXd& eigenVectorsNormal = _solver._eigenVectorsNormal;
	const Eigen::MatrixXd& normals = _solver._normals;


	// Compute interpolated rotation + translation
	double alpha = ((_step * _dt + _ts) - _t1) / (_t2 - _t1);
	Eigen::Quaternion<REAL> quat = quat1.slerp(alpha, quat2).cast<REAL>();
	Eigen::RowVector3<REAL> translation = ((1. - alpha) * _translation1 + alpha * _translation2).cast<REAL>();


	Eigen::VectorXi I; Eigen::MatrixX<REAL> W;  // Closest triangle indices + barycentric weights

	// For closest point query, first transform boundary face positions into rest frame
	Eigen::Matrix3<REAL> invRot = quat.inverse().toRotationMatrix();
	Eigen::Matrix<REAL, Eigen::Dynamic, 3, Eigen::RowMajor> B = _B.rowwise() - translation;
	B = B.eval() * invRot.transpose();
	
	_closestPoint(I, W, B, _V0);


	// Set entries of modeToBoundary matrix (assumes already allocated)
	for (int bid = 0; bid < _N_points; bid++)
	{
		Eigen::Vector3<REAL> bn = _BN.row(bid);
		for (int vert = 0; vert < 3; vert++)
		{
			REAL weight = W(bid, vert);		// barycentric weights
			int vertexID = _F(I[bid], vert);

			Eigen::Vector3<REAL> normal = normals.row(vertexID).cast<REAL>();
			weight *= bn.dot(quat * normal);

			modeToBoundary.row(bid) += weight * eigenVectorsNormal.row(vertexID).cast<REAL>();
		}
	}
}


/** */
void Modal::compute(REAL* d_vb, int global_bid)
{
	int N_modes = _solver._qDot_c_plus.size();  // TODO: cleanup ModalSound submodule
	Eigen::Quaterniond quat1(_rotation1[0], _rotation1[1], _rotation1[2], _rotation1[3]);
	Eigen::Quaterniond quat2(_rotation2[0], _rotation2[1], _rotation2[2], _rotation2[3]);


	// Build modeToBoundary matrix at batch start (we will eventually interpolate in between batch start and end)
	_modeToBoundary1 = Eigen::MatrixX<REAL>::Zero(_N_points, N_modes);
	_setModeToBoundary(_modeToBoundary1);

	cudaFree(d_modeToBoundary1);
	cudaMalloc((void**)&d_modeToBoundary1, _modeToBoundary1.size() * sizeof(REAL));
	cudaMemcpy(d_modeToBoundary1, _modeToBoundary1.data(), _modeToBoundary1.size() * sizeof(REAL), cudaMemcpyHostToDevice);


	// Initialize surface velocity buffers (we will set the data as we go)
	_modeVels = Eigen::MatrixX<REAL>::Zero(N_modes, _N_samples);
	_accelNoise = Eigen::MatrixX<REAL>::Zero(_N_points, _N_samples);
	
	Eigen::Vector3<double> accelPulse1;		// current acceleration
	Eigen::Vector3<double> accelPulse2 = Eigen::Vector3<double>::Zero();	// next acceleration
	

	// Timestep modal vibrations
	for (int k = 0; k < _N_samples; k++)
	{
		_modeVels.col(k) = _solver._qDot_c_plus.cast<REAL>();

		// -------------------- Acceleration noise -------------------- //
		double time = _step * _dt + _ts;
		
		double alpha = (time - _t1) / (_t2 - _t1);
		Eigen::Quaterniond quat = quat1.slerp(alpha, quat2);

		std::vector<ModalSound::ImpactRecord> impactRecords;		// TODO: remove nesting
		ModalSound::ImpulseSeries& impulseSeries = _solver._impulseSeries;
		
		for (int idx = 0; idx < impulseSeries._impulses.size(); ++idx)
		{
			const ModalSound::ImpactRecord& record = impulseSeries._impulses[idx];
			if (time >= record.timestamp && time < record.timestamp + record.supportLength)
				impactRecords.push_back(record);
		}

		accelPulse1 = accelPulse2;
		accelPulse2 = Eigen::Vector3<double>::Zero();

		for (const ModalSound::ImpactRecord& impulse : impactRecords)
		{
			if (impulse.supportLength < 1e-12) { continue; }

			const Eigen::Vector3d& J = impulse.impactVector;
			const Eigen::Vector3d r = impulse.impactPosition - _solver._centerOfMass;

			double S = 0.;
			if (time <= impulse.timestamp + impulse.supportLength && time >= impulse.timestamp)
				S = std::sin(M_PI * (time - impulse.timestamp) / impulse.supportLength);

			// Translational acceleration
			accelPulse2 += J * (M_PI * S / (2. * impulse.supportLength * _solver._mass));

			// Rotational acceleration (Eq. 13) from [Chadwick et al. 2012]
			Eigen::Matrix3d& I_Inv = _solver._I_Inv;
			const Eigen::Vector3d alpha = (I_Inv * r.cross(J)) * (M_PI * S) / (2. * impulse.supportLength);
			accelPulse2 += alpha.cross(r);
		}

		accelPulse2 = quat * accelPulse2.eval();  // rotate vector from rest frame to animation frame

		for (int bid = 0; bid < _N_points; bid++)	// set normal velocities
		{
			if (_BN(bid, 0) != 0.f) _accelNoise(bid, k) = _accelSum[0];
			else if (_BN(bid, 1) != 0.f) _accelNoise(bid, k) = _accelSum[1];
			else if (_BN(bid, 2) != 0.f) _accelNoise(bid, k) = _accelSum[2];
		}
		// ------------------------------------------------------------------------------- //

		_solver.step(_step * _dt + _ts);
		if (k >= _N_samples - 1) { break; }

		_accelSum += (accelPulse1 + accelPulse2) / 2.f * _dt;	// trapezoidal rule integrator
		_step += 1;
	}

	cudaFree(d_modeVels);
	cudaMalloc((void**) &d_modeVels, _modeVels.size() * sizeof(REAL));
	cudaMemcpy(d_modeVels, _modeVels.data(), _modeVels.size() * sizeof(REAL), cudaMemcpyHostToDevice);

	cudaFree(d_accelNoise);
	cudaMalloc((void**) &d_accelNoise, _accelNoise.size() * sizeof(REAL));
	cudaMemcpy(d_accelNoise, _accelNoise.data(), _accelNoise.size() * sizeof(REAL), cudaMemcpyHostToDevice);
	

	// Read next animation data in order to build modeToBoundary matrix at batch end
	if (_animFileStream.is_open()) { _readAnimation(); }

	_modeToBoundary2 = Eigen::MatrixX<REAL>::Zero(_N_points, N_modes);
	_setModeToBoundary(_modeToBoundary2);

	cudaFree(d_modeToBoundary2);
	cudaMalloc((void**)&d_modeToBoundary2, _modeToBoundary2.size() * sizeof(REAL));
	cudaMemcpy(d_modeToBoundary2, _modeToBoundary2.data(), _modeToBoundary2.size() * sizeof(REAL), cudaMemcpyHostToDevice);

	
	// Multiply modeToBoundary matrix with modal velocities on GPU
	const dim3 threads(16, 16);
	dim3 matmul_blocks((_N_points + 15) / 16, (_N_samples + 15) / 16);
	
	Ker_modeMatmul<<<matmul_blocks, threads>>>(d_vb, global_bid, d_modeToBoundary1, d_modeToBoundary2, d_modeVels, d_accelNoise, _N_points, N_modes, _N_samples);
}
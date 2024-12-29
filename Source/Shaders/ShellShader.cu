/** (c) 2024 Kangrui Xue
 *
 * \file ShellShader.cu
 * \brief Implements Shell acoustic shader (right now, it's just loading precomputed simulation data from disk)
 */

#include "Shaders.h"


/**  */
__global__ void Ker_shellMatcpy(REAL* d_vb, int global_bid, const REAL* d_batchVels, int N_points, int N_samples)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int k = threadIdx.y + blockDim.y * blockIdx.y;
	if (i >= N_points || k >= N_samples) { return; }

	d_vb[(global_bid + i) * N_samples + k] = d_batchVels[N_samples * i + k];
}


/** */
void Shell::_readShellAnimation()
{
	_vertDisplace = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Zero(_V0.rows(), 3);
	int lookahead = _N_samples - 1;

	int frame1 = _step + lookahead;  // For now, we assume shader srate = 44.1 kHz and start time = 0 (TODO: generalize)
	std::stringstream ss; ss << std::setw(9) << std::setfill('0') << frame1;

	int N_U = 0, N_C = 0;

	// UNCONSTRAINED
	std::string file = _shellAnimDir + ss.str() + ".displacement";
	std::ifstream inFile1(file, std::ios::in | std::ios::binary);
	if (inFile1.good())
	{
		inFile1.read((char*) &N_U, sizeof(int));
		inFile1.read((char*) _vertDisplace.data(), N_U * sizeof(double));


		// CONSTRAINED
		file = _shellAnimDir + ss.str() + ".constraint_displacement";
		std::ifstream inFile2(file, std::ios::in | std::ios::binary);

		inFile2.read((char*) &N_C, sizeof(int));
		inFile2.read((char*) _vertDisplace.data() + N_U, N_C * sizeof(double));
	}

	assert(_V2.rows() == (N_U + N_C) / 3);
	_V1 = _V2; _tree.init(_V1, _F);
	for (int r = 0; r < _V2.rows(); r++)
	{
		int internalID = _vertMap[r];
		if (internalID < N_U / 3) { _V2.row(r) = _V0.row(r) + _vertDisplace.row(internalID).cast<REAL>(); }
		else if (internalID < (N_U + N_C) / 3) { _V2.row(r) = _V0.row(r) + _vertDisplace.row(internalID).cast<REAL>(); }
	}
	_changed = true;
}


/** */
void Shell::_readVertexMap(std::string mapFile)
{
	std::ifstream inFile(mapFile);
	if (inFile.good())   // if vertex_map file exists, read in
	{
		std::string line;
		std::getline(inFile, line);

		int r = 0;
		while (!line.empty() && inFile.good())
		{
			std::istringstream is(line);
			std::getline(inFile, line);
			int orig2intern, intern2orig; is >> orig2intern >> intern2orig;
			_vertMap.insert(std::pair<int, int>(r, orig2intern));
			r += 1;
		}
	}
	else   // otherwise, use identity mapping
	{
		for (int r = 0; r < _V0.rows(); r++)
		{
			_vertMap.insert(std::pair<int, int>(r, r));
		}
	}
}


/** */
void Shell::compute(REAL* d_vb, int global_bid)
{
	Eigen::VectorXi I1; Eigen::MatrixX<REAL> W1;  // Closest triangle
	_closestPoint(I1, W1, _B, _V1);

	
	// Load next batch of vertex velocities
	if (_step % (_megabatchsize - 1) == 0)
	{
		Eigen::RowVectorXd initVertVels = _vertVels.row(_vertVels.rows() - 1);

		_vertVels = Eigen::MatrixXd::Zero(_megabatchsize, _V2.rows() * 3);
		for (int k = 0; k < _megabatchsize; k++)
		{
			//int frame = int((_step + k) * _dt * 44100. + 0.001);
			int frame = _step + k;  // For now, assume shader srate = 44.1 kHz  (TODO: generalize)
			std::stringstream ss; ss << std::setw(9) << std::setfill('0') << frame;
			
			int N_U = 0, N_C = 0;

			// UNCONSTRAINED
			std::string file = _shellAccelDir + ss.str() + ".wsacc";
			std::ifstream inFile1(file, std::ios::in | std::ios::binary);
			_vertAccel0 = _vertAccel1;
			if (inFile1.good())
			{
				inFile1.read((char*) &N_U, sizeof(int));
				inFile1.read((char*) _vertAccel1.data(), N_U * sizeof(double));

				// CONSTRAINED
				file = _shellAccelDir + ss.str() + ".constraint_acceleration";
				std::ifstream inFile2(file, std::ios::in | std::ios::binary);

				inFile2.read((char*) &N_C, sizeof(int));
				inFile2.read((char*) _vertAccel1.data() + N_U, N_C * sizeof(double));
			}
			else
			{
				std::cout << "MISSING Shell accel. file: " << file << std::endl;
				_vertAccel1.setZero();
			}
			if (k == 0) { _vertVels.row(0) = initVertVels; }
			else { _vertVels.row(k) = _vertVels.row(k - 1) + (_vertAccel0 + _vertAccel1) / 2. * _dt; }
		}
	}


	_batchVels = Eigen::MatrixX<REAL>::Zero(_N_points, _N_samples);
	int offset = _step % (_megabatchsize - 1);
	for (int k = 0; k < _N_samples; k++)
	{
		for (int bid = 0; bid < _N_points; bid++)
		{
			Eigen::Vector3<REAL> BN = _BN.row(bid);

			REAL alpha = (REAL) k / (_N_samples - 1);
			for (int vert = 0; vert < 3; vert++)
			{
				REAL weight1 = 1. / 3.; //(1.f - alpha) * W1(bid, vert);
				int internalID1 = _vertMap[_F(I1[bid], vert)];

				Eigen::Vector3d vertVel1 = _vertVels.row(offset + k).segment<3>(3 * internalID1);  // TODO: for now, assumes shader srate = 44.1 kHz
				_batchVels(bid, k) += weight1 * BN.cwiseAbs().dot(vertVel1.cast<REAL>()); 
			}
		}
		if (k < (_N_samples - 1)) { _step += 1; }
	}

	// Copy boundary velocity to d_vb (on device) directly
	cudaMemcpy(d_vb + global_bid * _N_samples, _batchVels.data(), _batchVels.size() * sizeof(REAL), cudaMemcpyHostToDevice);
	
	
	_readShellAnimation();  // read next set of vertex displacements
}
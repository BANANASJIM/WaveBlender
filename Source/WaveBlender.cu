/** (c) 2024 Kangrui Xue 
 * 
 * \file WaveBlender.cu
 * 
 * References:
 *   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains
 */

#include "WaveBlender.h"
#include "tribox.h"


/**  */
bool WaveBlender::runBatch()
{
	if (_step >= (_simParams.tf - _simParams.ts) * _simParams.FDTD_srate) { return false; }

	_cell1 = _cell2;

	// 1. Check if Objects have moved; if not, we can skip step 2
	std::set<int> unchanged_oids; bool anyChanged = false;
	for (int oid = 0; oid < _objects.size(); oid++)
	{
		if (!_objects[oid]->changed()) { unchanged_oids.insert(oid); }
		else { anyChanged = true; }
	}
	// 2. Rasterize Objects and setup shaders
	if (anyChanged)
	{
		for (int cid = 0; cid < _cell2.size(); cid++)  // Only clear cells of Objects that moved
		{
			if (unchanged_oids.count(_cell2[cid] - 1) == 0) { _cell2[cid] = 0; }
		}
		_rasterize(unchanged_oids);

		_detectCavities();  // Runtime cavity detection

		if (_step == 0)  // Initialize d_beta
		{
			std::vector<REAL> buf(_gridSize);
			for (int cid = 0; cid < buf.size(); cid++) { buf[cid] = (_cell2[cid] > 0) ? 1. : 0.; }
			cudaMemcpy(d_beta, buf.data(), buf.size() * sizeof(REAL), cudaMemcpyHostToDevice);
		}
		_setupShaders();
	}
	// 3. Compute shader values at all sampled positions and times
	// (for now, assumes point sources are specified at the end of config file, TODO: generalize)
	int global_bid = 0;
	for (std::shared_ptr<Object> obj : _objects)
	{
		obj->compute(d_shaderData, global_bid);
		if (obj->hasShader()) { global_bid += obj->N_points(); }
	}

	// 4. Additional "per-batch overhead"
	_freshCellPressure();  // Fresh cell extrapolation
	_freshCellVelocity();

	_shaderReInit();  // Shader velocity re-initialization

	// 5. Run FDTD update
	runFDTD();

	return true;
}


/** Conservative CPU rasterizer based on triangle-box overlap test */
void WaveBlender::_rasterize(const std::set<int>& unchanged_oids, bool logRaster)
{
	for (int oid = 0; oid < _objects.size(); oid++)
	{
		if (!_objects[oid]->changed()) { continue; }

		const Eigen::Vector3<REAL>& offset = _offsets[oid];
		const Eigen::MatrixX<REAL>& V = _objects[oid]->V();
		const Eigen::MatrixXi& F = _objects[oid]->F();

		// Point source rasterization special handling
		// (for now, assumes point sources are specified at the end of config file, TODO: generalize)
		if (_objects[oid]->shaderClass() == SHADER_CLASS::POINT)
		{
			for (int r = 0; r < V.rows(); r++)
			{
				Eigen::Vector3<REAL> pt = V.row(r); pt += offset;

				const int i = int(pt[0] / _simParams.dx + _simParams.Nx / 2.);
				const int j = int(pt[1] / _simParams.dx + _simParams.Ny / 2.);
				const int k = int(pt[2] / _simParams.dx + _simParams.Nz / 2.);

				int cid = _cid(i, j, k);
				if (_cell2[cid] == 0) { _cell2[cid] = oid + 1; }

				// Neighboring cells needed for point-to-grid
				int neighbor_cids[6] = { _cid(i - 1, j, k), _cid(i + 1, j, k),	  // left, right
										 _cid(i, j - 1, k), _cid(i, j + 1, k),	  // down, up
										 _cid(i, j, k - 1), _cid(i, j, k + 1) };  // back, front
				for (int n = 0; n < 6; n += 2)
				{
					if (_cell2[neighbor_cids[n]] == 0) { _cell2[neighbor_cids[n]] = oid + 1; }
				}
				// (we brute-force consider all 6 neighbors for good measure, though only 3 are needed)
			}
			continue;
		}
		// Density rasterization special handling
		else if (_objects[oid]->shaderClass() == SHADER_CLASS::DENSITY)
		{
			for (int r = 0; r < V.rows(); r++)
			{
				Eigen::Vector3<REAL> pt = V.row(r); pt += offset;

				const int i = int(pt[0] / _simParams.dx + _simParams.Nx / 2.);
				const int j = int(pt[1] / _simParams.dx + _simParams.Ny / 2.);
				const int k = int(pt[2] / _simParams.dx + _simParams.Nz / 2.);

				int cid = _cid(i, j, k);
				int density = F(r, 0);  // HACK to encode density in Object's faces

				const int THRESHOLD = 97;
				if (density >= THRESHOLD  && unchanged_oids.count(_cell2[cid] - 1) == 0) { _cell2[cid] = oid + 1; }
			}
			continue;
		}
		// General triangle mesh rasterization
		for (int r = 0; r < F.rows(); r++)
		{
			const Eigen::Vector3i face = F.row(r);
			Eigen::Matrix3<REAL> TriV;
			TriV.row(0) = V.row(face[0]);
			TriV.row(1) = V.row(face[1]);
			TriV.row(2) = V.row(face[2]);

			Eigen::Vector3<REAL> min = TriV.colwise().minCoeff(); min += offset;
			Eigen::Vector3<REAL> max = TriV.colwise().maxCoeff(); max += offset;

			const int min_i = int(min[0] / _simParams.dx + _simParams.Nx / 2.);
			const int max_i = int(max[0] / _simParams.dx + _simParams.Nx / 2.);

			const int min_j = int(min[1] / _simParams.dx + _simParams.Ny / 2.);
			const int max_j = int(max[1] / _simParams.dx + _simParams.Ny / 2.);

			const int min_k = int(min[2] / _simParams.dx + _simParams.Nz / 2.);
			const int max_k = int(max[2] / _simParams.dx + _simParams.Nz / 2.);

			for (int i = min_i; i <= max_i; i++)
			{
				for (int j = min_j; j <= max_j; j++)
				{
					for (int k = min_k; k <= max_k; k++)
					{
						Eigen::Vector3<REAL> P = _Pos(i, j, k) - offset;

						REAL boxcenter[3] = { P[0], P[1], P[2] };
						REAL boxhalfsize[3] = { _simParams.dx / 2., _simParams.dx / 2., _simParams.dx / 2. };
						REAL triverts[3][3] = { {TriV(0, 0), TriV(0, 1), TriV(0, 2)},
											  {  TriV(1, 0), TriV(1, 1), TriV(1, 2)},
											  {  TriV(2, 0), TriV(2, 1), TriV(2, 2)} };
						if (triBoxOverlap(boxcenter, boxhalfsize, triverts))
						{
							if (i < 0 || i >= _simParams.Nx || j < 0 || j >= _simParams.Ny || k < 0 || k >= _simParams.Nz) { continue; }

							int cid = _cid(i, j, k);
							if (unchanged_oids.count(_cell2[cid] - 1) == 0) { _cell2[cid] = oid + 1; }
						}
					}
				}
			}
		} // loop over triangles
	} // loop over objects

	// Log rasterization
	if (logRaster)
	{
		std::ofstream raster_file("logs/raster-" + std::to_string(_step) + ".txt");
		raster_file << _simParams.Nx << " " << _simParams.Ny << " " << _simParams.Nz << " " << _simParams.dx << std::endl;  // first line is metadata

		for (int cid = 0; cid < _cell2.size(); cid++)
		{
			if (_cell2[cid] != 0 && _cell2[cid] != CAVITY_INTERIOR)
			{
				int i = cid % _simParams.Nx;
				int j = (cid / _simParams.Nx) % _simParams.Ny;
				int k = (cid / _simParams.Nx) / _simParams.Ny;

				raster_file << _cell2[cid] << " " << i << " " << j << " " << k << std::endl;
			}
		}
	}
}


/** Section 6.2.3 from [Xue et al. 2024] */
void WaveBlender::_detectCavities()
{
	std::vector<bool> connected(_cell2.size());
	std::queue<int> Q; REAL tmp = 0.;

	// Determine bounding box
	int min_i = _simParams.Nx - 1, max_i = 0;
	int min_j = _simParams.Ny - 1, max_j = 0;
	int min_k = _simParams.Nz - 1, max_k = 0;
	for (int oid = 0; oid < _objects.size(); oid++)
	{
		if (_objects[oid]->shaderClass() == SHADER_CLASS::POINT) { continue; }

		const Eigen::MatrixX<REAL>& V = _objects[oid]->V();
		const Eigen::Vector3<REAL>& offset = _offsets[oid];

		Eigen::Vector3<REAL> min = V.colwise().minCoeff(); min += offset;
		Eigen::Vector3<REAL> max = V.colwise().maxCoeff(); max += offset;

		tmp = int(min[0] / _simParams.dx + _simParams.Nx / 2.); if (tmp < min_i) { min_i = tmp; }
		tmp = int(max[0] / _simParams.dx + _simParams.Nx / 2.); if (tmp > max_i) { max_i = tmp; }

		tmp = int(min[1] / _simParams.dx + _simParams.Ny / 2.); if (tmp < min_j) { min_j = tmp; }
		tmp = int(max[1] / _simParams.dx + _simParams.Ny / 2.); if (tmp > max_j) { max_j = tmp; }

		tmp = int(min[2] / _simParams.dx + _simParams.Nz / 2.); if (tmp < min_k) { min_k = tmp; }
		tmp = int(max[2] / _simParams.dx + _simParams.Nz / 2.); if (tmp > max_k) { max_k = tmp; }
	}
	min_i -= 1; min_j -= 1; min_k -= 1;
	max_i += 1; max_j += 1; max_k += 1;

	// Flood fill to detect connected components
	Q.push(_cid(max_i, max_j, max_k));
	while (!Q.empty())
	{
		int cid = Q.front(); Q.pop();

		int i = cid % _simParams.Nx;
		int j = (cid / _simParams.Nx) % _simParams.Ny;
		int k = (cid / _simParams.Nx) / _simParams.Ny;

		// Enforce that the PML contains no objects to save time on flood fill
		if (i < min_i || i > max_i || j < min_j || j > max_j || k < min_k || k > max_k) { continue; }
		if ((!connected[cid] && _cell2[cid] == 0) ||
			(!connected[cid] && _objects[_cell2[cid] - 1]->shaderClass() == SHADER_CLASS::POINT))
		{
			connected[cid] = true;
			Q.push(_cid(i - 1, j, k)); Q.push(_cid(i + 1, j, k));  // left, right
			Q.push(_cid(i, j - 1, k)); Q.push(_cid(i, j + 1, k));  // down, up
			Q.push(_cid(i, j, k - 1)); Q.push(_cid(i, j, k + 1));  // back, front
		}
	}
	// Fill in cavities
	for (int i = min_i; i <= max_i; i++)
	{
		for (int j = min_j; j <= max_j; j++)
		{
			for (int k = min_k; k <= max_k; k++)
			{
				int cid = _cid(i, j, k);
				if ((!connected[cid] && _cell2[cid] == 0) ||
					(!connected[cid] && _objects[_cell2[cid] - 1]->shaderClass() == SHADER_CLASS::POINT)) 
				{ 
					_cell2[cid] = CAVITY_INTERIOR;
				}
			}
		}
	}
}


/** Section 6.1 from [Xue et al. 2024] */
void WaveBlender::_setupShaders()
{
	std::vector<int> shaderMap;
	std::set<int> used_vxids, used_vyids, used_vzids;
	for (int oid = 0; oid < _objects.size(); oid++)
	{
		if (!_objects[oid]->hasShader() || _objects[oid]->shaderClass() == SHADER_CLASS::POINT) { continue; }

		int bid = 0;
		std::vector<Eigen::Vector3<REAL>> Bvec;
		std::vector<Eigen::Vector3<REAL>> BNvec;
		for (int cid = 0; cid < _cell2.size(); cid++)
		{
			if (_cell1[cid] != oid + 1 && _cell2[cid] != oid + 1) { continue; }

			int i = cid % _simParams.Nx;
			int j = (cid / _simParams.Nx) % _simParams.Ny;
			int k = (cid / _simParams.Nx) / _simParams.Ny;

			if (i == 0 || i >= _simParams.Nx - 1 || j == 0 || j >= _simParams.Ny - 1 || k == 0 || k >= _simParams.Nz - 1) { continue; }

			Eigen::Vector3<REAL> P = _Pos(i, j, k) - _offsets[oid];    // world-space to object-space

			int neighbor_cids[6] = { _cid(i - 1, j, k), _cid(i + 1, j, k),	  // left, right
									 _cid(i, j - 1, k), _cid(i, j + 1, k),	  // down, up
									 _cid(i, j, k - 1), _cid(i, j, k + 1) };  // back, front
			for (int n = 0; n < 6; n++)
			{
				if ((_cell1[cid] == oid + 1 && _cell1[neighbor_cids[n]] == 0) ||
					(_cell2[cid] == oid + 1 && _cell2[neighbor_cids[n]] == 0))
				{
					int D = (n % 2 == 0); REAL sign = (n % 2 == 0) ? -1. : 1.;

					if (n < 2 && used_vxids.count(_cid(i - D, j, k)) > 0) { continue; }
					if (n >= 2 && n < 4 && used_vyids.count(_cid(i, j - D, k)) > 0) { continue; }
					if (n >= 4 && n < 6 && used_vzids.count(_cid(i, j, k - D)) > 0) { continue; }

					if (n < 2)
					{
						shaderMap.push_back(3 * _cid(i - D, j, k) + 0);
						BNvec.push_back(sign * Eigen::Vector3<REAL>({ 1., 0., 0. }));
						used_vxids.insert(_cid(i - D, j, k));
					}
					else if (n < 4)
					{
						shaderMap.push_back(3 * _cid(i, j - D, k) + 1);
						BNvec.push_back(sign * Eigen::Vector3<REAL>({ 0., 1., 0. }));
						used_vyids.insert(_cid(i, j - D, k));
					}
					else if (n < 6)
					{
						shaderMap.push_back(3 * _cid(i, j, k - D) + 2);
						BNvec.push_back(sign * Eigen::Vector3<REAL>({ 0., 0., 1. }));
						used_vzids.insert(_cid(i, j, k - D));
					}
					Bvec.push_back(P.cast<REAL>() + (0.5 * _simParams.dx) * BNvec[bid]);
					bid += 1;
				}
			}
		} // loop over cells
		
		Eigen::MatrixX<REAL> B(Bvec.size(), 3);
		Eigen::MatrixX<REAL> BN(BNvec.size(), 3);
		for (int bid = 0; bid < Bvec.size(); bid++)
		{
			B.row(bid) = Bvec[bid]; BN.row(bid) = BNvec[bid];
		}
		_objects[oid]->setSamplePoints(B, BN);

	} // loop over objects

	used_vxids.clear(), used_vyids.clear(), used_vzids.clear();
	for (int oid = 0; oid < _objects.size(); oid++)
	{
		if (_objects[oid]->shaderClass() != SHADER_CLASS::POINT) { continue; }

		int bid = 0;
		std::vector<Eigen::Vector3<REAL>> Bvec;
		std::vector<Eigen::Vector3<REAL>> BNvec;
		for (int cid = 0; cid < _cell2.size(); cid++)
		{
			if (_cell2[cid] != oid + 1) { continue; }
			_cell2[cid] = 0;  // We don't want to rasterize point sources, so set cell back to 0

			int i = cid % _simParams.Nx;
			int j = (cid / _simParams.Nx) % _simParams.Ny;
			int k = (cid / _simParams.Nx) / _simParams.Ny;

			if (i == 0 || i >= _simParams.Nx - 1 || j == 0 || j >= _simParams.Ny - 1 || k == 0 || k >= _simParams.Nz - 1) { continue; }

			Eigen::Vector3<REAL> P = _Pos(i, j, k) - _offsets[oid];    // world-space to object-space

			int neighbor_cids[6] = { _cid(i - 1, j, k), _cid(i + 1, j, k),	  // left, right
									 _cid(i, j - 1, k), _cid(i, j + 1, k),	  // down, up
									 _cid(i, j, k - 1), _cid(i, j, k + 1) };  // front, back
			for (int n = 0; n < 6; n++)
			{
				{
					int D = (n % 2 == 0); REAL sign = (n % 2 == 0) ? -1. : 1.;

					if (n < 2 && used_vxids.count(_cid(i - D, j, k)) > 0) { continue; }
					if (n >= 2 && n < 4 && used_vyids.count(_cid(i, j - D, k)) > 0) { continue; }
					if (n >= 4 && n < 6 && used_vzids.count(_cid(i, j, k - D)) > 0) { continue; }

					if (n < 2)
					{
						shaderMap.push_back(-3 * _cid(i - D, j, k) - 0);
						BNvec.push_back(sign * Eigen::Vector3<REAL>({ 1., 0., 0. }));
						used_vxids.insert(_cid(i - D, j, k));
					}
					else if (n < 4)
					{
						shaderMap.push_back(-3 * _cid(i, j - D, k) - 1);
						BNvec.push_back(sign * Eigen::Vector3<REAL>({ 0., 1., 0. }));
						used_vyids.insert(_cid(i, j - D, k));
					}
					else if (n < 6)
					{
						shaderMap.push_back(-3 * _cid(i, j, k - D) - 2);
						BNvec.push_back(sign * Eigen::Vector3<REAL>({ 0., 0., 1. }));
						used_vzids.insert(_cid(i, j, k - D));
					}
					Bvec.push_back(P.cast<REAL>() + (0.5 * _simParams.dx) * BNvec[bid]);
					bid += 1;
				}
			}
		} // loop over cells
		
		Eigen::MatrixX<REAL> B(Bvec.size(), 3);
		Eigen::MatrixX<REAL> BN(BNvec.size(), 3);
		for (int bid = 0; bid < Bvec.size(); bid++)
		{
			B.row(bid) = Bvec[bid]; BN.row(bid) = BNvec[bid];
		}
		_objects[oid]->setSamplePoints(B, BN);

	} // loop over objects


	// Set both raster and shader memory
	cudaMemcpy(d_cell, _cell2.data(), _cell2.size() * sizeof(int), cudaMemcpyHostToDevice);

	_N_shader_points = shaderMap.size();
	if (_N_shader_points > _max_N_shader_points)  // Reallocate 
	{
		_max_N_shader_points = _N_shader_points;

		cudaFree(d_shaderData);
		cudaMalloc((void**)&d_shaderData, _N_shader_points * _N_shader_samples * sizeof(REAL));

		cudaFree(d_shaderMap);
		cudaMalloc((void**)&d_shaderMap, _N_shader_points * sizeof(int));
	}
	cudaMemset(d_shaderData, 0, _N_shader_points * _N_shader_samples * sizeof(REAL));
	cudaMemcpy(d_shaderMap, shaderMap.data(), _N_shader_points * sizeof(int), cudaMemcpyHostToDevice);
}


/** Computes a_n, required to enforce Neumann boundary conditions for fresh cell */
__global__ void Ker_prepareFreshCell(REAL* __restrict__ d_ax, REAL* __restrict__ d_ay, REAL* __restrict__ d_az,
	REAL* __restrict__ d_shaderData, const int* d_shaderMap, int N_shader_samples, int N_shader_points, REAL inv_dt, REAL incr)
{
	int global_bid = threadIdx.x + blockDim.x * blockIdx.x;
	if (global_bid >= N_shader_points) { return; }

	// Decode shader map
	if (d_shaderMap[global_bid] < 0) { return; }  // Skip if isForce
	int dir = d_shaderMap[global_bid] % 3;
	int cid = d_shaderMap[global_bid] / 3;

	// Decode shader values
	int floor = (global_bid * N_shader_samples);
	REAL vb0 = d_shaderData[floor];
	REAL vb0_dt = (1.f - incr) * d_shaderData[floor] + (incr) * d_shaderData[floor + 1];

	REAL a_0 = (vb0_dt - vb0) * inv_dt;
	if (dir == X_DIR) { d_ax[cid] = a_0; }
	else if (dir == Y_DIR) { d_ay[cid] = a_0; }
	else if (dir == Z_DIR) { d_az[cid] = a_0; }
}

/** Eq. 13 from [Xue et al. 2024] */
__global__ void Ker_freshCellPressure(REAL* __restrict__ d_p, REAL* __restrict__ d_ax, REAL* __restrict__ d_ay, REAL* __restrict__ d_az, 
	const REAL* __restrict__ d_beta, int Nx, int Ny, int Nz, REAL rho_dx)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x + PML_WIDTH;
	int j = threadIdx.y + blockDim.y * blockIdx.y + PML_WIDTH;
	int k = threadIdx.z + blockDim.z * blockIdx.z + PML_WIDTH;

	if (i >= Nx - 1 - PML_WIDTH || j >= Ny - 1 - PML_WIDTH || k >= Nz - 1 - PML_WIDTH) { return; }
	int cid = CID(i, j, k, Nx, Ny);

	int neighbor_cids[6] = { CID(i - 1, j, k, Nx, Ny), CID(i + 1, j, k, Nx, Ny),	// left, right
							 CID(i, j - 1, k, Nx, Ny), CID(i, j + 1, k, Nx, Ny),	// down, up
							 CID(i, j, k - 1, Nx, Ny), CID(i, j, k + 1, Nx, Ny) };	// front, back
	REAL p_fresh = 0.f; int N_solid = 0, N_air = 0;
	for (int n = 0; n < 6; n++)
	{
		if (d_beta[neighbor_cids[n]] >= 1.f) { N_solid += 1; }  // neighboring cell is solid
		else  // neighboring cell is air
		{ 
			int dir = n / 3; REAL sign = (n % 2 == 0) ? 1.f : -1.f;
			int vid = (n % 2 == 0) ? neighbor_cids[n] : cid;

			if (dir == X_DIR) { p_fresh += d_p[neighbor_cids[n]] - sign * rho_dx * d_ax[vid]; }
			else if (dir == Y_DIR) { p_fresh += d_p[neighbor_cids[n]] - sign * rho_dx * d_ay[vid]; }
			else if (dir == Z_DIR) { p_fresh += d_p[neighbor_cids[n]] - sign * rho_dx * d_az[vid]; }
			N_air += 1;
		}
	}
	if (N_air > 0) { p_fresh /= N_air; }

	d_p[cid] = (1.f - d_beta[cid]) * d_p[cid] + d_beta[cid] * p_fresh;
}

/** Sets interior velocities to 0 (and clears acceleration a_n buffers) */
__global__ void Ker_clearSolid(REAL* __restrict__ d_vx, REAL* __restrict__ d_vy, REAL* __restrict__ d_vz, 
	REAL* __restrict__ d_ax, REAL* __restrict__ d_ay, REAL* __restrict__ d_az, const REAL* __restrict__ d_beta, int Nx, int Ny, int Nz)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x + PML_WIDTH;
	int j = threadIdx.y + blockDim.y * blockIdx.y + PML_WIDTH;
	int k = threadIdx.z + blockDim.z * blockIdx.z + PML_WIDTH;

	if (i >= Nx - 1 - PML_WIDTH || j >= Ny - 1 - PML_WIDTH || k >= Nz - 1 - PML_WIDTH) { return; }
	int cid = CID(i, j, k, Nx, Ny);

	if (d_beta[cid] >= 1.f)
	{
		if (d_beta[CID(i + 1, j, k, Nx, Ny)] >= 1.f) { d_vx[cid] = 0.f; }
		if (d_beta[CID(i, j + 1, k, Nx, Ny)] >= 1.f) { d_vy[cid] = 0.f; }
		if (d_beta[CID(i, j, k + 1, Nx, Ny)] >= 1.f) { d_vz[cid] = 0.f; }
	}
	d_ax[cid] = 0.f; d_ay[cid] = 0.f; d_az[cid] = 0.f;
}

/**  */
void WaveBlender::_freshCellPressure()
{
	const dim3 FDTD_threads(8, 8, 8);
	dim3 FDTD_blocks((_simParams.Nx + 7 - PML_WIDTH) / 8, (_simParams.Ny + 7 - PML_WIDTH) / 8, (_simParams.Nz + 7 - PML_WIDTH) / 8);

	const dim3 shader_threads(32);
	dim3 shader_blocks((_N_shader_points + 31) / 32);

	const REAL INCR = (REAL) _simParams.shader_srate / _simParams.FDTD_srate;  // For now, assumes shader rate is same for all objects

	// We use split-pressure buffers as acceleration buffers for convenience (since split-pressure only used in PML) 
	// -- just make sure to clear afterwards
	Ker_prepareFreshCell<<<shader_blocks, shader_threads>>>(d_px, d_py, d_pz, 
		d_shaderData, d_shaderMap, _N_shader_samples, _N_shader_points, 1./_simParams.dt, INCR);
	Ker_freshCellPressure<<<FDTD_blocks, FDTD_threads>>>(d_p, d_px, d_py, d_pz, d_beta, 
		_simParams.Nx, _simParams.Ny, _simParams.Nz, _simParams.RHO*_simParams.dx);
	Ker_clearSolid<<<FDTD_blocks, FDTD_threads>>>(d_vx, d_vy, d_vz, d_px, d_py, d_pz, d_beta, 
		_simParams.Nx, _simParams.Ny, _simParams.Nz);
}


/** */
void WaveBlender::_freshCellVelocity()
{
	// Determine fresh cells (based on _cell1 and _cell2 difference)
	std::set<int> fresh_cids;
	for (int cid = 0; cid < _cell2.size(); cid++)
	{
		if (_cell2[cid] == 0 && _cell1[cid] > 0 && _cell1[cid] != CAVITY_INTERIOR)
		{
			int oid = _cell1[cid] - 1;
			if (_objects[oid]->shaderClass() != SHADER_CLASS::POINT) { fresh_cids.insert(cid); }
		}
	}
	std::cout << "  # Fresh Cells = " << fresh_cids.size() << std::endl;

	// Prepare velocity fresh cells: determine interior faces
	std::map<int, int> interior_vxids, interior_vyids, interior_vzids;
	int vxcount = 0, vycount = 0, vzcount = 0;
	for (int cid : fresh_cids)
	{
		int i = cid % _simParams.Nx;
		int j = (cid / _simParams.Nx) % _simParams.Ny;
		int k = (cid / _simParams.Nx) / _simParams.Ny;

		int neighbor_cids[6] = { _cid(i - 1, j, k), _cid(i + 1, j, k),	 // left, right
								 _cid(i, j - 1, k), _cid(i, j + 1, k),	 // down, up
								 _cid(i, j, k - 1), _cid(i, j, k + 1) }; // back, front
		for (int n = 0; n < 6; n++)
		{
			if (_cell1[neighbor_cids[n]] == 0) { continue; }

			int D = (n % 2 == 0);
			if ((n == 0 || n == 1) && interior_vxids.count(_cid(i - D, j, k)) == 0)  // x-direction
			{
				interior_vxids.insert(std::pair<int, int>(_cid(i - D, j, k), vxcount));
				vxcount += 1;
			}
			else if ((n == 2 || n == 3) && interior_vyids.count(_cid(i, j - D, k)) == 0)  // y-direction
			{
				interior_vyids.insert(std::pair<int, int>(_cid(i, j - D, k), vycount));
				vycount += 1;
			}
			else if ((n == 4 || n == 5) && interior_vzids.count(_cid(i, j, k - D)) == 0)  // z-direction
			{
				interior_vzids.insert(std::pair<int, int>(_cid(i, j, k - D), vzcount));
				vzcount += 1;
			}
		}
	} // for (int cid : fresh_cids)

	// Build least-squares matrix A and vector g: copy velocities from device to host
	int N_DOF = vxcount + vycount + vzcount;
	Eigen::MatrixX<REAL> A = Eigen::MatrixX<REAL>::Zero(fresh_cids.size(), N_DOF);
	Eigen::VectorX<REAL> g = Eigen::VectorX<REAL>::Zero(fresh_cids.size());

	int row = 0;
	for (int cid : fresh_cids)
	{
		int i = cid % _simParams.Nx;
		int j = (cid / _simParams.Nx) % _simParams.Ny;
		int k = (cid / _simParams.Nx) / _simParams.Ny;

		for (int n = 0; n < 6; n++)
		{
			int D = (n % 2 == 0); REAL sign = (n % 2 == 0) ? -1. : 1.;

			int vnid; int col = -1; REAL vn;
			if (n < 2)  // x-direction
			{
				vnid = _cid(i - D, j, k);
				if (interior_vxids.count(vnid) > 0) { col = interior_vxids[vnid]; }
				cudaMemcpy(&vn, d_vx + vnid, sizeof(REAL), cudaMemcpyDeviceToHost);
			}
			else if (n < 4)  // y-direction
			{
				vnid = _cid(i, j - D, k);
				if (interior_vyids.count(vnid) > 0) { col = vxcount + interior_vyids[vnid]; }
				cudaMemcpy(&vn, d_vy + vnid, sizeof(REAL), cudaMemcpyDeviceToHost);
			}
			else if (n < 6)  // z-direction
			{
				vnid = _cid(i, j, k - D);
				if (interior_vzids.count(vnid) > 0) { col = vxcount + vycount + interior_vzids[vnid]; }
				cudaMemcpy(&vn, d_vz + vnid, sizeof(REAL), cudaMemcpyDeviceToHost);
			}

			if (col != -1) { A(row, col) = sign; }
			else { g[row] -= sign * vn; }
		}
		row += 1;
	} // for (int cid : fresh_cids)

	if (fresh_cids.size() > 0 && N_DOF > 0)
	{
		Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixX<REAL>> qr(A);  // least squares
		Eigen::VectorX<REAL> u = qr.solve(g);

		// Copy fresh cell velocity solve results back to device
		for (std::pair<int, int> pair : interior_vxids)  // x-direction
		{
			int vxid = pair.first; int col = pair.second;
			cudaMemcpy(d_vx + vxid, u.data() + col, sizeof(REAL), cudaMemcpyHostToDevice);
		}
		for (std::pair<int, int> pair : interior_vyids)  // y-direction
		{
			int vyid = pair.first; int col = vxcount + pair.second;
			cudaMemcpy(d_vy + vyid, u.data() + col, sizeof(REAL), cudaMemcpyHostToDevice);
		}
		for (std::pair<int, int> pair : interior_vzids)  // z-direction
		{
			int vzid = pair.first; int col = vxcount + vycount + pair.second;
			cudaMemcpy(d_vz + vzid, u.data() + col, sizeof(REAL), cudaMemcpyHostToDevice);
		}
		std::cout << "residual: " << (A * u - g).norm() << std::endl;
	}
}


/** Section 6.2.2 from [Xue et al. 2024] */
__global__ void Ker_shaderReInit(const REAL* __restrict__ d_vx, const REAL* __restrict__ d_vy, const REAL* __restrict__ d_vz,
	REAL* __restrict__ d_shaderData, const int* d_shaderMap, int N_shader_samples, int N_shader_points)
{
	int global_bid = threadIdx.x + blockDim.x * blockIdx.x;
	if (global_bid >= N_shader_points) { return; }

	// Decode shader map
	if (d_shaderMap[global_bid] < 0) { return; }  // Skip if isForce
	int dir = d_shaderMap[global_bid] % 3;
	int cid = d_shaderMap[global_bid] / 3;

	// Decode shader values
	int floor = (global_bid * N_shader_samples);
	REAL vb0 = d_shaderData[floor]; REAL v_0;

	if (dir == X_DIR) { v_0 = d_vx[cid]; }
	else if (dir == Y_DIR) { v_0 = d_vy[cid]; }
	else if (dir == Z_DIR) { v_0 = d_vz[cid]; }

	for (int t = 0; t < N_shader_samples; t++) { d_shaderData[floor + t] -= (vb0 - v_0); }
}

void WaveBlender::_shaderReInit()
{
	const dim3 shader_threads(32);
	dim3 shader_blocks((_N_shader_points + 31) / 32);

	Ker_shaderReInit<<<shader_blocks, shader_threads>>>(d_vx, d_vy, d_vz,
		d_shaderData, d_shaderMap, _N_shader_samples, _N_shader_points);
}
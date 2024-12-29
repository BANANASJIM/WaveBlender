/** (c) 2024 Kangrui Xue
 *
 * \file Shaders.h
 * \brief Declares Shader class and all sound sources based on it
 * 
 * See the .cpp/.cu files associated with each sound source for implementation details 
 */

#ifndef _SHADERS_H
#define _SHADERS_H


#ifdef USE_CUDA
	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"
	#include "cublas_v2.h"
#endif

#include "igl/read_triangle_mesh.h"
#include "igl/barycentric_coordinates.h"
#include "igl/AABB.h"

#include "ModalSound.h"
#include "FluidSound.h"

typedef float REAL;

enum SHADER_CLASS { MONOPOLE, SPEAKER, OCCLUDER, BUBBLES, MODAL, THIN_SHELL, POINT, DENSITY };

/** 
 * \class Shader
 * \brief Acoustic shader virtual class for representing vibration boundary conditions or forcing terms 
 */
class Shader 
{
public:
	Shader(int blendrate, int shader_srate, double ts) : _srate(shader_srate), _dt(1. / _srate),
		_ts(ts), _N_samples(shader_srate / blendrate + 1)
	{ }

	/** 
	 * \brief Given a block of memory and global index offset, populates the block with boundary velocity data. 
	 *		  A total of (_N_samples * _N_points) values will be written.
	 * \param[out] d_vb		   (..., _N_samples) block of linear memory (row major)
	 * \param[in]  global_bid  row offset into d_vb
	 */
	virtual void compute(REAL* d_vb, int global_bid) = 0;

	/** 
	 * \brief Sets the Shader sample points for the current batch 
	 * \param[in]  B   (_N_points, 3) matrix of boundary points to compute shader samples at
	 * \param[in]  BN  (_N_points, 3) matrix of corresponding boundary normals
	 */
	void setSamplePoints(const Eigen::MatrixX<REAL> &B, const Eigen::MatrixX<REAL> &BN)
	{
		_N_points = B.rows(); _B = B; _BN = BN;

		cudaFree(d_B); cudaFree(d_BN);
		cudaMalloc((void**) &d_B, _B.size() * sizeof(REAL));
		cudaMalloc((void**) &d_BN, _BN.size() * sizeof(REAL));

		cudaMemcpy(d_B, _B.data(), _B.size() * sizeof(REAL), cudaMemcpyHostToDevice);
		cudaMemcpy(d_BN, _BN.data(), _BN.size() * sizeof(REAL), cudaMemcpyHostToDevice);
	}

	int N_points() const { return _N_points; }
	SHADER_CLASS shaderClass() const { return _class; }

protected:
	int _srate = 48000;			//!< shader sample rate
	double _dt = 1. / _srate;	//!< shader timestep size
	int _step = 0;				//!< current shader timestep
	double _ts = 0.;			//!< simulation start time

	int _N_samples = 0;			//!< Number of shader samples in time (i.e., the batchsize)
	int _N_points = 0;			//!< Number of boundary points

	SHADER_CLASS _class;

	// B and BN are the set of all boundary positions and corresponding boundary normals
	// to compute Shader samples at, \see setSamplePoints()
	Eigen::Matrix<REAL, Eigen::Dynamic, 3, Eigen::RowMajor> _B, _BN;
	REAL* d_B, * d_BN;
};


/** 
 * \class Object 
 * \brief Extends Shader class with object geometry and animation handling
 */
class Object : public Shader 
{
public:
	Object(int blendrate, int shader_srate, double ts) : Shader(blendrate, shader_srate, ts) { }

	bool hasShader() const { return _hasShader; }
	bool changed() const { return _changed; }
	
	const Eigen::MatrixX<REAL> & V() { return _V2; }
	const Eigen::MatrixXi & F() { return _F; }

protected:
	bool _hasShader = false;
	bool _changed = true;		//!< whether the Object has moved / its geometry has changed since the last batch

	Eigen::MatrixX<REAL> _V0;	//!< original, non-animated surface mesh vertex positions
	Eigen::MatrixXi _F;			//!< surface mesh faces
	
	Eigen::MatrixX<REAL> _V1, _V2;	// current and next surface mesh vertex positions
	// (due to how we rasterize, we need to "look ahead" to the next triangle mesh)

	// ----- Closest point query -----
	igl::AABB<Eigen::MatrixX<REAL>, 3> _tree;
	void _closestPoint(Eigen::VectorXi& I, const Eigen::MatrixX<REAL>& B, const Eigen::MatrixX<REAL>& V);
	void _closestPoint(Eigen::VectorXi& I, Eigen::MatrixX<REAL>& W, const Eigen::MatrixX<REAL>& B, const Eigen::MatrixX<REAL> &V);

	// ----- Animation and I/O -----
	double _t1 = 0., _t2 = 0.;	// animation keyframe endpoint times 
	Eigen::Vector3d _translation1, _translation2;	// translation vectors
	Eigen::Vector4d _rotation1, _rotation2;			// rotation quaternions

	std::ifstream _animFileStream;	//!< animation file stream
	void _readAnimation();			//!< reads animation data for current _step

	/** \brief Reads .obj file and saves vertices and faces to V and F */
	static bool _readObj(const std::string& filename, Eigen::MatrixX<REAL> &V, Eigen::MatrixXi &F)
	{
		bool success = igl::read_triangle_mesh(filename, V, F);
		return success;
	}
};


/** 
 * \class Monopole
 * \brief Basic mono-frequency, monopole source shader used for testing
 */
class Monopole : public Object {
public:
	Monopole(int blendrate, int shader_srate, double ts, std::string meshFile, REAL freqHz, REAL speed, REAL C = 343.2) 
		: Object(blendrate, shader_srate, ts), _freqHz(freqHz), _speed(speed), _C(C)
	{
		_hasShader = true; _class = SHADER_CLASS::MONOPOLE;
		_readObj(meshFile, _V2, _F);
	}
	void compute(REAL* d_vb, int global_bid);

private:
	REAL _freqHz = 2000.;	//!< source frequency
	REAL _speed = 0.;		//!< source speed (x-direction) 
	REAL _C = 343.2;		//!< speed of sound
};


/**
 * \class Speaker 
 * \brief Speaker shader for pre-recorded input audio
 */
class Speaker : public Object {
public:
	Speaker(int blendrate, int shader_srate, double ts, std::string meshFile, std::string wavFile, int direction, std::string animFile) 
		: Object(blendrate, shader_srate, ts), _direction(direction)
	{
		_hasShader = true; _class = SHADER_CLASS::SPEAKER;
		_readWAV(wavFile);
		_readObj(meshFile, _V0, _F); _V2 = _V0;
		if (animFile != "") { _animFileStream.open(animFile); _readAnimation(); }
	}
	void compute(REAL* d_vb, int global_bid);

private:
	int _direction = 0;  //!< 0: Left, 1: Right, 2: Down, 3: Up, 4: Front, 5: Back
	// TODO: currently doesn't support non axis-aligned directions
	
	void _readWAV(std::string wavFile);
	REAL* d_audio;
};


/**
 * \class Occluder 
 * \brief Rigid, non-vibrating shader for animated occluders
 */
class Occluder : public Object {
public:
	Occluder(int blendrate, int shader_srate, double ts, std::string meshFile, std::string animFile) :
		Object(blendrate, shader_srate, ts)
	{
		_hasShader = (animFile != ""); _class = SHADER_CLASS::OCCLUDER;
		_readObj(meshFile, _V0, _F); _V2 = _V0;
		if (animFile != "") { _animFileStream.open(animFile); _readAnimation(); }
	}
	void compute(REAL* d_vb, int global_bid)
	{
		if (_animFileStream.is_open()) { _readAnimation(); }
		else if (_step > 0) { _changed = false; }
		_step += (_N_samples - 1);	// needed for _readAnimation()
	}
};


/**
 * \class Bubbles
 * \brief Bubble-based water sound shader
 */
class Bubbles : public Object {
public:
	Bubbles(int blendrate, int shader_srate, double ts, std::string bubFile, std::string fluidMeshDir, REAL dx = 0.)
		: Object(blendrate, shader_srate, ts), _solver(bubFile, 1. / shader_srate, 1, ts), _dx(dx), _fluidMeshDir(fluidMeshDir)
	{
		_hasShader = true; _class = SHADER_CLASS::BUBBLES;
		_readFluidMesh();
		cublasStatus_t status = cublasCreate(&_handle);
	}
	void compute(REAL* d_vb, int global_bid);

private:
	FluidSound::Solver<double> _solver;		//!< solver for timestepping bubble oscillations
	REAL _dx = 0.;				//!< FDTD cell size (for flux normalization)
	
	std::string _fluidMeshDir;	//!< path to directory containing fluid meshes
	void _readFluidMesh();
	
	std::vector<int> _activeOscIDs;			//!< IDs of active Oscillators during current batch
	int _max_N_osc = 0;		//!< maximum number of Oscillators allocated so far

	// --- Host (CPU) ---
	std::vector<REAL> _bubData;
	Eigen::MatrixX<REAL> _bubToBoundary;
	Eigen::MatrixX<REAL> _bubVels;

	// --- Device (GPU) ---
	REAL* d_bubData;
	REAL* d_bubToBoundary;
	REAL* d_bubVels;
	REAL* d_ones, * d_flux;  // for flux normalization

	cublasHandle_t _handle;
};


/**
 * \class Modal 
 * \brief Modal sound (plus acceleration noise) shader for vibrating rigid bodies
 */
class Modal : public Object {
public:
	Modal(int blendrate, int shadersrate, double ts, std::string meshFile, std::string animFile, std::string dataPrefix, std::string material)
		: Object(blendrate, shadersrate, ts), _solver(dataPrefix, meshFile, material, 1. / shadersrate)
	{
		_hasShader = true; _class = SHADER_CLASS::MODAL;
		_readObj(meshFile, _V0, _F); _V2 = _V0;
		_tree.init(_V0, _F);
		
		if (animFile != "") { _animFileStream.open(animFile); _readAnimation(); }
		cublasStatus_t status = cublasCreate(&_handle);
	}
	void compute(REAL* d_vb, int global_bid);

private:
	ModalSound::Solver _solver;		//!< solver for timestepping modal vibrations
	Eigen::Vector3d _accelSum = Eigen::Vector3d::Zero();

	void _setModeToBoundary(Eigen::MatrixX<REAL>& modeToBoundary);

	// --- Host (CPU) ---
	Eigen::MatrixX<REAL> _modeToBoundary1, _modeToBoundary2;
	Eigen::MatrixX<REAL> _modeVels;
	Eigen::MatrixX<REAL> _accelNoise;

	// --- Device (GPU) ---
	REAL* d_modeToBoundary1, *d_modeToBoundary2;
	REAL* d_modeVels;
	REAL* d_accelNoise;

	cublasHandle_t _handle;
};


/**
 * \class Shell
 * \brief Thin shell shader (right now, it's just loading precomputed simulation data from disk)
 */
class Shell : public Object {
public:
	Shell(int blendrate, int shadersrate, double ts, std::string meshFile, std::string animDir, std::string accDir, std::string mapFile) 
		: Object(blendrate, shadersrate, ts), _shellAnimDir(animDir), _shellAccelDir(accDir)
	{
		_hasShader = true; _class = SHADER_CLASS::THIN_SHELL;
		_readObj(meshFile, _V0, _F); _V2 = _V0;
		_readVertexMap(mapFile);
		
		_readShellAnimation();
		_vertAccel0 = Eigen::RowVectorXd::Zero(_V2.rows() * 3);
		_vertAccel1 = Eigen::RowVectorXd::Zero(_V2.rows() * 3);

		_megabatchsize = 100 * (_N_samples - 1) + 1;  // load 100 batches of accel data at once
		_vertVels = Eigen::MatrixXd::Zero(1, _V2.rows() * 3);
	}
	void compute(REAL* d_vb, int global_bid);

private:
	std::string _shellAnimDir;		//!< directory containing shell animation data
	std::string _shellAccelDir;		//!< directory containing shell vertex acceleration data
	void _readShellAnimation();

	Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> _vertDisplace;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _vertVels;
	Eigen::RowVectorXd _vertAccel0, _vertAccel1;	//!< current and next vertex accelerations (read from disk)

	void _readVertexMap(std::string mapFile);
	std::map<int, int> _vertMap;	//!< mapping from original (mesh) vertex indices to internal re-ordered indices

	// --- Host (CPU) ---
	Eigen::Matrix<REAL, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _batchVels;
	int _megabatchsize = 0;

	// (No device buffers because we copy _batchVels directly to d_vb)
};


/** 
 * \class Point
 * \brief Point force shader
 */
class Point : public Object {
public:
	Point(int blendrate, int shadersrate, double ts, std::string impulseFile, REAL dx)
		: Object(blendrate, shadersrate, ts), _dx(dx)
	{
		_hasShader = true; _class = SHADER_CLASS::POINT;
		_readImpulses(impulseFile);
		_getActiveImpulseList();
	}
	void compute(REAL* d_force, int global_bid);

private: 
	REAL _dx = 0.;		//!< FDTD cell size

	void _readImpulses(const std::string& filename);
	void _getActiveImpulseList();
	
	// --- All impulse data ---
	// TODO: unify with accel noise in modal shader
	std::vector<double> _times;				//!< impact times
	std::vector<double> _tau;				//!< Hertz contact timescale
	std::vector<Eigen::Vector3<REAL>> _DV;	//!< body's velocity change
	std::vector<Eigen::Vector3<REAL>> _P;	//!< impact position

	// --- Host (CPU) ---
	std::set<int> _activeImpulseIDs;
	std::vector<REAL> _impulseData;
	Eigen::MatrixX<REAL> _impulseVels;
	
	// --- Device (GPU) ---
	REAL* d_impulseData;
	REAL* d_impulseToBoundary;
	REAL* d_impulseVels;
};


/**
 * \class Density 
 * \brief User-defined density ("auxiliary beta")
 */
class Density : public Object {
public:
	Density(int blendrate, int shader_srate, double ts, std::string betaDir) : 
		Object(blendrate, shader_srate, ts), _betaDir(betaDir)
	{
		_hasShader = true; _class = SHADER_CLASS::DENSITY;
		_readDensity();
	}
	void compute(REAL* d_vb, int global_bid) 
	{ 
		_step += (_N_samples - 1); 
		_readDensity();
	}
private:
	std::string _betaDir;

	const int FPS = 60;  // TODO: remove hard-coding
	void _readDensity()
	{
		std::vector<Eigen::Vector3<REAL>> Pos; 
		std::vector<int> betas;

		int frame = (_step * _dt + _ts) * FPS + 1;
		std::stringstream ss; ss << std::setw(5) << std::setfill('0') << frame;
		std::string filename = _betaDir + "betaField_6x40x6_" + ss.str() + ".txt";

		std::ifstream inFile(filename); 
		std::string line;
		
		std::getline(inFile, line);  // Skip first line
		while (!line.empty() && inFile.good())
		{
			std::getline(inFile, line);
			std::istringstream is(line);

			REAL density, Px, Py, Pz;
			is >> density >> Px >> Py >> Pz;

			Pos.push_back(Eigen::Vector3<REAL>(Px, Py, Pz)); 
			betas.push_back(int(100 * density));
		}
		_V2 = Eigen::MatrixX<REAL>::Zero(Pos.size(), 3);
		_F = Eigen::MatrixXi::Zero(betas.size(), 3);
		for (int r = 0; r < Pos.size(); r++)
		{
			_V2.row(r) = Pos[r]; 
			_F(r, 0) = betas[r];  // Hack to encode density in cell face
		}
		_changed = inFile.is_open();
	}
};


#endif // #ifndef _SHADERS_H
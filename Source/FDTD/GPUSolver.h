/** (c) 2024 Kangrui Xue
 *
 * \file GPUSolver.h
 * \brief Declares FDTDSolver class (GPU)
 */

#ifndef GPU_SOLVER_H
#define GPU_SOLVER_H

#include "FDTDCommon.h"
#include "GPUListener.h"

#define X_DIR 0
#define Y_DIR 1
#define Z_DIR 2

#define PML_WIDTH 8
#define CID(i, j, k, Nx, Ny) ( ((Ny) * (Nx)) * (k) + (Nx) * (j) + (i) )

/** 
 * \class FDTDSolver
 * \brief Base GPU FDTD acoustic wavesolver (see GPUSolver.cu for kernel implementations)
 * 
 * NOTE: rasterization and object-level logic handled in WaveBlender.h/cu
 */
class FDTDSolver
{
public:
    FDTDSolver(const SimParams& params);
    ~FDTDSolver();
    
    /** \brief Runs FDTD simulation for '_N_FDTD_samples' timesteps */
    void runFDTD();

    /** 
     * \brief Adds GPUListener to simulation
     * \param[in]  format       Listener format ("Mono" only for now)
     * \param[in]  listenerP    (x, y, z) Position of listener (for MonoListener)
     * \param[in]  output_name  Output file name 
     */
    void addListener(const std::string& format, const std::vector<REAL>& listenerP, const std::string& output_name = "output");

    /** \brief Writes pressure and beta z-slice (z = 0) to file for debugging */
    void logZSlice(const std::string& filetag);

protected:
    SimParams _simParams;       //!< simulation parameters
    int _step = 0;              //!< current simulation time step

    int _gridSize = 0;          //!< total number of cells (Nx * Ny * Nz)
    int _N_FDTD_samples = 0;    //!< number of FDTD samples (timesteps) per batch

    // ----- Device (GPU) buffers -----
    REAL* d_p;      //!< pressure
    REAL* d_vx;     //!< x-velocity
    REAL* d_vy;     //!< y-velocity
    REAL* d_vz;     //!< z-velocity
    
    REAL* d_beta;   //!< blending field
    int* d_cell;    //!< rasterized cell states, i.e., which object each cell contains (0 for air)
    /*
     * While d_p[cid] and d_beta[cid] values are stored at cell centers, velocities are staggered on
     * cell faces. By convention, each cell index stores the velocities on its RIGHT, TOP, and BACK faces.
     */
    REAL* d_px, * d_py, * d_pz;     // split pressure field (for PML)
    REAL* d_pmlNp, * d_pmlDp, * d_pmlNv, * d_pmlDv;     // PML weights (separate pressure and velocity)
    

    int* d_shaderMap;       //!< map from boundary cell to shader velocity (or force) in d_shaderData
    REAL* d_shaderData;     //!< linear array of all shader samples for current batch
    /* 
     * For boundary points (b1, ..., bN) and times (0, ..., T), d_shaderData corresponds to:
     *   [ v_b1(0) ... v_b1(T) ] 
     *   [   :            :    ]
     *   [ v_bN(0) ... v_bN(T) ]
     */
    int _N_shader_samples = 0;      //!< number of shader samples in time per batch
    int _N_shader_points = 0;       //!< number of shader points (object boundary faces)
    int _max_N_shader_points = 0;   //!< maximum number of shader points allocated in memory so far


    // ----- Basic helper functions -----
    int _cid(int i, int j, int k) const  //! Converts (i, j, k) indices to global cell index
    {
        return (_simParams.Ny * _simParams.Nx) * k + _simParams.Nx * j + i;
    }
    int _cid(const Eigen::Vector3<REAL>& P) const  //! Converts (x, y, z) Position to global cell index
    {
        int i = (P[0] / _simParams.dx) + (_simParams.Nx - 1) / 2.;
        int j = (P[1] / _simParams.dx) + (_simParams.Ny - 1) / 2.;
        int k = (P[2] / _simParams.dx) + (_simParams.Nz - 1) / 2.;
        return _cid(i, j, k);
    }
    Eigen::Vector3<REAL> _Pos(int i, int j, int k) const  //! Converts (i, j, k) indices to (x, y, z) Position
    {
        return Eigen::Vector3<REAL>({ (i - (_simParams.Nx - 1) / 2.f) * _simParams.dx,
            (j - (_simParams.Ny - 1) / 2.f) * _simParams.dx,
            (k - (_simParams.Nz - 1) / 2.f) * _simParams.dx });
    }
    bool _inBounds(int i, int j, int k) const  //! Checks if (i, j, k) indices are within simulation bounds
    {
        return (i >= 0 && i < _simParams.Nx && j >= 0 && j < _simParams.Ny && k >= 0 && k < _simParams.Nz);
    }

private:
    // Precomputed constants
    REAL _RHO_CC_dt, _inv_dx, _inv_RHO_dt;

    REAL _damping = 0.;  // this is a temporary solution in lieu of frequency-dependent boundary conditions 

    std::vector<GPUListener*> _listeners;

    /** \private Allocates memory and precomputes constants for PML */
    void _initializePML();
};

#endif  // #ifndef GPU_SOLVER_H
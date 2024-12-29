/** (c) 2024 Kangrui Xue
 * 
 * \file WaveBlender.h
 * \brief Public interface for WaveBlender sound-source rendering engine
 * 
 * NOTES:
 *  Ordering of Objects matters: Point sources MUST be placed at end; later Objects will override previous rasterizations
 */

#ifndef WAVEBLENDER_H
#define WAVEBLENDER_H

#ifdef USE_CUDA
    #include "GPUSolver.h"
#else
    #include "CPUSolver.h"  // TODO: NOT IMPLEMENTED
#endif

const int CAVITY_INTERIOR = 255;

/**
 * \class WaveBlender 
 * \brief Extends FDTDSolver class with the gritty implementation details (e.g., fresh cell extrapolation)
 *        needed to make the WaveBlender system work.
 */
class WaveBlender : public FDTDSolver
{
public:
    WaveBlender(const SimParams& params) : FDTDSolver(params)
    {
        _cell1.resize(_gridSize); _cell2.resize(_gridSize);
    }

    /** \brief Runs a complete simulation batch: from rasterization to FDTD timestepping */
    bool runBatch();

    /** 
     * \brief Adds an Object to the simulator (\see Shaders.h) 
     * \param[in]  offset      global  Object offset position 
     * \param[in]  object_ptr  Object  Object shared_ptr (for now, instantiation handled in main.cpp)
     */
    void addObject(Eigen::Vector3<REAL> offset, std::shared_ptr<Object> object_ptr) 
    { 
        _offsets.push_back(offset);
        _objects.push_back(object_ptr); 
    }

private:
    std::vector<Eigen::Vector3<REAL>> _offsets;
    std::vector<std::shared_ptr<Object>> _objects;

    std::vector<int> _cell1, _cell2;    // rasterized cell states at batch endpoints t1 and t2

    // ----- Rasterization -----
    void _rasterize(const std::set<int>& unchanged_oids, bool logRaster=false);
    void _setupShaders();

    // ----- Per-batch overhead -----
    void _detectCavities();
    void _freshCellPressure();
    void _freshCellVelocity();
    void _shaderReInit();
};

#endif // #ifndef WAVEBLENDER_H
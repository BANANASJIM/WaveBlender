/** (c) 2024 Kangrui Xue 
 *
 * \file FDTDCommon.h
 * \brief Defines FDTD global simulation parameters
 */

#ifndef _FDTD_COMMON_H
#define _FDTD_COMMON_H

#include "Shaders.h"


enum SCHEME { NO_BLEND, SMOOTHSTEP };

struct SimParams  // default simulation parameters
{
    int Nx = 80, Ny = 80, Nz = 80;      //!< FDTD grid dimensions
    REAL dx = 0.005;                    //!< cell size

    int FDTD_srate = 120000;            //!< FDTD sample rate
    double dt = 1. / FDTD_srate;        //!< FDTD timestep size
    double ts = 0., tf = 0.;            //!< simulation start and end times

    int shader_srate = 48000;           //!< shader sample rate
    int blendrate = 100;                //!< blending rate (how often to rasterize geometry)
    SCHEME scheme = SCHEME::SMOOTHSTEP;
    
    const REAL C = 343.2;               //!< speed of sound
    const REAL RHO = 1.204;             //!< density of acoustic medium

    REAL damping = 0.;      // TODO: frequency-dependent boundary conditions
};

#endif // #ifndef _FDTD_COMMON_H
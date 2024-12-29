/** (c) 2024 Kangrui Xue
 *
 * \file SpeakerShader.cu
 * \brief Implements Speaker acoustic shader for pre-recorded input audio
 * 
 * References:
 *   [Xue et al. 2024] WaveBlender: Practical Sound-Source Animation in Blended Domains
 */

#include "Shaders.h"

#include "AudioFile.h"

#ifdef USE_CUDA

__constant__ REAL SCALE = 100.;		// Manually scale input audio amplitude

/** Section 5.1.1 from [Xue et al. 2024] */
__global__ void Ker_speaker(REAL* d_vb, int global_bid, const REAL* d_BN, const REAL* d_audio,
	int dir, int N_points, int N_samples, int start_step)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int k = threadIdx.y + blockDim.y * blockIdx.y;
	if (i >= N_points || k >= N_samples) { return; }

	REAL xn = d_BN[3 * i]; REAL yn = d_BN[3 * i + 1]; REAL zn = d_BN[3 * i + 2];

	REAL vb = 0.f;  // Manually specify speaker direction, TODO: generalize to non-planar surfaces
	if		(dir == 0 && xn <= -1.f) { vb = -d_audio[start_step + k]; }  // LEFT
	else if (dir == 1 && xn >=  1.f) { vb =  d_audio[start_step + k]; }  // RIGHT
	
	else if (dir == 2 && yn <= -1.f) { vb = -d_audio[start_step + k]; }  // DOWN
	else if (dir == 3 && yn >=  1.f) { vb =  d_audio[start_step + k]; }  // UP
	
	else if (dir == 4 && zn <= -1.f) { vb = -d_audio[start_step + k]; }  // FRONT
	else if (dir == 5 && zn >=  1.f) { vb =  d_audio[start_step + k]; }  // BACK

	d_vb[(global_bid + i) * N_samples + k] = SCALE * vb;
}
#else

// NOT IMPLEMENTED

#endif

/** */
void Speaker::_readWAV(std::string wavFile)
{
	AudioFile<REAL> audioFile;
	audioFile.load(wavFile);
	if (audioFile.getSampleRate() != _srate) { throw std::runtime_error("Mismatch in input audio sample rate!"); }

	std::vector<REAL> audio_vn;
	REAL an1 = 0., an2 = 0.;
	for (int i = 0; i < audioFile.getNumSamplesPerChannel(); i++)
	{
		an1 = an2;
		an2 = audioFile.samples[0][i];  // read next sample (assumes mono channel)
	
		REAL vn = !audio_vn.empty() ? audio_vn.back() : 0.f;
		audio_vn.push_back(vn + (an1 + an2) / 2. * _dt);  // Trapezoidal rule (bilinear transform)
	}
	cudaMalloc((void**)&d_audio, audio_vn.size() * sizeof(REAL));
	cudaMemcpy(d_audio, audio_vn.data(), audio_vn.size() * sizeof(REAL), cudaMemcpyHostToDevice);
}

/** */
void Speaker::compute(REAL* d_vb, int global_bid)
{
	const dim3 threads(16, 16);
	dim3 blocks((_N_points + 15) / 16, (_N_samples + 15) / 16);

	Ker_speaker<<<blocks, threads>>>(d_vb, global_bid, d_BN, d_audio, _direction, _N_points, _N_samples, _step);
	_step += (_N_samples - 1);

	if (_animFileStream.is_open()) { _readAnimation(); }
	else if (_step > 0) { _changed = false; }
}

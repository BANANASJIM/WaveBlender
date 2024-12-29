/** (c) 2024 Kangrui Xue 
 *
 * \file GPUListener.h
 * \brief Defines Listener class for GPU FDTD wavesolver
 */

#ifndef _GPU_LISTENER_H
#define _GPU_LISTENER_H


/**
 * \class GPUListener
 * \brief Base listener class for writing GPUSolver output
 */
class GPUListener
{
public:
	/** 
	 * \brief Adds a single pressure field sample
	 * \param[in]  d_p	current pressure field (on device memory)
	 * \param[in]  s	current sample index (must satisfy s < GPUSolver._N_FDTD_samples)
	 */
	virtual void addSample(REAL* d_p, int s) = 0;

	/** \brief Writes a batch of pressure samples to file */
	virtual void write() = 0;

	~GPUListener() { cudaFree(d_buf); _outFileStream.close(); }

protected:
	REAL* d_buf;		//!< output buffer (on device memory)
	int _bufsize = 0;	//!< number of samples in buffer

	std::ofstream _outFileStream;	//!< output filestream
};


/** 
 * \class MonoListener
 * \brief Mono-channel, stationary listener for single-point pressure output 
 */
class MonoListener : public GPUListener
{
public:
	/** 
	 * \brief Constructor 
	 * \param[in]  listener_cid    cell index of listening position
	 * \param[in]  N_FDTD_samples  number of FDTD samples per batch
	 * \param[in]  output_name	   output file name
	 */
	MonoListener(int listener_cid, int N_FDTD_samples, std::string output_name = "output")
	{
		_listener_cid = listener_cid;
		_bufsize = N_FDTD_samples;

		cudaMalloc((void**)&d_buf, N_FDTD_samples * sizeof(REAL));
		_outFileStream.open(output_name + ".bin", std::ofstream::binary);
	}

	/** \brief Samples pressure at the listening position and writes to d_buf[s] */
	void addSample(REAL* d_p, int s)
	{
		cudaMemcpy(d_buf + s, d_p + _listener_cid, sizeof(REAL), cudaMemcpyDeviceToDevice);
	}

	/** \brief Writes a batch of pressure samples to file */
	void write()
	{
		std::vector<REAL> out(_bufsize);
		cudaMemcpy(out.data(), d_buf, _bufsize * sizeof(REAL), cudaMemcpyDeviceToHost);
		_outFileStream.write((char*)out.data(), _bufsize * sizeof(REAL));
	}

private:
	int _listener_cid = 0;
};


#endif // #ifndef _GPU_LISTENER_H
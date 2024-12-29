/** (c) 2024 Kangrui Xue
 *
 * \file main.cpp
 * \brief Parses JSON config file and runs simulation
 */

#include "WaveBlender.h"

#include "json.hpp"
using json = nlohmann::json;


void Run(const std::string &configFile)
{
	std::ifstream config_ifstream(configFile);
	json config; config_ifstream >> config;

	// Parse global simulation parameters
	SimParams params;
	params.Nx = config["Nx"]; params.Ny = config["Ny"]; params.Nz = config["Nz"];
	params.dx = config["cellsize"];

	params.FDTD_srate = config["FDTD_srate"];
	params.dt = 1. / params.FDTD_srate;
	params.ts = config["ts"]; params.tf = config["tf"];

	params.shader_srate = config["shader_srate"];
	params.blendrate = config["blendrate"];

	params.damping = config["damping"];

	WaveBlender solver(params);


	// Parse object parameters
	for (const json& object_config : config["objects"])
	{
		std::vector<REAL> tmp_offset = object_config["offset"];
		Eigen::Vector3<REAL> offset = { tmp_offset[0], tmp_offset[1], tmp_offset[2] };

		std::string shader = object_config["shader"];
		if (shader == "Monopole")
		{
			std::string meshFile = object_config["meshFile"];
			REAL freqHz = object_config["freqHz"];
			REAL speed = object_config["speed"];
			solver.addObject(offset, std::shared_ptr<Object>(new Monopole(params.blendrate, params.shader_srate,
				params.ts, meshFile, freqHz, speed, params.C)));
		}
		else if (shader == "Speaker")
		{
			std::string meshFile = object_config["meshFile"];
			std::string wavFile = object_config["wavFile"];
			int direction = object_config["direction"];
			std::string animFile = object_config["animFile"];
			solver.addObject(offset, std::shared_ptr<Object>(new Speaker(params.blendrate, params.shader_srate,
				params.ts, meshFile, wavFile, direction, animFile)));
		}
		else if (shader == "Occluder")
		{
			std::string meshFile = object_config["meshFile"];
			std::string animFile = object_config["animFile"];
			solver.addObject(offset, std::shared_ptr<Object>(new Occluder(params.blendrate, params.shader_srate,
				params.ts, meshFile, animFile)));
		}
		else if (shader == "Bubbles")
		{
			std::string bubFile = object_config["bubFile"];
			std::string meshDir = object_config["meshDir"];
			solver.addObject(offset, std::shared_ptr<Object>(new Bubbles(params.blendrate, params.shader_srate,
				params.ts, bubFile, meshDir, params.dx)));
		}
		else if (shader == "Modal")
		{
			std::string meshFile = object_config["meshFile"];
			std::string animFile = object_config["animFile"];
			std::string dataPrefix = object_config["dataPrefix"];
			std::string material = object_config["material"];
			solver.addObject(offset, std::shared_ptr<Object>(new Modal(params.blendrate, params.shader_srate,
				params.ts, meshFile, animFile, dataPrefix, material)) );
		}
		else if (shader == "Shell")
		{
			std::string meshFile = object_config["meshFile"];
			std::string animDir = object_config["animDir"];
			std::string accDir = object_config["accDir"];
			std::string mapFile = object_config["mapFile"];
			solver.addObject(offset, std::shared_ptr<Object>(new Shell(params.blendrate, params.shader_srate,
				params.ts, meshFile, animDir, accDir, mapFile)) );
		}
		else if (shader == "Point")
		{
			std::string impulseFile = object_config["impulseFile"];
			solver.addObject(offset, std::shared_ptr<Object>(new Point(params.blendrate, params.shader_srate, 
				params.ts, impulseFile, params.dx)) );
		}
		else if (shader == "Density")
		{
			std::string betaDir = object_config["betaDir"];
			solver.addObject(offset, std::shared_ptr<Object>(new Density(params.blendrate, params.shader_srate, 
				params.ts, betaDir)) );
		}
		else { throw std::runtime_error("Invalid Shader: " + shader); }
	}


	// Parse listener parameters
	for (const json& listener_config : config["listeners"])
	{
		std::string format = listener_config["format"];
		std::string output_name = listener_config["output"];
		std::vector<REAL> position = listener_config["position"];

		solver.addListener(format, position, output_name);
	}


	// Run simulation
	while (solver.runBatch()) { }
}


// Usage: ./WaveBlender [path to config file]
int main(int argc, char* argv[])
{
	std::string configFile = "../Scenes/CupPhone/config.json";
	if (argc > 1) { configFile = std::string(argv[1]); }

	Run(configFile);
	return 0;
}
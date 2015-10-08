#ifndef CUDAUTIL_HPP
#define CUDAUTIL_HPP

#include <vector>

namespace cuda {
	//Taken from https://github.com/ComputationalRadiationPhysics/HaseOnGpu
	//Author: Erik Zenker, Carlchristian Eckert
	std::vector<unsigned> getFreeDevices(unsigned maxGpus);
}

#endif
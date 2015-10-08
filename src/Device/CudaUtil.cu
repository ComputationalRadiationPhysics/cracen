#include <iostream>
#include "CudaUtil.hpp"
#include "../Config/Constants.hpp"

std::ostream& operator<<(std::ostream& lhs, const cudaDeviceProp& rhs) {
	lhs << "Device found:" << std::endl;
	lhs << rhs.name << std::endl;
	lhs << "Cuda " << rhs.major << "." << rhs.minor << ", " << rhs.totalGlobalMem << " GB of global memory." << std::endl;
	if( (rhs.major < MIN_COMPUTE_CAPABILITY_MAJOR) || (rhs.major == MIN_COMPUTE_CAPABILITY_MAJOR && rhs.minor < MIN_COMPUTE_CAPABILITY_MINOR) ) {
		lhs << "Compute capability of this device is too low. (Minumum CC "<< MIN_COMPUTE_CAPABILITY_MAJOR << "." << MIN_COMPUTE_CAPABILITY_MINOR << ")" << std::endl;
	}
	return lhs;
}

//Taken from https://github.com/ComputationalRadiationPhysics/HaseOnGpu
//Author: Erik Zenker, Carlchristian Eckert
std::vector<unsigned> cuda::getFreeDevices(unsigned maxGpus){
		cudaDeviceProp prop;
		unsigned int minMajor = MIN_COMPUTE_CAPABILITY_MAJOR;
		unsigned int minMinor = MIN_COMPUTE_CAPABILITY_MINOR;
		int count;
		std::vector<unsigned> devices;

		// Get number of devices
		cudaError err = cudaGetDeviceCount(&count);
		/* Check the cuda runtime environment */
		if(err != cudaSuccess) {
			std::cerr << "Something went wrong during the creation the context, or no Cuda capable devices are installed on the system." << std::endl;
			std::cerr << "Exit." << std::endl;
			return devices;
		}
		// Check devices for compute capability and if device is busy
		unsigned devicesAllocated = 0;
		for(int i=0; i < count; ++i){
			cudaGetDeviceProperties(&prop, i);
			std::cout << prop << std::endl;
			if( (prop.major > minMajor) || (prop.major == minMajor && prop.minor >= minMinor) ){
				cudaSetDevice(i);
				int* occupy; //TODO: occupy gets allocated, but never cudaFree'd -> small memory leak!
				if(cudaMalloc((void**) &occupy, sizeof(int)) == cudaSuccess){
					devices.push_back(i);
					devicesAllocated++;
					if(devicesAllocated == maxGpus)
						break;
				}
			}
		}
		// Exit if no device was found
		if(devices.size() == 0){
			std::cout << "None of the free CUDA-capable devices is sufficient!" << std::endl;
			exit(1);
		}

		// Print device information
		cudaSetDevice(devices.at(0));
		std::cout << "Found " << int(devices.size()) << " available CUDA devices with Compute Capability >= " << minMajor << "." << minMinor << "):" << std::endl;
		for(unsigned i=0; i<devices.size(); ++i){
			cudaGetDeviceProperties(&prop, devices[i]);
			std::cout << "[" << devices[i] << "] " << prop.name << " (Compute Capability " << prop.major << "." << prop.minor << ")" << std::endl;
		}

		return devices;

	}
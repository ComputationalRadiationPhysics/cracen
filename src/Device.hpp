#ifndef DEVICE_HPP
#define DEVICE_HPP

#define DEVICE __device__
#ifdef DEBUG_ENABLED
#include <cstdio>
#define handleLastError() cudaDeviceSynchronize(); handle_error( cudaGetLastError(),"Kernel Error occured:\"", __LINE__, __FILE__)
void handle_error(cudaError_t err, const char* error, int line, const char* file) {
	if(err != cudaSuccess) std::cerr << error << cudaGetErrorString(err) << "\" in Line " << line << " in File " << file << std::endl;
}
#else
#define handleLastError()
#endif

#endif

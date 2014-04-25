//! \file

#ifndef LEVMARQ_H
#define LEVMARQ_H

#include <thrust/device_vector.h>
#include "Types.h"
#include "UtilKernels.h"
#include "GaussJordan.h"
//TODO: remove Util.h
#include "tests/Util.h"
#include "FitFunctor.h"

texture<DATATYPE, 2, cudaReadModeElementType> dataTexture0, dataTexture1, dataTexture2, dataTexture3, dataTexture4, dataTexture5;

/*!
 * \brief getSample returns the y value of a given sample index
 * \param I sample index
 * \param INDEXDATASET index of the current dataset (GPU mode) or not used (CPU mode)
 * \return y value
 */
template<unsigned int tex>
__device__ float getSample(float I, int INDEXDATASET);

template<> __device__ float getSample<0>(float I, int INDEXDATASET) {
	return tex2D(dataTexture0, (I) + 0.5, (INDEXDATASET) + 0.5);
}
template<> __device__ float getSample<1>(float I, int INDEXDATASET) {
	return tex2D(dataTexture1, (I) + 0.5, (INDEXDATASET) + 0.5);
}
template<> __device__ float getSample<2>(float I, int INDEXDATASET) {
	return tex2D(dataTexture2, (I) + 0.5, (INDEXDATASET) + 0.5);
}
template<> __device__ float getSample<3>(float I, int INDEXDATASET) {
	return tex2D(dataTexture3, (I) + 0.5, (INDEXDATASET) + 0.5);
}
template<> __device__ float getSample<4>(float I, int INDEXDATASET) {
	return tex2D(dataTexture4, (I) + 0.5, (INDEXDATASET) + 0.5);
}
template<> __device__ float getSample<5>(float I, int INDEXDATASET) {
	return tex2D(dataTexture5, (I) + 0.5, (INDEXDATASET) + 0.5);
}

//Fit should be a FitFunctor
template < class Fit, unsigned int tex>
__global__ void calcF(int wave, float* param, float* F, unsigned int offset, const unsigned int sample_count, const unsigned int interpolation_count) {
	if(threadIdx.x*interpolation_count < sample_count) {
		float x = threadIdx.x*interpolation_count+offset;
		float y = getSample<tex>(x,wave);
		F[threadIdx.x] = -1*Fit::modelFunction(x,y,param);
	} else {
		F[threadIdx.x] = 0;
	}
}

//Fit should be a FitFunctor
template <class Fit, unsigned int tex>
__global__ void calcDerivF(int wave, float* param, float mu, float* deriv_F, const unsigned int offset, const unsigned sample_count, const unsigned int interpolation_count) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	MatrixAccess<float> deriv(Fit::numberOfParams(), deriv_F);
	
	if(y < sample_count/interpolation_count+Fit::numberOfParams() && x < Fit::numberOfParams()) {
		if(y*interpolation_count < sample_count) {
			deriv[x][y] = Fit::derivation(x,y*interpolation_count+offset,getSample<tex>(y*interpolation_count+offset,wave),param);
		} else {
			if((y - sample_count/interpolation_count) == x) {
				deriv[x][y] = mu;
			} else {
				deriv[x][y] = 0;
			}
		}
	}
}

//template <class T>
float* pcast(thrust::device_vector<float>& dev) {
	return thrust::raw_pointer_cast(&dev[0]);
}

template <class Fit>
__global__ void getWindowKernel(int i, int sample_count, Window* window) {
	*window = Fit::getWindow(i, sample_count);
}

template <class Fit>
Window getWindowWrapper(int i, int sample_count) {
	Window *device, host;
	cudaMalloc((void**) &device, sizeof(Window));
	getWindowKernel<Fit><<<1,1>>>(i, sample_count, device);
	cudaMemcpy(&host, device, sizeof(Window), cudaMemcpyDeviceToHost);
	return host;
}
//Fit should be a FitFunctor
template <class Fit, unsigned int tex>
int levenbergMarquardt(const unsigned sample_count, const unsigned int max_window_size, const unsigned int chunk_count, const unsigned int interpolation_count) {
	//TODO: Convert to Kernel
	//TODO: Waveformen Sequenziell abarbeiten (keine Taskparallelität)
	unsigned int numberOfParams = Fit::numberOfParams();
	thrust::device_vector<float> F(max_window_size+numberOfParams), b(numberOfParams), s(numberOfParams), 
								 deriv_F((max_window_size+numberOfParams)*numberOfParams),
								 param(numberOfParams), param2(numberOfParams), param_old(numberOfParams),
								 G((numberOfParams)*(numberOfParams)), G_inverse((numberOfParams)*(numberOfParams));
	float mu, u1;
	for(int i = 0; i < numberOfParams; i++) param[i] = 0;		
					 
	for(int i = 0; i < chunk_count; i++) {
		////std::cout << "Sample: " << i << std::endl;
		float roh;
		mu = 1;
		int counter = 0;
		do {
			counter++;
			for(int j = 0; j < numberOfParams; j++) param_old[j] = param[j];
			////std::cout << "param:" << std::endl;
			//printMat(param, 1 , numberOfParams);
			/* Abschnitt 1 */
			//Calc F(param)
			Window window = getWindowWrapper<Fit>(i, sample_count);
			int sampling_points = window.width/interpolation_count;
			calcF<Fit, tex><<<1,sampling_points+numberOfParams>>>(i, pcast(param), pcast(F), window.offset, sample_count, interpolation_count);
			//for(int j = 0; j < sampling_points; j++) //std::cout << "f("<< j*interpolation_count << ")=" << F[j] << std::endl;
			////std::cout << "F: " << std::endl;
			//printMat(F, 1, sampling_points+numberOfParams);
			handleLastError();
			//Calc F'(param)
			dim3 blockSize(32,32);
			dim3 gridSize(ceil((float) numberOfParams/32), ceil((float) sampling_points+numberOfParams/32));
			calcDerivF<Fit, tex><<<gridSize,blockSize>>>(i, pcast(param), mu, pcast(deriv_F), window.offset, sample_count, interpolation_count);
			handleLastError();
		
			////std::cout << "F': " << std::endl;
			//printMat(deriv_F, sampling_points+numberOfParams, numberOfParams);
		
			/* Abschnitt 2 */
		
			//Solve minimization problem
			//calc A^T*A => G
			transpose(pcast(deriv_F), sampling_points+numberOfParams, numberOfParams);
			//printMat(deriv_F, numberOfParams, sampling_points+numberOfParams);
			handleLastError();
		
			orthogonalMatProd(pcast(deriv_F), pcast(G), numberOfParams, sampling_points+numberOfParams);
			handleLastError();
			////std::cout << "G: " << std::endl;
			//printMat(G, numberOfParams, numberOfParams);
	
			//calc G^-1
			gaussJordan(pcast(G), pcast(G_inverse), numberOfParams);
			handleLastError();
			////std::cout << "G^-1: " << std::endl;
			//printMat(G_inverse, numberOfParams, numberOfParams);
		
			//calc A^T*F => b
			matProduct(pcast(deriv_F), pcast(F), pcast(b), numberOfParams, sampling_points+numberOfParams, sampling_points+numberOfParams, 1);
			handleLastError();
			////std::cout << "b" << std::endl;
			//printMat(b,1, numberOfParams);
			
			//calc G^-1*b => s
			matProduct(pcast(G_inverse), pcast(b), pcast(s), numberOfParams, numberOfParams, numberOfParams, 1);
			handleLastError();
			
			////std::cout << "s" << std::endl;
			//printMat(s,1, numberOfParams);
			
			/* Abschnitt 3 */
		
			//Fold F(param)
			thrust::device_vector<float> Temp(1);
			matProduct(pcast(F), pcast(F), pcast(Temp), 1, sampling_points+numberOfParams, sampling_points+numberOfParams, 1);
		
			handleLastError();
			u1 = Temp[0];
			//std::cout << "u1=" << u1;
		
			//Calc F(param+s)
			thrust::device_vector<float> F1(sampling_points);
			for(int j = 0; j < numberOfParams; j++) param2[j] = param[j] + s[j];
			//Fold F(param+s)
			calcF<Fit, tex><<<1,sampling_points+numberOfParams>>>(i, pcast(param2), pcast(F1), window.offset, sample_count, interpolation_count);
			matProduct(pcast(F1), pcast(F1), pcast(Temp),1, sampling_points, sampling_points, 1);
			handleLastError();
			float u2 = Temp[0];
			//std::cout << ";u2=" << u2;
		

			//Calc F'(param)*s
			transpose(pcast(deriv_F), numberOfParams, sampling_points+numberOfParams);
			matProduct(pcast(deriv_F), pcast(s), pcast(F1), sampling_points, numberOfParams, numberOfParams, 1);
			//matProduct(pcast(deriv_F),pcast(s), pcast(F), numberOfParams, sampling_points, 1, numberOfParams);
			handleLastError();
			////std::cout << "F'*s" << std::endl;
			//printMat(F1, 1, sampling_points);
			//Calc F(param) + F'(param)*s'
			//Fold F'(param)*s
			for(int j = 0; j < sampling_points; j++) F1[j] = -1*F[j]+F1[j];
			////std::cout << "F'*s+F:" << std::endl;
			//printMat(F1, 1, sampling_points);
			matProduct(pcast(F1), pcast(F1), pcast(Temp), 1, sampling_points, sampling_points, 1);
			handleLastError();
			float u3 = Temp[0];
			//std::cout << ";u3=" << u3 << std::endl;
		
			//calc roh

			roh = (u1-u2)/(u1-u3);
			//std::cout << "roh=" << roh << ", mu=" << mu << std::endl;
			//std::cout << "plot [0:1]";
			/*
			for(int j = 0; j < numberOfParams; j++) {
				//std::cout << "(" << param[j] << ")" << "*x**" << i;
				if(i != numberOfParams-1) std::cout << "+";
			}
			*/
			//std::cout << std::endl;
			if(roh <= 0.2) {
				//s verwerfen, mu erhöhen
				mu *= 2;
			} else  {
				for(int j = 0; j < numberOfParams; j++) param[j] = param[j] + s[j];
				if( roh >= 0.8){
					mu /= 2;
				}
			}
			//std::cout << u1/(sample_count/interpolation_count) << std::endl;
			//std::cout << "Sample: " << i << std::endl;
		} while(u1/(sample_count/interpolation_count) > 1e-3 && mu > 1e-3 && counter < 7);
		//decide if s is accepted or discarded
	}
	
	//TODO: return 0 if everything went well
	return -1;
}

#endif

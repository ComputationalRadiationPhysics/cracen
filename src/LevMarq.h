//! \file

#ifndef LEVMARQ_H
#define LEVMARQ_H

#include <thrust/device_vector.h>
#include "Types.h"
#include "UtilKernels.h"
#include "GaussJordan.h"
//TODO: remove Util.h
#include "tests/Util.h"

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


/* USER DEFINITIONS */

const int numberOfParams      = 6;

/*!
 * \brief fit function in the form 0 = f(x,y)
 */
__device__ inline float modelFunction(float x, float y, float *param) {
	return param[0] + param[1]*x + param[2]*x*x + param[3]*x*x*x + param[4]*x*x*x*x + param[5]*x*x*x*x*x - y;
}

/*
	nichtriviale Ableitungen der Summanden der Fitfunktion nach dx
	=> Konstante entfällt
*/


__device__ float deriv0(float x, float y, float *param) {
	return 0;
}

__device__  float deriv1(float x, float y, float *param) {
	return param[1];
}

__device__ float deriv2(float x, float y, float *param) {
	return 2*param[2]*x;
}

__device__ float deriv3(float x, float y, float *param) {
	return 3*param[3]*x*x;
}

__device__ float deriv4(float x, float y, float *param) {
	return 4*param[4]*x*x*x;
}

__device__ float deriv5(float x, float y, float *param) {
	return 5*param[5]*x*x*x*x;
}

typedef float (*Derivation)(float, float, float *);
__device__ const Derivation derivations[numberOfParams] = {
	&deriv0,
	&deriv1,
	&deriv2,
	&deriv3,
	&deriv4,
	&deriv5
};


/* End User Definitions */

template <unsigned int tex>
__global__ void calcF(int wave, float* param, float* F) {
	if(threadIdx.x*INTERPOLATION_COUNT < SAMPLE_COUNT) {
		F[threadIdx.x] = -1*modelFunction(threadIdx.x*INTERPOLATION_COUNT,getSample<tex>(threadIdx.x*INTERPOLATION_COUNT,wave),param);
	} else {
		F[threadIdx.x] = 0;
	}
}

template <unsigned int tex>
__global__ void calcDerivF(int wave, float param[], float mu, float deriv_F[]) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	MatrixAccess<float> deriv(numberOfParams, deriv_F);
	
	if(y < SAMPLE_COUNT/INTERPOLATION_COUNT+numberOfParams && x < numberOfParams) {
		if(y*INTERPOLATION_COUNT < SAMPLE_COUNT) {
			deriv[x][y] = (*derivations[x])(y*INTERPOLATION_COUNT,getSample<tex>(y*INTERPOLATION_COUNT,wave),param);
		} else {
			if((y - SAMPLE_COUNT/INTERPOLATION_COUNT) == x) {
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

template <unsigned int tex>
int levenberMarquardt(cudaStream_t& stream) {
	//TODO: Convert to Kernel
	//TODO: Waveformen Sequenziell abarbeiten (keine Taskparallelität)
	const int sampling_points = SAMPLE_COUNT/INTERPOLATION_COUNT;
	thrust::device_vector<float> F(sampling_points+numberOfParams), b(sampling_points+numberOfParams), s(sampling_points+numberOfParams), 
								 deriv_F((sampling_points+numberOfParams)*numberOfParams),
								 param(numberOfParams), param2(numberOfParams),
								 G((numberOfParams)*(numberOfParams)), G_inverse((numberOfParams)*(numberOfParams));
	float mu = 1;
	param[0] = 1;
	param[1] = 1;
	param[2] = 0.01;
	param[3] = 0.00001;
	param[4] = 0.0000001;
	param[5] = 0.0000000001;		
				
					 
	for(int i = 0; i < CHUNK_COUNT; i++) {
		/* Abschnitt 1 */
		//Calc F(param)
		calcF<tex><<<1,sampling_points+numberOfParams, 0, stream>>>(i, pcast(param), pcast(F));
		//for(int i = 0; i < sampling_points; i++) std::cout << F[i] << std::endl;
		handleLastError();
		//Calc F'(param)
		dim3 blockSize(32,32);
		dim3 gridSize(ceil((float) numberOfParams/32), ceil((float) sampling_points+numberOfParams/32));
		calcDerivF<tex><<<gridSize,blockSize, 0, stream>>>(i, pcast(param), mu, pcast(deriv_F));
		handleLastError();
		
		std::cout << "F': " << std::endl;
		printMat(deriv_F, sampling_points+numberOfParams, numberOfParams);
		
		/* Abschnitt 2 */
		
		//Solve minimization problem
		//calc A^T*A => G
		//transpose(pcast(deriv_F), sampling_points+numberOfParams, numberOfParams);
		//printMat(deriv_F, numberOfParams, sampling_points+numberOfParams);
		handleLastError();
		
		orthogonalMatProd(pcast(deriv_F), pcast(G), numberOfParams, sampling_points+numberOfParams);
		handleLastError();
		std::cout << "G: " << std::endl;
		printMat(G, numberOfParams, numberOfParams);
	
		//calc G^-1
		gaussJordan(pcast(G), pcast(G_inverse), numberOfParams);
		handleLastError();
		std::cout << "G^-1: " << std::endl;
		printMat(G_inverse, numberOfParams, numberOfParams);
		
		//calc A^T*F => b
		matProduct(pcast(deriv_F), pcast(F), pcast(b), sampling_points+numberOfParams, sampling_points+numberOfParams, sampling_points+numberOfParams, 1);
		handleLastError();
	
		//calc G^-1*b => s
		matProduct(pcast(G_inverse), pcast(b), pcast(s), numberOfParams, numberOfParams, numberOfParams, 1);
		handleLastError();
		
		/* Abschnitt 3 */
		
		//Fold F(param)
		matProduct(pcast(F), pcast(F), pcast(F),sampling_points, 1, 1, sampling_points);
		handleLastError();
		float u1 = F[0];
		std::cout << "u1=" << u1 << std::endl;
		
		//Calc F(param+s)
		for(int j = 0; j < numberOfParams; j++) param2[j] = param[j] + s[j];
		printMat(param, 1, numberOfParams);
		printMat(s, 1, numberOfParams);
		printMat(param2, 1, numberOfParams);
		calcF<tex><<<1,sampling_points+numberOfParams, 0, stream>>>(i, pcast(param2), pcast(F));
		handleLastError();
		printMat(F, 1, sampling_points+numberOfParams);
		//Fold F(param+s)
		matProduct(pcast(F), pcast(F), pcast(F),sampling_points, 1, 1, sampling_points);
		handleLastError();
		printMat(F, 1, sampling_points+numberOfParams);
		float u2 = F[0];
		std::cout << "u2=" << u2 << std::endl;
		
		//Calc F'(param)*s
		matProduct(pcast(deriv_F),pcast(s), pcast(F), numberOfParams, sampling_points, 1, numberOfParams);
		handleLastError();
		//Fold F'(param)*s
		matProduct(pcast(F), pcast(F), pcast(F), sampling_points, 1, 1, sampling_points);
		handleLastError();
		float u3 = F[0];
		std::cout << "u3=" << u3 << std::endl;
	
		//calc roh
		float roh = (u1-u2)/(u1-u3);
		std::cout << "roh=" << roh << std::endl;
		//decide if s is accepted or discarded
		
	}
	
	//TODO: return 0 if everything went well
	return -1;
}

#endif

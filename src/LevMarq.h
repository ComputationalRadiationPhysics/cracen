//! \file

#ifndef LEVMARQ_H
#define LEVMARQ_H

#include<thrust/device_vector.h>
#include "Types.h"

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
	if(threadIdx.x < SAMPLE_COUNT) {
		F[threadIdx.x] = -1*modelFunction(threadIdx.x,getSample<tex>(threadIdx.x,wave),param);
	} else {
		F[threadIdx.x] = 0;
	}
}

template <unsigned int tex>
__global__ void calcDerivF(int wave, float param[], float mu, float deriv_F[]) {
	if(threadIdx.x < SAMPLE_COUNT) {
		deriv_F[threadIdx.x+threadIdx.y*blockDim.x] = (*derivations[threadIdx.y])(threadIdx.x,getSample<tex>(threadIdx.x,wave),param);
	} else {
		if(threadIdx.x-SAMPLE_COUNT=threadIdx.y) {
			deriv_F[threadIdx.x+threadIdx.y*blockDim.x] = mu;
		} else {
			deriv_F[threadIdx.x+threadIdx.y*blockDim.x] = 0;
		}
	}
}

template <class c>
float* pcast(thrust::device_vector<c>& dev) {
	return thrust::raw_pointer_cast(&dev[0]);
}
template <unsigned int tex>
int levenberMarquardt(cudaStream_t& stream) {
	//Waveformen Sequenziell abarbeiten (keine Taskparallelität)
	thrust::device_vector<float> F(SAMPLE_COUNT+numberOfParams), 
								 deriv_F(SAMPLE_COUNT*numberOfParams+numberOfParams),
								 param(numberOfParams);
	float mu = 1;
	param[0] = 1;
	param[1] = 1;
	param[2] = 1;
	param[3] = 1;
	param[4] = 1;
	param[5] = 1;		
				
					 
	for(int i = 0; i < CHUNK_COUNT; i++) {
	/* Abschnitt 1 */
	//Calc F(param)
	calcF<tex><<<1,SAMPLE_COUNT+numberOfParams, 0, stream>>>(i, pcast(param), pcast(F));
	//for(int i = 0; i < SAMPLE_COUNT; i++) std::cout << F[i] << std::endl;
	
	//Calc F'(param)
	dim3 bs(SAMPLE_COUNT+numberOfParams, numberOfParams, 1);
	calcDerivF<tex><<<1,bs, 0, stream>>>(i, pcast(param), mu, pcast(deriv_F));
	for(int i=0; i < SAMPLE_COUNT+numberOfParams; i++) {
		for(int j= 0; j < numberOfParams; j++) {
		
		}
	}
	/* Abschnitt 2 */
	//Solve minimization problem
	/* Abschnitt 3 */
	//Calc F(param+s)
	//Calc F'(param)*s
	
	//Fold F(param)
	//Fold F(param+s)
	//Fold F'(param)*s
	
	//calc _roh_
	
	//decide if s is accepted or discarded
	}
	
	//TODO: return 0 if everything went well
	return -1;
}

#endif

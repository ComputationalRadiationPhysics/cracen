//! \file

#ifndef LEVMARQ_H
#define LEVMARQ_H
#define DEBUG_ENABLED

#include <stdio.h>
#include "Types.h"
#include "UtilKernels.h"
#include "GaussJordan.h"
#include "FitFunctor.h"

/*!
 * \brief getSample returns the y value of a given sample index
 * \param I sample index
 * \param INDEXDATASET index of the current dataset (GPU mode) or not used (CPU mode)
 * \return y value
 */


DEVICE float getSample(cudaTextureObject_t texObj, float I, int INDEXDATASET) {
	return tex2D<float>(texObj, I, INDEXDATASET);
}

template <class Fit, unsigned int bs>
DEVICE void calcF(cudaTextureObject_t texObj, float* param, float* F, const Window& window, const unsigned int sample_count, const unsigned int interpolation_count) {
	for(int i = 0; i*bs*interpolation_count < sample_count) {
		int x = (threadIdx.x+i*bs)*interpolation_count+offset
		if(x < window.offset+window.width) {
			float xval = static_cast<float>(x);
			float yval = getSample(texObj,x+0.5,blockIdx.x);
			F[threadIdx.x+i*bs] = -1*Fit::modelFunction(xval,yval,param);
		} else {
			F[threadIdx.x+i*bs] = 0;
		}	
	}
}



template <class Fit, unsigned int bs>
DEVICE void calcDerivF(cudaTextureObject_t texObj, float* param, float mu, float* deriv_F, const Window& window, const unsigned sample_count, const unsigned int interpolation_count) {
	for(int i = threadIdx.x*interpolation_count; i < Fit::numberOfParams*(window.width/interpolation_count+numberOfParams); i+=bs*interpolation_count) {
		int x = i%numberOfParams;
		int y = i/numberOfParams;

		if(y/interpolation_count < window.width) {
			deriv_F[x+y*Fit::numberOfParams] = Fit::derivation(x,i,getSample(texObj, i, blockIdx.x),param);
		} else {
			const float v = ((y/interpolation_count - window.width/interpolation_count) == x);
			deriv_F[x+y*Fit::numberOfParams] = mu*v;
		}
	}
}

template <class Fit, unsigned int bs>
__global__ void levMarqIt(cudaTextureObject_t texObj, FitData<Fit::numberOfParams>* results, const unsigned sample_count, const unsigned int max_window_size, const unsigned int interpolation_count) {
	const unsigned int numberOfParams = Fit::numberOfParams;

	MatrixAccess<> F(1, max_window_size+numberOfParams);
	MatrixAccess<> F1(1, max_window_size+numberOfParams);
	MatrixAccess<> b(1, numberOfParams);
	MatrixAccess<> s(1, numberOfParams);
	MatrixAccess<> A(numberOfParams, (max_window_size+numberOfParams));
	MatrixAccess<float, trans> AT = A.transpose();
	MatrixAccess<> G(numberOfParams, numberOfParams);
	MatrixAccess<> G_inverse(numberOfParams, numberOfParams);
	MatrixAccess<> param(1, numberOfParams);
	MatrixAccess<> param2(1, numberOfParams);
	MatrixAccess<> u1(1,1), u2(1,1), u3(1,1);
	MatrixAccess<float, trans> FT = F.transpose();
	MatrixAccess<float, trans> F1T = F1.transpose();

	float mu = 1, roh;
	int counter = 0;
	if(threadIdx.x < Fit::numberOfParams) param[make_uint2(0,threadIdx.x)] = 0;		
	__syncthreads();
					 
	int i = blockIdx.x;
	do {
		counter++;
		/* Abschnitt 1 */
		//Calc F(param)
		Window window = Fit::getWindow(texObj, threadIdx.x, sample_count);
		int sampling_points = window.width/interpolation_count;
		calcF<Fit, bs>(texObj, i, param.getRawPointer(), F.getRawPointer(), window, sample_count, interpolation_count);
		//Calc F'(param)
		calcDerivF<Fit, bs>(texObj, i, param.getRawPointer(), mu, A.getRawPointer(), window, sample_count, interpolation_count);

		/* Abschnitt 2 */

		//Solve minimization problem
		//calc A^T*A => G
		matProdKernel<bs>(G, AT, A);
		//calc G^-1
		gaussJordan(G_inverse, G);
		//calc A^T*F => b		
		matProdKernel<bs>(b, AT, F);
		//calc G^-1*b => s
		matProdKernel<bs>(s, G_inverse, b);
		/* Abschnitt 3 */
		//Reduce F(param)
		matProdKernel<bs>(u1, FT, F);
		
		//Calc F(param+s)
		for(int j = 0; j < numberOfParams; j++) {
			const uint2 c = make_uint2(0,j);
			param2[c] = param[c] + s[c];
		}
		
		//Fold F(param+s)
		calcF<Fit><<<1,sampling_points+numberOfParams>>>(texObj, i, param2.getRawPointer(), F1.getRawPointer(), window.offset, sample_count, interpolation_count);
		matProdKernel<bs>(u2, F1T, F1);

		//Calc F'(param)*s
		matProdKernel<bs>(F1, A, s);

		//Calc F(param) + F'(param)*s'
		//Fold F'(param)*s
		for(int j = 0; j < sampling_points; j++) {
			uint2 c = make_uint2(0,j);
			F1[c] = -1*F[c]+F1[c];
		}
		matProdKernel<bs>(u3, F1T, F1);

		//calc roh
		roh = (u1[make_uint2(0,0)]-u2[make_uint2(0,0)])/(u1[make_uint2(0,0)]-u3[make_uint2(0,0)]);

		//decide if s is accepted or discarded
		if(roh <= 0.2) {
			//s verwerfen, mu erhöhen
			mu *= 2;
		} else  {
			for(uint2 j = make_uint2(0,0); j.y < numberOfParams; j.y++) {
				param[j] = param[j] + s[j];
			} 
			if( roh >= 0.8){
				mu /= 2;
			}
		}
	} while(u1[make_uint2(0,0)]/(sample_count/interpolation_count) > 1e-5 && mu > 1e-5 && counter < 25);
	
	for(uint2 j = make_uint2(0,0); j.y < numberOfParams; j.y++) {
		float p = param[j];
		results[waveform].param[j.y] = p;
	}
	
	F.finalize();
	F1.finalize();
	b.finalize();
	s.finalize();
	A.finalize();
	G.finalize();
	G_inverse.finalize();
	param.finalize();
	param2.finalize();
	u1.finalize();
	u2.finalize();
	u3.finalize();

	return;
}

template <class Fit>
__global__ void dispatch(cudaTextureObject_t texObj, FitData<Fit::numberOfParams>* results, const unsigned sample_count, const unsigned int max_window_size, const unsigned int interpolation_count) {
	//Start Levenberg Marquard algorithm on a single date
	//printf("Dispatch %i\n", threadIdx.x);
	levMarqIt<Fit><<<1,1>>>(texObj, results, sample_count, max_window_size, threadIdx.x, interpolation_count);
}

template <class Fit>
int levenbergMarquardt(cudaStream_t& stream, cudaTextureObject_t texObj, FitData<Fit::numberOfParams>* results, const unsigned sample_count, const unsigned int max_window_size, const unsigned int chunk_count, const unsigned int interpolation_count) {
	dim3 gs(1,1);
	dim3 bs(chunk_count,1);
	//printf("LevMarq %i\n", chunk_count);
	//FitData<numberOfParams>* results = new FitData<numberOfParams>[sample_count];
	dispatch<Fit><<<gs,bs, 0, stream>>>(texObj, results, sample_count, max_window_size,interpolation_count);
	handleLastError();
	return 0;

}

#endif

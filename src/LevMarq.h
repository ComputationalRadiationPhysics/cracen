//! \file

#ifndef LEVMARQ_H
#define LEVMARQ_H
//#define DEBUG_ENABLED

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


DEVICE float getSample(const cudaTextureObject_t texObj, const float I, const int INDEXDATASET) {
	return tex2D<float>(texObj, I+0.5, INDEXDATASET);
}

template <class Fit, unsigned int bs>
DEVICE void calcF(const cudaTextureObject_t texObj, const float* const param, float * const F, const Window& window, const unsigned int sample_count, const unsigned int interpolation_count) {
	for(int i = threadIdx.x; i < window.width+Fit::numberOfParams; i++) {
		const int x = i*interpolation_count+window.offset;
		if(i < window.width) {
			const float xval = static_cast<float>(x);
			const float yval = getSample(texObj,x,blockIdx.x);
			F[i] = -1*Fit::modelFunction(xval,yval,param);
		} else {
			F[i] = 0;
		}	
	}
	__syncthreads();
}



template <class Fit, unsigned int bs, class MatrixAccess>
DEVICE void calcDerivF(const cudaTextureObject_t texObj, const float * const param, const float mu, MatrixAccess& deriv_F, const Window& window, const unsigned sample_count, const unsigned int interpolation_count) {
	for(int i = threadIdx.x; i < Fit::numberOfParams*(window.width+Fit::numberOfParams); i+=bs) {
		const int x = i%Fit::numberOfParams;
		const int y = i/Fit::numberOfParams;
		
		if(y < window.width) {
			deriv_F[make_uint2(x,y)] = Fit::derivation(x,window.offset+y*interpolation_count,getSample(texObj, window.offset+y*interpolation_count, blockIdx.x),param);
		} else if(y < window.width + Fit::numberOfParams) {
			if((y - window.width) == x) {
				deriv_F[make_uint2(x,y)] = mu;
			} else {
				deriv_F[make_uint2(x,y)] = 0;
			}
		}
	}
	__syncthreads();
}

template <class Fit, unsigned int bs>
__global__ void levMarqIt(const cudaTextureObject_t texObj, FitData* const results, const unsigned sample_count, const unsigned int max_window_size, const unsigned int interpolation_count) {
	const unsigned int numberOfParams = Fit::numberOfParams;
	__shared__ MatrixAccess<> G,G_inverse,u1,u2,u3,AT,FT,F1T;
	__shared__ MatrixAccess<float, trans> F,F1,b,s,A,param,param2,param_last_it;
	__shared__ float sleft[bs];
	__shared__ float b_shared[numberOfParams], s_shared[numberOfParams], G_shared[numberOfParams*numberOfParams], 
					 G_inverse_shared[numberOfParams*numberOfParams],
					 param_shared[numberOfParams], param2_shared[numberOfParams], param_last_it_shared[numberOfParams],
					 u1_shared[1], u2_shared[1], u3_shared[1];
					 
	__shared__ bool finished;
	if(threadIdx.x == 0) {
		F = MatrixAccess<float, trans>(max_window_size+numberOfParams, 1);
		b = MatrixAccess<float, trans>(b_shared, numberOfParams, 1);
		s = MatrixAccess<float, trans>(s_shared, numberOfParams, 1);
		A = MatrixAccess<float, trans>(max_window_size+numberOfParams, numberOfParams);
		F1 = MatrixAccess<float, trans>(max_window_size+numberOfParams, 1);
		AT = A.transpose();
		G = MatrixAccess<>(G_shared, numberOfParams, numberOfParams);
		G_inverse = MatrixAccess<>(G_inverse_shared, numberOfParams, numberOfParams);
		param = MatrixAccess<float, trans>(param_shared, numberOfParams, 1);
		param2 = MatrixAccess<float, trans>(param2_shared, numberOfParams, 1);
		param_last_it = MatrixAccess<float, trans>(param_last_it_shared, numberOfParams, 1);
		u1 = MatrixAccess<>(u1_shared, 1,1);
		u2 = MatrixAccess<>(u2_shared, 1,1);
		u3 = MatrixAccess<>(u3_shared, 1,1);
		FT = F.transpose();
		F1T = F1.transpose();
	}	
	__syncthreads();
	float mu = 1, roh;
	int counter = 0;
	if(threadIdx.x < Fit::numberOfParams) param[make_uint2(0,threadIdx.x)] = 0;		
	__syncthreads();
					 
	do {
		counter++;
		/* Abschnitt 1 */
		//Calc F(param)
		
		const Window window = Fit::getWindow(texObj, threadIdx.x, sample_count);
		const int sampling_points = window.width/interpolation_count;
		calcF<Fit, bs>(texObj, param.getRawPointer(), F.getRawPointer(), window, sample_count, interpolation_count);
		//Calc F'(param)
		calcDerivF<Fit, bs>(texObj, param.getRawPointer(), mu, A, window, sample_count, interpolation_count);
		/* Abschnitt 2 */

		//Solve minimization problem
		//calc A^T*A => G
		matProdKernel<bs>(G, AT, A, sleft);
		//calc G^-1
		gaussJordan<Fit,bs>(G_inverse, G);
		//calc A^T*F => b		
		matProdKernel<bs>(b, AT, F, sleft);
		//calc G^-1*b => s
		matProdKernel<bs>(s, G_inverse, b, sleft);
		/* Abschnitt 3 */
		//Reduce F(param)
		matProdKernel<bs>(u1, FT, F, sleft);
		
		//Calc F(param+s)
		const uint2 c = make_uint2(0,threadIdx.x);
		if(threadIdx.x < numberOfParams) param2[c] = param[c] + s[c];
		__syncthreads();
		
		//Fold F(param+s)
		calcF<Fit, bs>(texObj, param2.getRawPointer(), F1.getRawPointer(), window, sample_count, interpolation_count);
		matProdKernel<bs>(u2, F1T, F1, sleft);

		//Calc F'(param)*s
		matProdKernel<bs>(F1, A, s, sleft);

		//Calc F(param) + F'(param)*s'
		//Fold F'(param)*s
		for(int j = threadIdx.x; j < sampling_points; j+=bs) {
			uint2 c = make_uint2(0,j);
			F1[c] = -1*F[c]+F1[c];
		}
		if(threadIdx.x == 0) finished = true;
		__syncthreads();
		matProdKernel<bs>(u3, F1T, F1, sleft);

		//calc roh
		roh = (u1[make_uint2(0,0)]-u2[make_uint2(0,0)])/(u1[make_uint2(0,0)]-u3[make_uint2(0,0)]);
		//decide if s is accepted or discarded
		if(roh <= 0.2) {
			//s verwerfen, mu erhöhen
			mu *= 2;
			if(threadIdx.x == 0) finished = false;
		} else  {
				uint2 j = make_uint2(0,threadIdx.x);
				if(threadIdx.x < numberOfParams) {
					param[j] = param[j] + s[j];
					if(s[j] > 1e-5) finished = false;
				}
			if( roh >= 0.8){
				mu /= 2;
			}
		}
		__syncthreads();
	} while(!finished && counter < 25);
	
		if(threadIdx.x < numberOfParams) {
			const float p = param[make_uint2(0,threadIdx.x)];
			results[blockIdx.x].param[threadIdx.x] = p;
		}
	
	if(threadIdx.x == 0) {
		F.finalize();
		F1.finalize();
		A.finalize();
	}
	
	return;
}

template <class Fit>
int levenbergMarquardt(cudaStream_t& stream, cudaTextureObject_t texObj, FitData* results, const unsigned sample_count, const unsigned int max_window_size, const unsigned int chunk_count, const unsigned int interpolation_count) {
	const unsigned int bsx = 256;
	const dim3 gs(chunk_count,1);
	const dim3 bs(bsx,1);
	levMarqIt<Fit,bsx><<<gs,bs, 0, stream>>>(texObj, results, sample_count, max_window_size,interpolation_count);
	handleLastError();
	return 0;

}

#endif

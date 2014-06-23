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

template <class Fit>
__global__ void calcF(cudaTextureObject_t texObj, int wave, float* param, float* F, unsigned int offset, const unsigned int sample_count, const unsigned int interpolation_count) {
	if(threadIdx.x*interpolation_count < sample_count) {
		float x = threadIdx.x*interpolation_count+offset;
		float y = getSample(texObj,x+0.5,wave);
		F[threadIdx.x] = -1*Fit::modelFunction(x,y,param);
	} else {
		F[threadIdx.x] = 0;
	}
}



template <class Fit>
__global__ void calcDerivF(cudaTextureObject_t texObj, int wave, float* param, float mu, float* deriv_F, const unsigned int offset, const unsigned sample_count, const unsigned int interpolation_count) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if(!(y < sample_count/interpolation_count+Fit::numberOfParams && x < Fit::numberOfParams)) return;
	if(y*interpolation_count < sample_count) {
		deriv_F[x+y*Fit::numberOfParams] = Fit::derivation(x,y*interpolation_count+offset,getSample(texObj, y*interpolation_count+offset,wave),param);
	} else {
		const float v = ((y - sample_count/interpolation_count) == x);
		deriv_F[x+y*Fit::numberOfParams] = mu*v;
	}

}

template <class Fit>
__global__ void levMarqIt(cudaTextureObject_t texObj, FitData<Fit::numberOfParams>* results, const unsigned sample_count, const unsigned int max_window_size, const unsigned int waveform, const unsigned int interpolation_count) {
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

	float mu;
	for(uint2 j = make_uint2(0,0); j.y < numberOfParams; j.y++) {
		param[j] = 0;		
	}
					 
	int i = waveform;
	float roh;
	mu = 1;
	int counter = 0;
	do {
		counter++;
		/* Abschnitt 1 */
		//Calc F(param)
		Window window = Fit::getWindow(texObj, threadIdx.x, sample_count);
		int sampling_points = window.width/interpolation_count;
		/** \TODO Generischer Implementieren für >1024 Werte */
		dim3 gs(1,1);
		dim3 bs(sampling_points+numberOfParams,1);
		calcF<Fit><<<gs,bs>>>(texObj, i, param.getRawPointer(), F.getRawPointer(), window.offset, sample_count, interpolation_count);
		//printf("F:\n");
		//printMat(F);
		//for(int j = 0; j < sampling_points; j++) //std::cout << "f("<< j*interpolation_count << ")=" << F[j] << std::endl;
		////std::cout << "F: " << std::endl;
		//printMat(F, 1, sampling_points+numberOfParams);
		//handleLastError();
		//Calc F'(param)
		dim3 blockSize(32,32); /** TODO: Blocksize reduzieren bs(256,1), A drehen*/
		dim3 gridSize(ceil((float) numberOfParams/32.f), ceil(static_cast<float>(sampling_points+numberOfParams)/32.f));
		calcDerivF<Fit><<<gridSize,blockSize>>>(texObj, i, param.getRawPointer(), mu, A.getRawPointer(), window.offset, sample_count, interpolation_count);
		//handleLastError();
		////std::cout << "F': " << std::endl;
		//printMat(deriv_F, sampling_points+numberOfParams, numberOfParams);

		/* Abschnitt 2 */

		//Solve minimization problem
		//calc A^T*A => G
		MatMul(G, AT, A);
		//printf("G:\n");
		//printMat(G);
		//calc G^-1
		
		gaussJordan(G_inverse, G);
		//printf("G_inverse:\n");
		//printMat(G_inverse);
		//calc A^T*F => b
		
		MatMul(b, AT, F);
	
		//printf("b:\n");
		//printMat(b);
		
		//calc G^-1*b => s
		MatMul(s, G_inverse, b);
		//printf("s:\n");
		//printMat(s);
		/* Abschnitt 3 */

		//Fold F(param)
		MatMul(u1, FT, F);
		
		//printMat(u1);
		//Calc F(param+s)
		for(int j = 0; j < numberOfParams; j++) {
			const uint2 c = make_uint2(0,j);
			param2[c] = param[c] + s[c];
		}
		
		//Fold F(param+s)
		calcF<Fit><<<1,sampling_points+numberOfParams>>>(texObj, i, param2.getRawPointer(), F1.getRawPointer(), window.offset, sample_count, interpolation_count);
		MatMul(u2, F1T, F1);

		//Calc F'(param)*s
		MatMul(F1, A, s);

		//Calc F(param) + F'(param)*s'
		//Fold F'(param)*s
		for(int j = 0; j < sampling_points; j++) {
			uint2 c = make_uint2(0,j);
			F1[c] = -1*F[c]+F1[c];
		}
		MatMul(u3, F1T, F1);

		//calc roh
		//printMat(u1);
		//printf("u1=%f, u2=%f, u3=%f\n", u1[make_uint2(0,0)], u2[make_uint2(0,0)], u3[make_uint2(0,0)]);
		roh = (u1[make_uint2(0,0)]-u2[make_uint2(0,0)])/(u1[make_uint2(0,0)]-u3[make_uint2(0,0)]);
		//printf("roh=%f, mu=%f\n", roh, mu);
		/*
		for(uint2 j = make_uint2(0,0); j.y < numberOfParams; j.y++) {
			float p = param[j];
			printf("(%f)*x**%i",p,j.y);
			if(j.y != numberOfParams-1) printf("+");
		}
		//printf("\n");
		*/
		//std::cout << std::endl;
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
		//printf("param:\n");
		//printMat(param);		
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

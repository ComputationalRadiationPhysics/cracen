//! \file

#ifndef LEVMARQ_H
#define LEVMARQ_H
#define DEBUG_ENABLED

#include <cstdio>
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
	
	if(!(y < sample_count/interpolation_count+Fit::numberOfParams() && x < Fit::numberOfParams())) return;
	if(y*interpolation_count < sample_count) {
		deriv_F[x+y*Fit::numberOfParams()] = Fit::derivation(x,y*interpolation_count+offset,getSample<tex>(y*interpolation_count+offset,wave),param);
	} else {
		const float v = ((y - sample_count/interpolation_count) == x);
		deriv_F[x+y*Fit::numberOfParams()] = mu*v;
	}

}

//Fit should be a FitFunctor
template <class Fit, unsigned int tex>
__global__ void levMarqIt(const unsigned sample_count, const unsigned int max_window_size, const unsigned int waveform, const unsigned int interpolation_count) {
	unsigned int numberOfParams = Fit::numberOfParams();
	
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
	
	float mu;
	for(int j = 0; j < numberOfParams; j++) {
		uint2 c = make_uint2(0,j);
		param[c] = 0;		
	}
					 
	int i = waveform;
	////std::cout << "Sample: " << i << std::endl;
	float roh;
	mu = 1;
	int counter = 0;
	do {
		counter++;
		////std::cout << "param:" << std::endl;
		//printMat(param, 1 , numberOfParams);
		/* Abschnitt 1 */
		//Calc F(param)
		Window window = Fit::getWindow(threadIdx.x, sample_count);
		int sampling_points = window.width/interpolation_count;
		/** \TODO Generischer Implementieren für >1024 Werte */
		calcF<Fit, tex><<<1,sampling_points+numberOfParams>>>(i, param.getRawPointer(), F.getRawPointer(), window.offset, sample_count, interpolation_count);
		//for(int j = 0; j < sampling_points; j++) //std::cout << "f("<< j*interpolation_count << ")=" << F[j] << std::endl;
		////std::cout << "F: " << std::endl;
		//printMat(F, 1, sampling_points+numberOfParams);
		//handleLastError();
		//Calc F'(param)
		dim3 blockSize(32,32); /** TODO: Blocksize reduzieren */
		dim3 gridSize(ceil((float) numberOfParams/32.f), ceil(static_cast<float>(sampling_points+numberOfParams)/32.f));
		calcDerivF<Fit, tex><<<gridSize,blockSize>>>(i, param.getRawPointer(), mu, A.getRawPointer(), window.offset, sample_count, interpolation_count);
		//handleLastError();
	
		////std::cout << "F': " << std::endl;
		//printMat(deriv_F, sampling_points+numberOfParams, numberOfParams);
	
		/* Abschnitt 2 */
	
		//Solve minimization problem
		//calc A^T*A => G
		MatMul(G, AT, A);
		//calc G^-1
		gaussJordan(G_inverse, G);

		//calc A^T*F => b
		MatMul(b, AT, F);
		
		//calc G^-1*b => s
		MatMul(s, G_inverse, b);
		
		/* Abschnitt 3 */
	
		//Fold F(param)
		MatMul(u1, F, F);
		//handleLastError();
		//std::cout << "u1=" << u1;
	
		//Calc F(param+s)

		for(int j = 0; j < numberOfParams; j++) {
			uint2 c = make_uint2(0,j);
			param2[c] = param[c] + s[c];
		}
		//Fold F(param+s)
		calcF<Fit, tex><<<1,sampling_points+numberOfParams>>>(i, param2.getRawPointer(), F1.getRawPointer(), window.offset, sample_count, interpolation_count);
		MatMul(u2, F1, F1);
		//handleLastError();
		//std::cout << ";u2=" << u2;
	

		//Calc F'(param)*s
		MatMul(F1, A, s);

		//handleLastError();
		////std::cout << "F'*s" << std::endl;
		//printMat(F1, 1, sampling_points);
		//Calc F(param) + F'(param)*s'
		//Fold F'(param)*s
		for(int j = 0; j < sampling_points; j++) {
			uint2 c = make_uint2(0,j);
			F1[c] = -1*F[c]+F1[c];
		}
		////std::cout << "F'*s+F:" << std::endl;
		//printMat(F1, 1, sampling_points);

		MatMul(u3, F1, F1);
		//handleLastError();
		//std::cout << ";u3=" << u3 << std::endl;
	
		//calc roh

		roh = (u1[make_uint2(0,0)]-u2[make_uint2(0,0)])/(u1[make_uint2(0,0)]-u3[make_uint2(0,0)]);
		//std::cout << "roh=" << roh << ", mu=" << mu << std::endl;
		//std::cout << "plot [0:1]";
		/*
		for(int j = 0; j < numberOfParams; j++) {
			//std::cout << "(" << param[j] << ")" << "*x**" << i;
			if(j != numberOfParams-1) std::cout << "+";
		}
		*/
		//std::cout << std::endl;
		if(roh <= 0.2) {
			//s verwerfen, mu erhöhen
			mu *= 2;
		} else  {
			for(uint2 j = make_uint2(0,0); j.x < numberOfParams; j.x++) {
				param[j] = param[j] + s[j];
			} 
			if( roh >= 0.8){
				mu /= 2;
			}
		}
		//std::cout << u1/(sample_count/interpolation_count) << std::endl;
		//std::cout << "Sample: " << i << std::endl;
	//decide if s is accepted or discarded
	if(waveform == 0) {
		for(uint2 j = make_uint2(0,0); j.x < numberOfParams; j.x++) {
			float p = param[j];
			printf("(%f)*x**%i",p,j);
			if(j.x != numberOfParams-1) printf("+");
		}
		printf("\n");
	}
	} while(u1[make_uint2(0,0)]/(sample_count/interpolation_count) > 1e-3 && mu > 1e-3 && counter < 25);
	/*	
	for(int j = 0; j < numberOfParams; j++) {
		float p = param[j];
		printf("(%f)*x**%i",p,j);
		if(j != numberOfParams-1) printf("+");
	}
	printf("\n");
	*/
	
	/*
	if(counter >= 25) return;
	if(*u1/(sample_count/interpolation_count) > 1e-3 ) return;
	*/
	return;
}

template <class Fit, unsigned int tex>
__global__ void dispatch(const unsigned sample_count, const unsigned int max_window_size, const unsigned int interpolation_count) {
	//Start Levenberg Marquard algorithm on a single date
	printf("%i\n", threadIdx.x);
	levMarqIt<Fit,tex><<<1,1>>>(sample_count, max_window_size, threadIdx.x, interpolation_count);
}

template <class Fit, unsigned int tex>
int levenbergMarquardt(cudaStream_t& stream, fitData* results, const unsigned sample_count, const unsigned int max_window_size, const unsigned int chunk_count, const unsigned int interpolation_count) {
	dim3 gs(1,1,1);
	dim3 bs(chunk_count,1,1);
	results = new fitData<Fit>[sample_count];
	dispatch<Fit,tex><<<gs,bs, 0, stream>>>(results, sample_count,max_window_size,interpolation_count);
	
	return 0;
}
#endif

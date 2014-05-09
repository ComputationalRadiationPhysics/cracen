//! \file

#ifndef LEVMARQ_H
#define LEVMARQ_H

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
	MatrixAccess<float> deriv(Fit::numberOfParams(), deriv_F);
	
	if(!(y < sample_count/interpolation_count+Fit::numberOfParams() && x < Fit::numberOfParams())) return;
	if(y*interpolation_count < sample_count) {
		deriv[x][y] = Fit::derivation(x,y*interpolation_count+offset,getSample<tex>(y*interpolation_count+offset,wave),param);
	} else {
		const float v = (y - sample_count/interpolation_count) == x);
		deriv[x][y] = mu*v;
	}

}

//Fit should be a FitFunctor
template <class Fit, unsigned int tex>
__global__ void levMarqIt(const unsigned sample_count, const unsigned int max_window_size, const unsigned int waveform, const unsigned int interpolation_count) {
	unsigned int numberOfParams = Fit::numberOfParams();
	//TODO: Speicher freigeben
	float* F 		 = new float[max_window_size+numberOfParams];
	float* b 		 = new float[numberOfParams];
	float* s 		 = new float[numberOfParams];
	float* deriv_F 	 = new float[(max_window_size+numberOfParams)*numberOfParams];
	float* param	 = new float[numberOfParams];
	float* param2	 = new float[numberOfParams];
	float* G		 = new float[(numberOfParams)*(numberOfParams)];
	float* G_inverse = new float[(numberOfParams)*(numberOfParams)];	
	float* F1 		 = new float[max_window_size+numberOfParams];
	float* u1 		 = new float;
	float* u2 		 = new float;
	float* u3 		 = new float;	
	
	float mu;
	for(int i = 0; i < numberOfParams; i++) param[i] = 0;		
					 
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
		calcF<Fit, tex><<<1,sampling_points+numberOfParams>>>(i, param, F, window.offset, sample_count, interpolation_count);
		//for(int j = 0; j < sampling_points; j++) //std::cout << "f("<< j*interpolation_count << ")=" << F[j] << std::endl;
		////std::cout << "F: " << std::endl;
		//printMat(F, 1, sampling_points+numberOfParams);
		//handleLastError();
		//Calc F'(param)
		dim3 blockSize(32,32); /** TODO: Blocksize reduzieren */
		dim3 gridSize(ceil((float) numberOfParams/32.f), ceil(static_cast<float>(sampling_points+numberOfParams)/32.f));
		calcDerivF<Fit, tex><<<gridSize,blockSize>>>(i, param, mu, deriv_F, window.offset, sample_count, interpolation_count);
		//handleLastError();
	
		////std::cout << "F': " << std::endl;
		//printMat(deriv_F, sampling_points+numberOfParams, numberOfParams);
	
		/* Abschnitt 2 */
	
		//Solve minimization problem
		//calc A^T*A => G
		/** TODO Mat Wrapper */
		transpose(deriv_F, sampling_points+numberOfParams, numberOfParams);
		//printMat(deriv_F, numberOfParams, sampling_points+numberOfParams);
		//handleLastError();
		/** TODO: Blocksize reduzieren, linearisieren */
		orthogonalMatProd(deriv_F, G, numberOfParams, sampling_points+numberOfParams);
		//handleLastError();
		////std::cout << "G: " << std::endl;
		//printMat(G, numberOfParams, numberOfParams);

		//calc G^-1
		gaussJordan(G, G_inverse, numberOfParams);
		//handleLastError();
		////std::cout << "G^-1: " << std::endl;
		//printMat(G_inverse, numberOfParams, numberOfParams);
	
		//calc A^T*F => b
		matProduct(deriv_F, F, b, numberOfParams, sampling_points+numberOfParams, sampling_points+numberOfParams, 1);
		//handleLastError();
		////std::cout << "b" << std::endl;
		//printMat(b,1, numberOfParams);
		
		//calc G^-1*b => s
		matProduct(G_inverse, b, s, numberOfParams, numberOfParams, numberOfParams, 1);
		//handleLastError();
		
		////std::cout << "s" << std::endl;
		//printMat(s,1, numberOfParams);
		
		/* Abschnitt 3 */
	
		//Fold F(param)
		matProduct(F, F, u1, 1, sampling_points+numberOfParams, sampling_points+numberOfParams, 1);
		//handleLastError();
		//std::cout << "u1=" << u1;
	
		//Calc F(param+s)

		for(int j = 0; j < numberOfParams; j++) param2[j] = param[j] + s[j];
		//Fold F(param+s)
		calcF<Fit, tex><<<1,sampling_points+numberOfParams>>>(i, param2, F1, window.offset, sample_count, interpolation_count);
		matProduct(F1, F1, u2,1, sampling_points, sampling_points, 1);
		//handleLastError();
		//std::cout << ";u2=" << u2;
	

		//Calc F'(param)*s
		transpose(deriv_F, numberOfParams, sampling_points+numberOfParams);
		matProduct(deriv_F, s, F1, sampling_points, numberOfParams, numberOfParams, 1);

		//handleLastError();
		////std::cout << "F'*s" << std::endl;
		//printMat(F1, 1, sampling_points);
		//Calc F(param) + F'(param)*s'
		//Fold F'(param)*s
		for(int j = 0; j < sampling_points; j++) F1[j] = -1*F[j]+F1[j];
		////std::cout << "F'*s+F:" << std::endl;
		//printMat(F1, 1, sampling_points);

		matProduct(F1, F1, u3, 1, sampling_points, sampling_points, 1);
		//handleLastError();
		//std::cout << ";u3=" << u3 << std::endl;
	
		//calc roh

		roh = (*u1-*u2)/(*u1-*u3);
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
			for(int j = 0; j < numberOfParams; j++) param[j] = param[j] + s[j];
			if( roh >= 0.8){
				mu /= 2;
			}
		}
		//std::cout << u1/(sample_count/interpolation_count) << std::endl;
		//std::cout << "Sample: " << i << std::endl;
	//decide if s is accepted or discarded
	if(waveform == 0) {
		for(int j = 0; j < numberOfParams; j++) {
			float p = param[j];
			printf("(%f)*x**%i",p,j);
			if(j != numberOfParams-1) printf("+");
		}
		printf("\n");
	}
	} while(*u1/(sample_count/interpolation_count) > 1e-3 && mu > 1e-3 && counter < 25);
	/*	
	for(int j = 0; j < numberOfParams; j++) {
		float p = param[j];
		printf("(%f)*x**%i",p,j);
		if(j != numberOfParams-1) printf("+");
	}
	printf("\n");
	*/
	delete F;
	delete b;
	delete s;
	delete deriv_F;
	delete param;
	delete param2;
	delete G;
	delete G_inverse;	
	delete F1;
	delete u1;
	delete u2;
	delete u3;	
	
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
int levenbergMarquardt(cudaStream_t& stream, const unsigned sample_count, const unsigned int max_window_size, const unsigned int chunk_count, const unsigned int interpolation_count) {
	dim3 gs(1,1,1);
	dim3 bs(chunk_count,1,1);
	printf("asd");
	dispatch<Fit,tex><<<gs,bs, 0, stream>>>(sample_count,max_window_size,interpolation_count);
	
	return 0;
}
#endif

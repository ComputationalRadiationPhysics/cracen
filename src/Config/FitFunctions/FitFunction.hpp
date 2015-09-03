#ifndef FITFUNCTION_HPP
#define FITFUNCTION_HPP

#include "FitFunctor.hpp"

/* User definitions */

#include <cstdio>

/*!
 * \brief fit function in the form 0 = f(x,y)
 */
 
#ifdef __CUDACC__
extern DEVICE float getSample(const cudaTextureObject_t texObj, const float I, const int INDEXDATASET);
#endif

class Gauss:public FitFunctor<4> {
	#ifdef __CUDACC__
public:
	// y = p0*e^(-1* ((x-p1)/p3)^2) + p2
	static DEVICE float modelFunction(const float x, const float y, const float * const param) {
		const float exp =  (x-param[1])/param[3];
		const float e = expf(-1.f*exp*exp);
		return param[0]*e + param[2]- y;
	}
	static DEVICE float derivation(const int param, const float x, const float y, const float * const params) {
		const float exp =  (x-params[1])/params[3];
		const float e = expf(-1.f*exp*exp);
		
		switch(param) {
			case 0: 
				return e;
			case 1:
				return 2.f*params[0]*e*(x-params[1])/(params[3]*params[3]);
			case 3:
				return 2.f*params[0]*e*(-1.f*params[1]+x)/(params[3]*params[3]);
			default:
				return 1.f;
		}
	}
	
	template <class MatrixAccess>
	static DEVICE void guessParams(const cudaTextureObject_t texObj, MatrixAccess& param, const Window& window) {
		int max_x = 0;
		if(threadIdx.x == 0) {
			float max = RANGE_MINIMUM;
			for(int i = window.offset; i < window.offset+window.width; i++) {
				if(getSample(texObj, i, blockIdx.x) > max) {
					max = getSample(texObj, i, blockIdx.x);
					max_x = i;
				}
			}
			param[make_uint2(0,0)] = GAUSS_PARAM0;
			param[make_uint2(0,1)] = static_cast<float>(max_x);
			param[make_uint2(0,2)] = GAUSS_PARAM2;
			param[make_uint2(0,3)] = GAUSS_PARAM3;
		}
		__syncthreads();
	}
	static DEVICE void getWindow(const cudaTextureObject_t texObj, Window& window, const int dataset, const int sample_count) {
		window.set(0, sample_count);
	}
	#endif
};

template <unsigned int order>
class Polynom:public FitFunctor<order+1> {
public:
	#ifdef __CUDACC__
	static DEVICE float modelFunction(const float x, const float y, const float * const param) {
		float res = 0;
		for(int i = 0; i <= order; i++) {
			if(i%2 == 0) res += param[i]*pow(x,i);
			else		 res += param[i]*x*pow(x,i-1);		
		}
		res -= y;
		return res;
	}
	static DEVICE float derivation(const int param, const float x, const float y, const float * const params) {
		//float res;
		//if(param%2 == 1) res = pow(x,static_cast<float>(param));
		//else                               res = x*pow(x,param-1);		
		return pow(x, param); 
		//return res;
	}

	static DEVICE void getWindow(const cudaTextureObject_t texObj, Window& window, const int dataset, const int sample_count) {
		window.set(0, sample_count);
	}
	
	template <class MatrixAccess>
	static DEVICE void guessParams(const cudaTextureObject_t texObj, MatrixAccess& param, const Window& window) {
		if(threadIdx.x == 0) {
			
			float max = RANGE_MINIMUM;
			int pos = 0;
			for(int i = 0; i < window.width; i++) {
				if(getSample(texObj, i, blockIdx.x) > max) {
					max = getSample(texObj, i, blockIdx.x);
					pos = i;
				}
			}
			const float a = pos;
			const float b = max;
			const float c = -1; //(getSample(texObj, window.offset, blockIdx.x)-b)/((window.offset-a)*(window.offset-a));
			param[make_uint2(0,0)] = b+c*a*a;
			param[make_uint2(0,1)] = -2.f*a*c;
			param[make_uint2(0,2)] = c;
			for(int i = 3; i <= order; i++) param[make_uint2(0,i)] = 1.f/pow(10.f,-1*i);
			
			/*
			param[make_uint2(0,0)] = 0;
			param[make_uint2(0,1)] = 0;
			param[make_uint2(0,2)] = 0;
			param[make_uint2(0,3)] = 0;
			*/
		}
		__syncthreads();
	}
	#endif
};

template <unsigned int order>
class WindowPolynom:public Polynom<order> {
	#ifdef __CUDACC__
public:
	static DEVICE void getWindow(const cudaTextureObject_t texObj, Window& window, const int dataset, const int sample_count) {
		int pos = 0;
		if(threadIdx.x == 0) {
			float max = RANGE_MINIMUM;
			for(int i = 0; i < sample_count; i++) {
				if(getSample(texObj, i, blockIdx.x) > max) {
					max = getSample(texObj, i, blockIdx.x);
					pos = i;
				}
			}
			
			if(pos > sample_count - window_size/2) {
				window.set(sample_count-window_size-1, window_size);
			}
			if(pos < window_size/2) {
				window.set(0, window_size);
			}
			window.set(pos-window_size/2, window_size);
		}
		__syncthreads();
	}
	template <class MatrixAccess>
	static DEVICE void guessParams(const cudaTextureObject_t texObj, MatrixAccess& param, const Window& window) {
		if(threadIdx.x == 0) {
			const float a = window.offset+window_size/2;
			const float b = getSample(texObj, a, blockIdx.x);
			const float c = -1; //(getSample(texObj, window.offset, blockIdx.x)-b)/((window.offset-a)*(window.offset-a));
			param[make_uint2(0,0)] = b+c*a*a;
			param[make_uint2(0,1)] = -2.f*a*c;
			param[make_uint2(0,2)] = c;
			for(int i = 3; i <= order; i++) param[make_uint2(0,i)] = 1.f/pow(10.f,-1*i);
		}
		__syncthreads();
	}
	#endif
};

/* End user definitions*/

#endif

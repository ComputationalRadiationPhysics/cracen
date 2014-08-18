#ifndef FITFUNCTION_HPP
#define FITFUNCTION_HPP

#include "FitFunctor.hpp"
/* User definitions */

#include <cstdio>

/*!
 * \brief fit function in the form 0 = f(x,y)
 */
 
#ifdef __CUDACC__
DEVICE float getSample(const cudaTextureObject_t texObj, const float I, const int INDEXDATASET) {
	return tex2D<float>(texObj, I+0.5, static_cast<float>(INDEXDATASET)+0.5);
}
#endif

class Gauss:public FitFunctor<4> {
	#ifdef __CUDACC__
public:
	static DEVICE float modelFunction(const float x, const float y, const float * const param) {
		const float exp =  (x-param[1])/param[3];
		const float e = expf(-1*exp*exp);
		return param[0]*e + param[2]- y;
	}
	static DEVICE float derivation(const int param, const float x, const float y, const float * const params) {
		const float exp =  (x-params[1])/params[3];
		const float e = expf(-1*exp*exp);
		
		switch(param) {
			case 0: 
				return e;
			case 1:
				return 2*params[0]*e*(x-params[1])/(params[3]*params[3]);
			case 3:
				return 2*params[0]*e*(-1*params[1]+x)/(params[3]*params[3]);
			default:
				return 1;
		}
	}
	
	template <class MatrixAccess>
	static DEVICE void guessParams(const cudaTextureObject_t texObj, MatrixAccess& param, const Window& window) {
		int max_x = 0;
		if(threadIdx.x == 0) {
			float max = -40000;
			for(int i = window.offset; i < window.offset+window.width; i++) {
				if(getSample(texObj, i, blockIdx.x) > max) {
					max = getSample(texObj, i, blockIdx.x);
					max_x = i;
				}
			}
			param[make_uint2(0,0)] = 20000;
			param[make_uint2(0,1)] = static_cast<float>(max_x);
			param[make_uint2(0,2)] = -30000;
			param[make_uint2(0,3)] = 175;
		}
		__syncthreads();
	}
	static DEVICE Window getWindow(const cudaTextureObject_t texObj, const int dataset, const int sample_count) {
		return Window(0, sample_count);
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
		float res;
		if(param%2 == 1) res = pow(x,static_cast<float>(param));
		else                               res = x*pow(x,param-1);		
		//return pow(x, param); //pow does not work for odd i
		return res;
	}

	static DEVICE void getWindow(const cudaTextureObject_t texObj, Window& window, const int dataset, const int sample_count) {
		window.set(0, sample_count);
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
			float max = -40000;
			for(int i = 0; i < sample_count; i++) {
				if(getSample(texObj, i, blockIdx.x) > max) {
					max = getSample(texObj, i, blockIdx.x);
					pos = i;
				}
			}
			
			if(pos > sample_count - 50) {
				window.set(sample_count-101, 100);
			}
			if(pos < 50) {
				window.set(0, 100);
			}
			window.set(pos-50, 100);
		}
		__syncthreads();
	}
	template <class MatrixAccess>
	static DEVICE void guessParams(const cudaTextureObject_t texObj, MatrixAccess& param, const Window& window) {
		if(threadIdx.x == 0) {
			const float a = window.offset+50;
			const float b = getSample(texObj, a, blockIdx.x);
			const float c = -1; //(getSample(texObj, window.offset, blockIdx.x)-b)/((window.offset-a)*(window.offset-a));
			param[make_uint2(0,0)] = b+c*a*a;
			param[make_uint2(0,1)] = -2*a*c;
			param[make_uint2(0,2)] = c;
		}
		__syncthreads();
	}
	#endif
};

/* End user definitions*/

#endif

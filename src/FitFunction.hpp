#ifndef FITFUNCTION_HPP
#define FITFUNCTION_HPP

#include "FitFunctor.hpp"

/* User definitions */


/*!
 * \brief fit function in the form 0 = f(x,y)
 */
class Gauss:public FitFunctor<4> {
public:
	#ifdef __CUDACC__
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
		return pow(x, param);
	}

	static DEVICE Window getWindow(const cudaTextureObject_t texObj, const int dataset, const int sample_count) {
		return Window(0, sample_count);
	}
	#endif
};

template <unsigned int order>
class WindowPolynom:public Polynom<order> {
	#ifdef __CUDACC__
private:
	static DEVICE float getSample(const cudaTextureObject_t texObj, const float I, const int INDEXDATASET) {
		return tex2D<float>(texObj, I+0.5, INDEXDATASET);
	}
public:
	static DEVICE Window getWindow(const cudaTextureObject_t texObj, const int dataset, const int sample_count) {
		int pos = sample_count / 2;
		if(pos > sample_count - 50) {
			return Window(sample_count-101, 100);
		}
		if(pos < 50) {
			return Window(0, 100);
		}
		return Window(pos-50, 100);
	}
	#endif
};

/* End user definitions*/

#endif

#ifndef FITFUNCTION_H
#define FITFUNCTION_H

#include "FitFunctor.h"
/* User definitions */


/*!
 * \brief fit function in the form 0 = f(x,y)
 */
//const unsigned int numberOfParams = 3;

template <unsigned int order>
class Polynom:public FitFunctor<order+1> {
public:
	static DEVICE float modelFunction(float x, float y, float *param) {
		float res = 0;
		#pragma loop unroll
		for(int i = 0; i <= order; i++) {
			if(i%2 == 0) res += param[i]*pow(x,i);
			else		 res += param[i]*x*pow(x,i-1);		
		}
		res -= y;
		return res;
	}
	static DEVICE float derivation(int param, float x, float y, float *params) {
		return pow(x, param);
	}

	static DEVICE Window getWindow(cudaTextureObject_t texObj, int dataset, int sample_count) {
		return Window(0, sample_count);
	}
};

template <unsigned int order>
class WindowPolynom:public Polynom<order> {
private:
	static DEVICE float getSample(cudaTextureObject_t texObj, float I, int INDEXDATASET) {
		return tex2D<float>(texObj, I, INDEXDATASET);
	}
public:
	static DEVICE Window getWindow(cudaTextureObject_t texObj, int dataset, int sample_count) {
		int pos = sample_count / 2;
		if(pos > sample_count - 50) {
			return Window(sample_count-101, 100);
		}
		if(pos < 50) {
			return Window(0, 100);
		}
		return Window(pos-50, 100);
	}
};

/* End user definitions*/

#endif

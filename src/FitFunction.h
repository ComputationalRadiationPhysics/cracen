#ifndef FITFUNCTION_H
#define FITFUNCTION_H

#include "FitFunctor.h"
/* User definitions */


/*!
 * \brief fit function in the form 0 = f(x,y)
 */

template <unsigned int order>
class Polynom:public FitFunctor<order+1> {
public:
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
};

template <unsigned int order>
class WindowPolynom:public Polynom<order> {
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
};

/* End user definitions*/

#endif

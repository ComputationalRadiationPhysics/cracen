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
		for(int i = 0; i <= order; i++) res += param[i]*pow(x,i);
		res -= y;
		return res;
	}
	static DEVICE float derivation(int param, float x, float y, float *params) {
		return pow(x, param);
	}

	static DEVICE Window getWindow(int dataset, int sample_count) { 
		return Window(0, sample_count);
	}
};

template <unsigned int order>
class WindowPolynom:public Polynom<order> {
public:
	static __device__ Window getWindow(int dataset, int sample_count) {
		/*
		int pos = 2;
		float value = getSample<tex>(0,dataset)+getSample<tex>(1,dataset)+getSample<tex>(2,dataset)+getSample<tex>(3,dataset)+getSample<tex>(4,dataset);
		float max = value/5;
		for(int i = 3; i < sample_count-2; i++) {
			value += getSample<tex>(i+2,dataset);
			value -= getSample<tex>(i-2,dataset);
			if(value/5 > max) {
				max = value/5;
				pos = i;
			}
		}
		if(pos > sample_count - 50) {
			return Window(sample_count-101, 100);
		}
		if(pos < 50) {
			return Window(0, 100);
		}
		return Window(pos-50, 100);
		*/
		return Window(400,100);
	}
};

/* End user definitions*/

#endif

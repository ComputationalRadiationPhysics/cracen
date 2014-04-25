#ifndef FITFUNCTION_H
#define FITFUNCTION_H

#include "FitFunctor.h"
/* User definitions */


/*!
 * \brief fit function in the form 0 = f(x,y)
 */

template <unsigned int order, unsigned int tex>
class Polynom:public FitFunctor<order+1, tex> {
public:
	static __device__ inline float modelFunction(float x, float y, float *param) {
		//x /= 1000;
		float res = 0;
		#pragma loop unroll
		for(int i = 0; i < order+1; i++) res += param[i]*pow(x,i);
		res -= y;
		return res;
	}
	static __device__ float derivation(int param, float x, float y, float *params) {
		//x/=1000;
		return pow(x, param);
	}

	static __device__ Window getWindow(int dataset, int sample_count) { 
		return Window(0, sample_count-1);
	}
};

template <unsigned int order, unsigned int tex>
class WindowPolynom:public Polynom<order, tex> {
public:
	static __device__ Window getWindow(int dataset, int sample_count) {
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
	}
};
/* End user definitions*/

#endif

#ifndef FITFUNCTOR_H
#define FITFUNCTOR_H

struct Window {
	unsigned int offset, width;
	Window() : offset(0), width(0) {};
	__device__ __host__ Window(unsigned int offset, unsigned int width) : 
		offset(offset),
		width(width) 
	{}
	Window(const Window& w) :
		offset(w.offset),
		width(w.width)
	{}
};

template <unsigned int paramCount>
class FitFunctor {
public:
	typedef struct {
		int quality;
		float params[paramCount];
	} Output;
	
	typedef float paramArray[paramCount];
	const static unsigned int numberOfParams = paramCount;
	
	static DEVICE float modelFunction(float x, float y, float* param); // = 0;
	static DEVICE float derivation(unsigned int param, float x, float y, float* params); // = 0;	
	
	static DEVICE Window getWindow(cudaTextureObject_t texObj, int dataset, int sample_count); // = 0;
};
#endif

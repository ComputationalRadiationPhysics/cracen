#ifndef FITFUNCTOR_H
#define FITFUNCTOR_H

#define DEVICE __device__ //__forceinline__

struct Window {
	unsigned int offset, width;
	Window() : offset(0), width(0) {};
	DEVICE Window(unsigned int offset, unsigned int width) : 
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
	
	static DEVICE float modelFunction(const float x, const float y, const float * const param); // = 0;
	static DEVICE float derivation(const unsigned int param, const float x, const float y, const float * const params); // = 0;	
	
	static DEVICE Window getWindow(const cudaTextureObject_t texObj, const int dataset, const int sample_count); // = 0;
};
#endif

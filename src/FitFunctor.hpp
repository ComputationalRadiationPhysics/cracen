#ifndef FITFUNCTOR_HPP
#define FITFUNCTOR_HPP

#define DEVICE __device__ //__forceinline__

struct Window {
	unsigned int offset, width;
	#ifdef __CUDACC__
	DEVICE Window() {};
	DEVICE Window(unsigned int offset, unsigned int width) : 
		offset(offset),
		width(width) 
	{}
	DEVICE void __forceinline__ set(unsigned int _offset, unsigned int _width) {
		offset = _offset;
		width = _width;
	}
	Window(const Window& w) :
		offset(w.offset),
		width(w.width)
	{}
	#endif
};

template <unsigned int paramCount>
class FitFunctor {
public:
	const static unsigned int numberOfParams = paramCount;
	#ifdef __CUDACC__
	typedef struct {
		int quality;
		float params[paramCount];
	} Output;
	
	typedef float paramArray[paramCount];
	
	static DEVICE float modelFunction(const float x, const float y, const float * const param); // = 0;
	static DEVICE float derivation(const unsigned int param, const float x, const float y, const float * const params); // = 0;	
	
	static DEVICE void getWindow(const cudaTextureObject_t texObj, Window& window, const int dataset, const int sample_count); // = 0;
	template <class MatrixAccess>
	static DEVICE void guessParams(const cudaTextureObject_t texObj, MatrixAccess& param, const Window& window); // = 0;
	#endif
};
#endif

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

template <unsigned int paramCount, unsigned int tex>
class FitFunctor {
public:
	static inline __device__ __host__ unsigned int numberOfParams() { return paramCount;}
	static __device__ float modelFunction(float x, float y, float* param); // = 0;
	static __device__ float derivation(unsigned int param, float x, float y, float* params); // = 0;	
	
	static __device__ Window getWindow(int dataset, int sample_count); // = 0;
};

template<unsigned int numberOfParams>
struct FitData {
	float param[FitFunction::numberOfParams()];
	int status;
	FitData() {}
};
#endif

#ifndef MATRIX_H
#define MATRIX_H

template <class Type>
class Matrix {
private:
	unsigned int rows, cols
public:
	Type operator(unsigned int x, unsigned int y) = 0;
	Matrix transpose
}


Matrix<Type> transpose Matrix
template <class Fit>
class PartialDerivation : public Matrix<float> {
private:
	float* param;
	unsigned int sample_count, interpolation_count;
public:
	PartialDerivation(float* param, unsigned int sample_count, unsigned interpolation_count) :
		param(param),
		sample_count(sample_count),
		interpolation_count(interpolation_count) 
	{}
	float operator()(unsigned int x, unsigned int y) {
		if(y < sample_count/interpolation_count+Fit::numberOfParams() && x < Fit::numberOfParams()) {
			if(y*interpolation_count < sample_count) {
				return Fit::derivation(x,y*interpolation_count+offset,getSample<tex>(y*interpolation_count+offset,wave),param);
			} else {
				if((y - sample_count/interpolation_count) == x) {
					return mu;
				} else {
					return 0;
				}
			}
		}
	}
	PartialDerivation transpose() {
	
	}
}


#endif

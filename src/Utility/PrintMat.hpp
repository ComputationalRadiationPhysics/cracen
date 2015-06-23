#ifndef UTILKERNELS_HPP
#define UTILKERNELS_HPP

#include <cstdio>

#ifdef DEBUG_ENABLED
template <typename T, class AccessMode>
DEVICE void printMat(MatrixAccess<T, AccessMode> mat) {
	printf("{");
	for(int j = 0; j < mat.getRows(); j++) {
		printf("{");
		for(int i = 0; i < mat.getCols(); i++) {
			printf("%f",mat[make_uint2(i,j)]);
			if(i<mat.getCols()-1) printf(", ");
		}
		printf("}");
		if(j<mat.getRows()-1) printf(",\n");
	}
	printf("}\n");
}


template <typename T, class AccessMode>
DEVICE void printIntMat(MatrixAccess<T, AccessMode>& mat) {
	for(int j = 0; j < mat.getRows(); j++) {
		for(int i = 0; i < mat.getCols(); i++) {
			printf("%i ",mat[make_uint2(i,j)]);
		}
		printf("\n");
	}
	printf("\n");
}
#endif

#endif

CXX = nvcc
CFLAGS = -g -arch=compute_30 -code=sm_30
LFLAGS = -lpthread
OBJECTS = Node.o Dsp.o OutputStream.o DataReader.o

MatMulTest: MatMulTest.cu ../UtilKernels.h Util.h
	nvcc -arch=compute_30 -code=sm_30 -o MatMulTest MatMulTest.cu
	./MatMulTest

GaussJordanTest: GaussJordanTest.cu ../GaussJordan.h Util.h
	nvcc -arch=compute_30 -code=sm_30 -o GaussJordanTest GaussJordanTest.cu
	./GaussJordanTest

	
.PHONY: clean
clean:
	@rm -f *.o
	@rm -f *~

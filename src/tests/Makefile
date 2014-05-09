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

LevMarqTest: LevMarqTest.cu ../LevMarq.h ../GaussJordan.h Util.h ../UtilKernels.h
	nvcc -arch=compute_30 -code=sm_30 -o LevMarqTest LevMarqTest.cu

LapTest: LapTest.cu ../GaussJordan.h ../UtilKernels.h
	nvcc  -arch=compute_30 -code=sm_30 -o LapTest LapTest.cu
	./LapTest
	
run: LevMarqTest
	./LevMarqTest
	
.PHONY: sync
sync:
	rsync -a --progress . rossendorf:~/build 
	ssh rossendorf "rsync -a --progress build/ hypnos2:~/build"

	
.PHONY: clean
clean:
	@rm -f *.o
	@rm -f *~

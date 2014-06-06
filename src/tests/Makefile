CXX = nvcc
CFLAGS = -arch=compute_35 -code=sm_35 -rdc=true -lcudadevrt
LFLAGS = -lpthread -rdc=true -lcudadevrt
OBJECTS = Node.o Dsp.o OutputStream.o DataReader.o

export CPLUS_INCLUDE_PATH=

MatMulTest: MatMulTest.cu ../UtilKernels.h Util.h
	nvcc $(CFLAGS) -o MatMulTest MatMulTest.cu
	./MatMulTest

GaussJordanTest: GaussJordanTest.cu ../GaussJordan.h Util.h
	nvcc $(CFLAGS) -o GaussJordanTest GaussJordanTest.cu
	./GaussJordanTest

LevMarqTest: LevMarqTest.cu ../LevMarq.h ../GaussJordan.h Util.h ../UtilKernels.h
	nvcc $(CFLAGS) -o LevMarqTest LevMarqTest.cu
	./LevMarqTest
	
LapTest: LapTest.cu ../GaussJordan.h ../UtilKernels.h
	nvcc  $(CFLAGS) -o LapTest LapTest.cu
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
	@rm LevMarqTest
	@rm GaussJordanTest
	@rm MatMulTest

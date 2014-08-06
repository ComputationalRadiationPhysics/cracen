CXX = nvcc
CFLAGS = -g -G -arch=compute_35 -code=sm_35 -rdc=true -lcudadevrt -DDEBUG_ENABLED
LFLAGS = -lpthread -rdc=true -lcudadevrt
OBJECTS = Node.o Dsp.o OutputStream.o DataReader.o

export CPLUS_INCLUDE_PATH=

MatMulTest: MatMulTest.cu ../MatMul.hpp Util.hpp
	nvcc $(CFLAGS) -o MatMulTest MatMulTest.cu
	./MatMulTest

LevMarqTest: LevMarqTest.cu ../LevMarq.hpp ../GaussJordan.hpp Util.hpp ../MatMul.hpp
	nvcc $(CFLAGS) -o LevMarqTest LevMarqTest.cu
	./LevMarqTest

LevMarqTest2: LevMarqTest2.cu ../LevMarq.hpp ../GaussJordan.hpp Util.hpp ../MatMul.hpp
	nvcc $(CFLAGS) -o LevMarqTest2 LevMarqTest2.cu
	./LevMarqTest2
	
LapTest: LapTest.cu ../GaussJordan.hpp ../UtilKernels.hpp
	nvcc  $(CFLAGS) -o LapTest LapTest.cu
	./LapTest

GaussJordanTest: GaussJordanTest.cu ../UtilKernels.hpp ../GaussJordan.hpp
	nvcc $(CFLAGS) -o GaussJordanTest GaussJordanTest.cu
	./GaussJordanTest
	
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
	@rm -f LevMarqTest
	@rm -f GaussJordanTest
	@rm -f MatMulTest

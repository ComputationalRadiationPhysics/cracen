CXX = nvcc
CFLAGS = -arch=compute_35 -code=sm_35 -Xptxas=-v -O3 --keep --keep-dir ../tmp --source-in-ptx -lineinfo
LFLAGS = -lpthread # -rdc=true -lcudadevrt 
OBJECTS = Node.o Dsp.o OutputStream.o DataReader.o
export CPLUS_INCLUDE_PATH=

all: $(OBJECTS)
	$(CXX) $(CFLAGS) $(LFLAGS) -o DSP $(OBJECTS)

Dsp.o: Dsp.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o Dsp.o Dsp.cu
	
Node.o: Node.cu Node.h LevMarq.h UtilKernels.h GaussJordan.h Constants.h FitFunction.h FitFunctor.h
	$(CXX) $(CFLAGS) $(LFLAGS) -dc -o Node.o Node.cu
	
%.o: %.cu %.h
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<
	

.PHONY: sync
sync:
	rsync -a --progress . rossendorf:~/build 
	ssh rossendorf "rsync -a --progress build/ hypnos2:~/build"

.PHONY: run
run: all
	./DSP

.PHONY: clean
clean:
	@rm -f *.o
	@rm -f *~

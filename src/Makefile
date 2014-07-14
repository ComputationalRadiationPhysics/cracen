CXX = nvcc
CFLAGS = 
CUDA_FLAGS = -arch=compute_35 -code=sm_35 -Xptxas=-v -O3 --keep --keep-dir ../tmp --source-in-ptx -lineinfo # -rdc=true -lcudadevrt 
LFLAGS = -lpthread
_OBJECTS = Node.o Dsp.o OutputStream.o DataReader.o StopWatch.o
BUILD_PATH = ../build
OBJECTS = $(patsubst %,$(BUILD_PATH)/%,$(_OBJECTS))
export CPLUS_INCLUDE_PATH=

all: $(OBJECTS)
	$(CXX) $(LFLAGS) $(CUDA_FLAGS) -o DSP $(OBJECTS)

$(BUILD_PATH)/Dsp.o: Dsp.cu
	$(CXX) $(CUDA_FLAGS) $(LFLAGS) -c -o $@ Dsp.cu
	
$(BUILD_PATH)/Node.o: Node.cu Node.h LevMarq.h UtilKernels.h GaussJordan.h Constants.h FitFunction.h FitFunctor.h
	$(CXX) $(CUDA_FLAGS) $(LFLAGS) -dc -o $@ Node.cu
	
$(BUILD_PATH)/%.o: %.cu %.h
	$(CXX) $(CUDA_FLAGS) $(LFLAGS) -c -o $@ $<


.PHONY: sync
sync:
	rsync -a --progress . rossendorf:~/DSP/src
	ssh rossendorf "rsync -a --progress DSP/src hypnos2:~/DSP"

.PHONY: run 
run: all
	./DSP

.PHONY: clean
clean:
	@rm -f $(BUILD_PATH)/*

CXX = nvcc
CFLAGS = -arch=compute_35 -code=sm_35 -Xptxas=-v -O3 --keep --keep-dir ../tmp --source-in-ptx -lineinfo
LFLAGS = -lpthread # -rdc=true -lcudadevrt 
_OBJECTS = Node.o Dsp.o OutputStream.o DataReader.o
BUILD_PATH = ../build
OBJECTS = $(patsubst %,$(BUILD_PATH)/%,$(_OBJECTS))
export CPLUS_INCLUDE_PATH=

all: $(OBJECTS)
	$(CXX) $(CFLAGS) $(LFLAGS) -o DSP $(OBJECTS)

$(BUILD_PATH)/Dsp.o: Dsp.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ Dsp.cu
	
$(BUILD_PATH)/Node.o: Node.cu Node.h LevMarq.h UtilKernels.h GaussJordan.h Constants.h FitFunction.h FitFunctor.h
	$(CXX) $(CFLAGS) $(LFLAGS) -dc -o $@ Node.cu
	
$(BUILD_PATH)/%.o: %.cu %.h
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<
	

.PHONY: sync
sync:
	rsync -a --progress . rossendorf:~/build 
	ssh rossendorf "rsync -a --progress build/ hypnos2:~/build"

.PHONY: run 
run: clean all
	./DSP

.PHONY: clean
clean:
	@rm -f $(BUILD_PATH)/*

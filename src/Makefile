CUXX = nvcc
CXX = g++
CFLAGS = -std=c++11 -I/opt/pkg/devel/boost/1.55.0/include/
CUDA_FLAGS = -arch=compute_35 -code=sm_35 -Xptxas=-v -O3 -DDEBUG_ENABLED\
#--keep --keep-dir ../tmp --source-in-ptx -lineinfo \
#-DBOOST_NOINLINE='__attribute__ ((noinline))' \
#-Xcudafe "--diag_suppress=boolean_controlling_expr_is_constant" \

LFLAGS = -lpthread 
_OBJECTS = Node.o Dsp.o OutputStream.o DataReader.o StopWatch.o
BUILD_PATH = ../build
OBJECTS = $(patsubst %,$(BUILD_PATH)/%,$(_OBJECTS))
export CPLUS_INCLUDE_PATH=

all: $(OBJECTS)
	$(CUXX) $(LFLAGS) $(CUDA_FLAGS) -o DSP $(OBJECTS)

$(BUILD_PATH)/Dsp.o: Dsp.cu
	$(CUXX) $(CUDA_FLAGS) $(LFLAGS) -c -o $@ Dsp.cu
	
$(BUILD_PATH)/Node.o: Node.cu Node.hpp LevMarq.hpp MatMul.hpp PrintMat.hpp GaussJordan.hpp Constants.hpp FitFunction.hpp FitFunctor.hpp Device.hpp
	$(CUXX) $(CUDA_FLAGS) $(LFLAGS) -dc -o $@ Node.cu
	
$(BUILD_PATH)/%.o: %.cpp %.hpp
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<
	
.PHONY: run 
run: all
	./DSP

.PHONY: clean
clean:
	@rm -f $(BUILD_PATH)/*

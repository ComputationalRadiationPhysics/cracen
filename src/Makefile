CXX = nvcc
CFLAGS = -g -arch=compute_30 -code=sm_30
LFLAGS = -lpthread
OBJECTS = Node.o Dsp.o OutputStream.o DataReader.o


all: $(OBJECTS)
	$(CXX) $(CFLAGS) $(LFLAGS) -o DSP $(OBJECTS)

Dsp.o: Dsp.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o Dsp.o Dsp.cu
	
Node.cu: Node.cu Node.h LevMarq.h
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o Node.o Node.cu
	
%.o: %.cu %.h
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<
	
.PHONY: run
run: all
	./DSP

.PHONY: clean
clean:
	@rm -f *.o
	@rm -f *~

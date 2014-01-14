CXX = nvcc
CFLAGS = 
LFLAGS = -lpthread
OBJECTS = Node.o Dsp.o

all: $(OBJECTS)
	$(CXX) $(CFLAGS) $(LFLAGS) -o DSP $(OBJECTS)
	
%.o: %.cpp
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<

%.o: %.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<
	
.PHONY: run
run: all
	./DSP

.PHONY: clean
	@rm -f *.o
	@rm -f *~

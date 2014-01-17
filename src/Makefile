CXX = nvcc
CFLAGS = 
LFLAGS = -lpthread
OBJECTS = Node.o Dsp.o
TESTOBJECTS = DataReader.o tests.o


all: $(OBJECTS)
	$(CXX) $(CFLAGS) $(LFLAGS) -o DSP $(OBJECTS)

test_DSP: $(TESTOBJECTS)
	$(CXX) $(CFLAGS) $(LFLAGS) -o test_DSP $(TESTOBJECTS)

# tests.o: test_DataReader.h test_RingbufferWf.h

%.o: %.cpp
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<

%.o: %.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<
	
.PHONY: run
run: all
	./DSP

.PHONY: tests
tests: test_DSP

.PHONY: clean
	@rm -f *.o
	@rm -f *~

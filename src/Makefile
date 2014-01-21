CXX = nvcc
CFLAGS = -g
LFLAGS = -lpthread
OBJECTS = Node.o Dsp.o OutputStream.o LevMarq.o DataReader.o
TESTOBJECTS = DataReader.o tests.o


all: $(OBJECTS)
	$(CXX) $(CFLAGS) $(LFLAGS) -o DSP $(OBJECTS)

test_DSP: $(TESTOBJECTS)
	$(CXX) $(CFLAGS) $(LFLAGS) -o test_DSP $(TESTOBJECTS)

Dsp.o: Dsp.cu
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o Dsp.o Dsp.cu
	
%.o: %.cu %.h
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<
	
.PHONY: run
run: all
	./DSP

.PHONY: tests
tests: test_DSP

.PHONY: clean
clean:
	@rm -f *.o
	@rm -f *~

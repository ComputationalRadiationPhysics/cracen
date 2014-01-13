CXX = nvcc
CFLAGS = -Wall
LFLAGS = 
OBJECTS = Node.o Main.o

all: OBJECTS
	$(CXX) $(CFLAGS) $(LFLAGS) -o DSP $(OBJECTS)
	
%.o: %.cpp
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<
	
.PHONY: run
run: all
	./DSP

.PHONY: clean
	@rm -f *.o

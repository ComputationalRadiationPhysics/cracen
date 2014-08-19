CXX = g++
CFLAGS = -std=c++11
LFLAGS = `pkg-config --libs --cflags gtk+-3.0` `pkg-config --libs --cflags gtkmm-3.0` -lpthread
OBJECTS = Viewer.o DataReader.o

all: $(OBJECTS)
	$(CXX) $(CFLAGS) $(LFLAGS) -o Viewer $(OBJECTS)


%.o: %.cpp
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<

%.o: ../%.cpp ../%.hpp
	$(CXX) $(CFLAGS) $(LFLAGS) -c -o $@ $<

.PHONY: run
run: all
	./Viewer&
	
.PHONY: clean
clean:
	rm -f *.o

plot: plot.gnu
	gnuplot plot.gnu
	eog plot.png

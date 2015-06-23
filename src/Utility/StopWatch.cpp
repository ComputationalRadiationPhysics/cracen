#include "StopWatch.hpp"

StopWatch::StopWatch() : state(0) {}

int StopWatch::start() {
	state = 1;
	return clock_gettime(CLOCK_REALTIME, &_start);
}

int StopWatch::stop() {
	if(state < 1) {
		std::cerr << "StopWatch has not been started." << std::endl;
		return -1;
	}
	state = 2;
	return clock_gettime(CLOCK_REALTIME, &_end);;
}

double StopWatch::elapsedSeconds() const {
	if(state < 2) {
		std::cerr << "StopWatch has not been stopped." << std::endl;
		return -1;
	}
	
	return _end.tv_sec + _end.tv_nsec/1e9 - _start.tv_sec - _start.tv_nsec/1e9;
}

std::ostream& operator<<(std::ostream& lhs, const StopWatch& stopWatch) {
	if(stopWatch.state < 2) {
		std::cerr << "StopWatch has not been stopped." << std::endl;
		return lhs;
	}
	lhs << stopWatch.elapsedSeconds() << " sec.";
	return lhs;
}

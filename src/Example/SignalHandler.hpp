#pragma once

#include <condition_variable>
#include <mutex>
#include <csignal>




namespace detail {

volatile bool ready;
std::mutex mutex;
std::condition_variable cv;

void signalHandler( int signum ) {

	std::cout << "Catched signal " << signum << std::endl;
	std::unique_lock<std::mutex> lock(mutex);
	ready = true;
	cv.notify_all();

}

}
void waitForSignal(int sig) {
	signal(sig, detail::signalHandler);
	std::unique_lock<std::mutex> lock(detail::mutex);
	detail::ready = false;
	while(!detail::ready) detail::cv.wait(lock);
}

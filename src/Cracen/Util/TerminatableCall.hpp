#pragma once

#include <iostream>
#include <future>
#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace Cracen {

namespace Util {

enum class TerminationState {
	Success,
	Failed
};

template <class Functor>
TerminationState terminateAble(Functor fn, std::shared_ptr<std::atomic<bool>> running, std::chrono::nanoseconds checkInterval = std::chrono::seconds(1)) {

	std::shared_ptr<bool> finished = std::make_shared<bool>(false);
	std::shared_ptr<std::mutex> mutex = std::make_shared<std::mutex>();
	std::shared_ptr<std::condition_variable> cv = std::make_shared<std::condition_variable>();

	std::thread call([=](){
		fn();
		std::unique_lock<std::mutex> lock(*mutex);
		*finished = true;
		cv->notify_all();
	});

	std::unique_lock<std::mutex> lock(*mutex);
	while(!(*finished)) {
		cv->wait_for(lock, checkInterval);
		if(!(*running)) {
			call.detach();
			return TerminationState::Failed;
		}
	}

	call.join();
	return TerminationState::Success;
}

} // End of Util

} // End of cracen

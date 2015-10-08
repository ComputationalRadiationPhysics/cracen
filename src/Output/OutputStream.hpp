#ifndef OUTPUTSTREAM_HPP
#define OUTPUTSTREAM_HPP

#include <fstream>
#include <string>
#include "../Utility/Ringbuffer.hpp"
#include "../Config/Types.hpp"
#include "../Config/Constants.hpp"

/*! Class that provides all functions to write the results of the computation  
 *  into a file.
 */
class OutputStream {
private:
	Ringbuffer<Output> oBuffer;
	std::string file;
	bool done;
	pthread_t thread_tid;
	
	void run();
	static void* run_helper(void* This) { 
		static_cast<OutputStream*>(This)->run();
		return NULL;
	};
	
public:
	//! Basic constructor.
	/*!
	 *  Constructor opens a filestream, initialise the output buffer and start
	 *  the thread, which takes elements from the buffers and writes them into 
	 *  the file.
	 *
	 *  \param file Filename of the output file.
	 */
	OutputStream(const std::string& file, int producer);
	
	//! Returns a reference of the buffer.
	Ringbuffer<Output>* getBuffer();
	
	//! Waits until the writing thread to stops
	void join();
};

#endif

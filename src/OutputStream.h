#ifndef OUTPUTSTREAM_H
#define OUTPUTSTREAM_H

#include <fstream>
#include <string>
#include "Ringbuffer.h"
#include "Types.h"
#include "Constants.h"

/*! Class that provides all functions to write the results of the computation  
 *  into a file.
 */
class OutputStream {
private:
	std::ofstream file;
	Ringbuffer<Output> oBuffer;
	bool done;
	
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
	OutputStream(const std::string& file);
	
	//! Returns a reference of the buffer.
	Ringbuffer<Output>* getBuffer();
	
	//! Signals, that no new data will be written into the buffer.
	/*!
	 * This function will make the Writeback Thread stop, after all remaining
	 * elements in the buffer are written into the output file
	 */
	void finish();
};

#endif

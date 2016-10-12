#ifndef OUTPUTSTREAM_HPP
#define OUTPUTSTREAM_HPP

#include <thread>
#include "../Utility/Ringbuffer.hpp"
#include "../Config/Types.hpp"
#include "../Config/Constants.hpp"

/*! Class that provides all functions to send data over the network
 */

template <class DataType, class Cage>
class GrayBatStream {
private:
	// Cage
	typedef typename Cage::Event  Event;
	typedef typename Cage::Vertex Vertex;
	typedef typename Cage::Edge Edge;
	
	Ringbuffer<DataType> oBuffer;
	Cage& cage;
	bool done;
	std::thread sendingThread;
	unsigned int roundRobinCounter;
	
	void run() {
		assert(cage.hostedVertices.size() > 0);
		
		while(!oBuffer.isFinished()) {
			DataType o = oBuffer.pop();
			//Send dataset away	
			Vertex source = cage.hostedVertices.at(0);
			std::vector<Edge> source_sink = cage.getOutEdges(source);
			
			//std::cout << "RRC=" << roundRobinCounter << "source_sink.size()="<< source_sink.size() << std::endl;
			cage.send(source_sink.at(roundRobinCounter), o);
			roundRobinCounter = (roundRobinCounter+1) % source_sink.size();
		}
	}
	
public:
	//! Basic constructor.
	/*!
	 *  Constructor opens a filestream, initialise the output buffer and start
	 *  the thread, which takes elements from the buffers and writes them into 
	 *  the file.
	 *
	 *  \param file Filename of the output file.
	 */
	GrayBatStream(int producer, Cage& cage) :
		oBuffer(CHUNK_BUFFER_COUNT, producer),
		cage(cage),
		done(false),
		sendingThread(&GrayBatStream::run, this),
		roundRobinCounter(0)
	{}
	
	~GrayBatStream() {sendingThread.join();}
	
	void send(const DataType& msg) {
		oBuffer.push(msg);
	}
	void quit() {
		oBuffer.producerQuit();
	}
	
	//! Returns a reference of the buffer.
	Ringbuffer<DataType>& getBuffer() {
		return oBuffer;
	}
	
};

#endif

#ifndef OUTPUTSTREAM_HPP
#define OUTPUTSTREAM_HPP

#include <thread>
#include "../Utility/Ringbuffer.hpp"
#include "../Config/Types.hpp"
#include "../Config/Constants.hpp"

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include <graybat/graphPolicy/BGL.hpp>
// GRAYBAT mappings
#include <graybat/mapping/PeerGroupMapping.hpp>
// GRAYBAT pattern
#include <graybat/pattern/Pipeline.hpp>

/*! Class that provides all functions to send data over the network
 */

template <class DataType>
class GrayBatStream {
private:
	/***************************************************************************
     * graybat configuration
     ****************************************************************************/
    // CommunicationPolicy
    typedef graybat::communicationPolicy::ZMQ CP;
    // GraphPolicy
    typedef graybat::graphPolicy::BGL<>    GP;
    
    // Cage
    typedef graybat::Cage<CP, GP> Cage;
    typedef typename Cage::Event  Event;
    typedef typename Cage::Vertex Vertex;
	typedef typename Cage::Edge Edge;
	
	Ringbuffer<DataType> oBuffer;
	CP cp;
	Cage cage;
	bool done;
	std::thread sendingThread;
	
	void run() {
		while(!oBuffer.isFinished()) {
			Output* o = oBuffer.reserveTailTry();
			if(o != NULL) {
				oBuffer.freeTail();
				//Send dataset away
				
				static unsigned int roundRobinCounter = 0;
				
				Vertex source = cage.hostedVertices.at(0);
				std::vector<Edge> source_sink = cage.getOutEdges(source);
				std::vector<DataType> send_buffer(1, *o);
				
				cage.send(source_sink[roundRobinCounter], send_buffer);
				roundRobinCounter = (roundRobinCounter+1) % source_sink.size();
				
				delete o;
			}
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
	GrayBatStream(int producer, const std::string& masterUri, const std::string& selfUri) :
		oBuffer(CHUNK_BUFFER_COUNT, producer),
		cp(masterUri, selfUri, 2),
		cage(cp),
		done(false),
		sendingThread(&GrayBatStream::run, this)
	{
		std::vector<unsigned int> stages;
		stages[0] = 1;
		stages[1] = 1;
		cage.setGraph(graybat::pattern::Pipeline(stages));
		cage.distribute(graybat::mapping::PeerGroupMapping(0));
	}
	
	void send(const DataType& msg) {
		DataType* head = oBuffer.reserveHead();
		std::copy(&msg, head);
		oBuffer.freeHead();
		
	}
	void quit() {
		oBuffer.producerQuit();
	}
	
	//! Returns a reference of the buffer.
	Ringbuffer<Output>* getBuffer() {
		return &oBuffer;
	}
	
	//! Waits until the writing thread to stops
	void join() {
		sendingThread.join();
	}
};

#endif

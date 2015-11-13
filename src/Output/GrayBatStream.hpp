#ifndef OUTPUTSTREAM_HPP
#define OUTPUTSTREAM_HPP

#include <thread>
#include "../Utility/Ringbuffer.hpp"
#include "../Config/Types.hpp"
#include "../Config/Constants.hpp"

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>
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
	//    typedef graybat::communicationPolicy::ZMQ CP;
	typedef graybat::communicationPolicy::BMPI CP;

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
	unsigned int roundRobinCounter;
	
	void run() {
		std::vector<unsigned int> stages(2);
		stages[0] = 1;
		stages[1] = 1;
		cage.setGraph(graybat::pattern::Pipeline(stages));
		auto g = graybat::pattern::Pipeline(stages)();
		std::cout << "Edges" << std::endl;
		for(auto e : g.second) {
			std::cout << e.first << "->" << e.second << std::endl;
		}
		cage.distribute(graybat::mapping::PeerGroupMapping(0));
		assert(cage.hostedVertices.size() > 0);
		
		while(!oBuffer.isFinished()) {
			DataType o = oBuffer.pop();
			//Send dataset away	
			std::cout << "Sending package." << std::endl;
			Vertex source = cage.hostedVertices.at(0);
			std::vector<Edge> source_sink = cage.getOutEdges(source);
			std::vector<DataType> send_buffer(1, o);
			
			cage.send(source_sink[roundRobinCounter], send_buffer);
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
	GrayBatStream(int producer, const std::string& masterUri, const std::string& selfUri) :
		oBuffer(CHUNK_BUFFER_COUNT, producer),
		//cp(masterUri, selfUri, 2),
		cage(cp),
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

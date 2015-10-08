#ifndef DATAREADER_HPP
#define DATAREADER_HPP

#include <thread>

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include <graybat/graphPolicy/BGL.hpp>
// GRAYBAT mappings
#include <graybat/mapping/PeerGroupMapping.hpp>
// GRAYBAT pattern
#include <graybat/pattern/Pipeline.hpp>

#include "../Config/Types.hpp"
#include "../Config/Constants.hpp"
#include "../Utility/Ringbuffer.hpp"

/*! GrayBatReader
 *  @brief
 */

template <class DataType>
class GrayBatReader {

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
	
	Ringbuffer<DataType> iBuffer;
	CP cp;
	Cage cage;
	bool done;
	std::thread receivingThread;
	
	void run() {
		while(!done) {
			Vertex sink = cage.hostedVertices.at(0);
			std::vector<DataType> receive_buffer(1);
			sink.collect(receive_buffer);
			DataType* head = iBuffer.reserveHead();
			head = new DataType;
			//*head = receive_buffer[0];
			std::copy(receive_buffer.begin(), receive_buffer.begin()+1, head);
			iBuffer.freeHead();
		}
	}
public:
    /**
     */
    GrayBatReader(const std::string& masterUri, const std::string& selfUri) :
		iBuffer(CHUNK_BUFFER_COUNT, 1),
		cp(masterUri, selfUri, 2),
		cage(cp),
		done(false),
		receivingThread(&GrayBatReader::run, this)
	{
		std::vector<unsigned int> stages;
		stages[0] = 1;
		stages[1] = 1;
		cage.setGraph(graybat::pattern::Pipeline(stages));
		cage.distribute(graybat::mapping::PeerGroupMapping(1));
	}
	
	
    ~GrayBatReader() {}

    void readToBuffer() {}

	Ringbuffer<DataType>* getBuffer() {
		return &iBuffer;
	}
};

#endif

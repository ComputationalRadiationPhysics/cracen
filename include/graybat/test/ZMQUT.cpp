// BOOST
#include <boost/test/unit_test.hpp>

// STL
#include <functional> /* std::plus */
#include <iostream>   /* std::cout, std::endl */
#include <array>      /* std::array */
#include <numeric>    /* std::iota */
#include <cstdlib>    /* std::getenv */
#include <string>     /* std::string, std::stoi */

// ZMQ
#include <zmq.hpp>

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/mapping/Random.hpp>
#include <graybat/mapping/Consecutive.hpp>
#include <graybat/mapping/Roundrobin.hpp>
#include <graybat/pattern/FullyConnected.hpp>
#include <graybat/pattern/InStar.hpp>
#include <graybat/pattern/Grid.hpp>


/***************************************************************************
 * Configuration
 ****************************************************************************/

// CommunicationPolicy
typedef graybat::communicationPolicy::ZMQ CP;
typedef typename CP::Context              Context;
typedef typename CP::Event                Event;

// GraphPolicy
typedef graybat::graphPolicy::BGL<>        GP;
    
// Cage
typedef graybat::Cage<CP, GP> Cage;
typedef typename Cage::Event  Event;
typedef typename Cage::Vertex Vertex;
typedef typename Cage::Edge   Edge;


BOOST_AUTO_TEST_SUITE( zmq )

// Global test variables
const unsigned nRuns = 100;
const std::string masterUri   = "tcp://127.0.0.1:5000";
const std::string peerUri     = "tcp://127.0.0.1:5001";
const unsigned contextSize    = std::stoi(std::getenv("OMPI_COMM_WORLD_SIZE"));	    


BOOST_AUTO_TEST_CASE( construct ){
    for(unsigned i = 0; i < nRuns; ++i){
	CP zmq(masterUri, peerUri, contextSize);
    }

}

BOOST_AUTO_TEST_CASE( context ){
    CP zmq(masterUri, peerUri, contextSize);    
    Context oldContext = zmq.getGlobalContext();

    for(unsigned i = 0; i < nRuns; ++i){
    	Context newContext = zmq.splitContext(true, oldContext);
	oldContext = newContext;
	
    }

}


BOOST_AUTO_TEST_CASE( send_recv ){
    const unsigned nElements = 10;
    const unsigned tag = 99;

    CP zmq(masterUri, peerUri, contextSize);  
    Context context = zmq.getGlobalContext();
    
    for(unsigned i = 0; i < nRuns; ++i){
    	std::vector<unsigned> recv (nElements, 0);
	std::vector<Event> events;

	for(unsigned vAddr = 0; vAddr < context.size(); ++vAddr){
	    std::vector<unsigned> data (nElements, 0);
	    std::iota(data.begin(), data.end(), context.getVAddr());
	    events.push_back(zmq.asyncSend(vAddr, tag, context, data));
        
	}

	for(unsigned vAddr = 0; vAddr < context.size(); ++vAddr){
	    zmq.recv(vAddr, tag, context, recv);

	    for(unsigned i = 0; i < recv.size(); ++i){
		BOOST_CHECK_EQUAL(recv[i], vAddr+i);
            
	    }

	}

	for(Event &e : events){
	    e.wait();
	}
	
    }

}


BOOST_AUTO_TEST_CASE( send_recv_all ){
    CP zmq(masterUri, peerUri, contextSize);  
    Context context = zmq.getGlobalContext();

    const unsigned nElements = 10;
    
    for(unsigned i = 0; i < nRuns; ++i){
	std::vector<unsigned> recv (nElements, 0);
	std::vector<Event> events;

	for(unsigned vAddr = 0; vAddr < context.size(); ++vAddr){
	    std::vector<unsigned> data (nElements, 0);
	    std::iota(data.begin(), data.end(), context.getVAddr());
	    events.push_back(zmq.asyncSend(vAddr, 99, context, data));
        
	}

	for(unsigned i = 0; i < context.size(); ++i){
	    Event e = zmq.recv(context, recv);

	    unsigned vAddr = e.vAddr;

	    for(unsigned i = 0; i < recv.size(); ++i){
		BOOST_CHECK_EQUAL(recv[i], vAddr+i);
            
	    }

	}


	for(Event &e : events){
	    e.wait();
	}

    }

}

BOOST_AUTO_TEST_CASE( send_recv_order ){
    CP zmq(masterUri, peerUri, contextSize);      
    Context context = zmq.getGlobalContext();

    const unsigned nElements = 10;
    const unsigned tag = 99;    

    for(unsigned run_i = 0; run_i < nRuns; ++run_i){
    
	std::vector<Event> events;

	std::vector<unsigned> recv1 (nElements, 0);
	std::vector<unsigned> recv2 (nElements, 0);
	std::vector<unsigned> recv3 (nElements, 0);

	std::vector<unsigned> data1 (nElements, context.getVAddr());
	std::vector<unsigned> data2 (nElements, context.getVAddr() + 1);
	std::vector<unsigned> data3 (nElements, context.getVAddr() + 2);

	for(unsigned vAddr = 0; vAddr < context.size(); ++vAddr){
	    events.push_back( zmq.asyncSend(vAddr, tag, context, data1));
	    events.push_back( zmq.asyncSend(vAddr, tag, context, data2));
	    events.push_back( zmq.asyncSend(vAddr, tag, context, data3));
        
	}

	for(unsigned vAddr = 0; vAddr < context.size(); ++vAddr){
	    zmq.recv(vAddr, tag, context, recv1);
	    zmq.recv(vAddr, tag, context, recv2);
	    zmq.recv(vAddr, tag, context, recv3);

	    for(unsigned i = 0; i < recv1.size(); ++i){
		BOOST_CHECK_EQUAL(recv1[i], vAddr);
            
	    }

	    for(unsigned i = 0; i < recv1.size(); ++i){
		BOOST_CHECK_EQUAL(recv2[i], vAddr + 1);
            
	    }

	    for(unsigned i = 0; i < recv1.size(); ++i){
		BOOST_CHECK_EQUAL(recv3[i], vAddr + 2);
            
	    }

	}

	for(Event &e : events){
	    e.wait();
	}

    }

}

BOOST_AUTO_TEST_CASE( cage ){
    const unsigned nElements = 1000;
    const unsigned nRuns = 100;

    CP communicationPolicy(masterUri, peerUri, contextSize);      
    Cage cage (communicationPolicy);
    cage.setGraph(graybat::pattern::FullyConnected(cage.getPeers().size()));
    cage.distribute(graybat::mapping::Roundrobin());
    
    for(unsigned run_i = 0; run_i < nRuns; ++run_i){
	std::vector<Event> events; 
	std::vector<unsigned> send(nElements,0);
	std::vector<unsigned> recv(nElements,0);

	for(unsigned i = 0; i < send.size();++i){
	    send.at(i) = i;
	}
    
	//Send state to neighbor cells
	for(Vertex &v : cage.hostedVertices){
	    for(Edge edge : cage.getOutEdges(v)){
		cage.send(edge, send, events);

	    }
	}

	//Recv state from neighbor cells
	for(Vertex &v : cage.hostedVertices){
	    for(Edge edge : cage.getInEdges(v)){
		cage.recv(edge, recv);
		for(unsigned i = 0; i < recv.size();++i){
		    BOOST_CHECK_EQUAL(recv.at(i), i);
		}

	    }
	
	}

	for(Event &e : events){
	    e.wait();
	}

    }

}

BOOST_AUTO_TEST_CASE( multi_cage ){

	CP communicationPolicy1(masterUri, peerUri, contextSize);          
	Cage cage1(communicationPolicy1);
	cage1.setGraph(graybat::pattern::FullyConnected(cage1.getPeers().size()));
	cage1.distribute(graybat::mapping::Roundrobin());

	CP communicationPolicy2(masterUri, peerUri, contextSize);              
	Cage cage2(communicationPolicy2);
	cage2.setGraph(graybat::pattern::FullyConnected(cage2.getPeers().size()));
	cage2.distribute(graybat::mapping::Roundrobin());
	
}


BOOST_AUTO_TEST_SUITE_END()

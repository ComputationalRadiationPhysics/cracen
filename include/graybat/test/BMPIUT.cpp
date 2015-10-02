// BOOST
#include <boost/test/unit_test.hpp>

// STL
#include <array>
#include <vector>
#include <iostream>
#include <functional>

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/mapping/Random.hpp>
#include <graybat/mapping/Consecutive.hpp>
#include <graybat/pattern/FullyConnected.hpp>
#include <graybat/pattern/InStar.hpp>
#include <graybat/pattern/Grid.hpp>

/***************************************************************************
 * Configuration
 ****************************************************************************/

// CommunicationPolicy
typedef graybat::communicationPolicy::BMPI CP;
    
// GraphPolicy
typedef graybat::graphPolicy::BGL<>        GP;
    
// Cage
typedef graybat::Cage<CP, GP> Cage;
typedef typename Cage::Event  Event;
typedef typename Cage::Vertex Vertex;
typedef typename Cage::Edge   Edge;


/***************************************************************************
 * Test Cases
 ****************************************************************************/

BOOST_AUTO_TEST_SUITE( bmpi )

CP communicationPolicy;
Cage allToAll(communicationPolicy);
Cage star(communicationPolicy);
Cage grid(communicationPolicy);

BOOST_AUTO_TEST_CASE( multi_cage ){
    Cage cage1(communicationPolicy);
    Cage cage2(communicationPolicy);

    cage1.setGraph(graybat::pattern::InStar(cage1.getPeers().size()));
    cage1.distribute(graybat::mapping::Consecutive());
    cage2.setGraph(graybat::pattern::InStar(cage2.getPeers().size()));
    cage2.distribute(graybat::mapping::Consecutive());


    const unsigned nElements = 1000;
    
    std::vector<Event> events; 
    std::vector<unsigned> send(nElements,0);
    std::vector<unsigned> recv1(nElements,0);
    std::vector<unsigned> recv2(nElements,0);

    for(unsigned i = 0; i < send.size();++i){
        send.at(i) = i;
    }

    // Send state to neighbor cells
    for(Vertex &v : cage1.hostedVertices){
        for(Edge edge : cage1.getOutEdges(v)){
            cage1.send(edge, send);
	    
        }
    }

    for(Vertex &v : cage2.hostedVertices){
        for(Edge edge : cage2.getOutEdges(v)){
            cage2.send(edge, send);
	    
        }
    }


    // Recv state from neighbor cells
    for(Vertex &v : cage1.hostedVertices){
        for(Edge edge : cage1.getInEdges(v)){
            cage1.recv(edge, recv1);
            for(unsigned i = 0; i < recv1.size();++i){
        	BOOST_CHECK_EQUAL(recv1.at(i), i);
            }

        }
	
    }

    // Recv state from neighbor cells
    for(Vertex &v : cage2.hostedVertices){
        for(Edge edge : cage2.getInEdges(v)){
            cage2.recv(edge, recv2);
            for(unsigned i = 0; i < recv2.size();++i){
        	BOOST_CHECK_EQUAL(recv2.at(i), i);
            }

        }
	
    }

    
}


BOOST_AUTO_TEST_CASE( send_recv ){
    star.setGraph(graybat::pattern::InStar(star.getPeers().size()));
    star.distribute(graybat::mapping::Consecutive());
    const unsigned nElements = 1000;
    
    std::vector<Event> events; 
    std::vector<unsigned> send(nElements,0);
    std::vector<unsigned> recv(nElements,0);

    for(unsigned i = 0; i < send.size();++i){
	send.at(i) = i;
    }

    // Send state to neighbor cells
    for(Vertex &v : star.hostedVertices){
	for(Edge edge : star.getOutEdges(v)){
	    star.send(edge, send);
	    
	}
    }

    // Recv state from neighbor cells
    for(Vertex &v : star.hostedVertices){
	for(Edge edge : star.getInEdges(v)){
	    star.recv(edge, recv);
	    for(unsigned i = 0; i < recv.size();++i){
		BOOST_CHECK_EQUAL(recv.at(i), i);
	    }

	}
	
    }
    
}


BOOST_AUTO_TEST_CASE( asyncSend_recv ){
    allToAll.setGraph(graybat::pattern::FullyConnected(allToAll.getPeers().size()));
    allToAll.distribute(graybat::mapping::Consecutive());

    const unsigned nElements = 1000;
    
    std::vector<Event> events; 
    std::vector<unsigned> send(nElements,0);
    std::vector<unsigned> recv(nElements,0);

    for(unsigned i = 0; i < send.size();++i){
	send.at(i) = i;
    }

    // Send state to neighbor cells
    for(Vertex &v : allToAll.hostedVertices){
	for(Edge edge : allToAll.getOutEdges(v)){
	    allToAll.send(edge, send, events);
	}
    }

    // Recv state from neighbor cells
    for(Vertex &v : allToAll.hostedVertices){
	for(Edge edge : allToAll.getInEdges(v)){
	    allToAll.recv(edge, recv);
	    for(unsigned i = 0; i < recv.size();++i){
		BOOST_CHECK_EQUAL(recv.at(i), i);
	    }

	}
	
    }
    
    // Wait to finish events
    for(unsigned i = 0; i < events.size(); ++i){
	events.back().wait();
	events.pop_back();
    }

}

BOOST_AUTO_TEST_CASE( reduce ){
    grid.setGraph(graybat::pattern::Grid(3,3));
    grid.distribute(graybat::mapping::Consecutive());

    const unsigned nElements = 10;
    
    std::vector<unsigned> send(nElements, 1);
    std::vector<unsigned> recv(nElements, 0);

    Vertex rootVertex = grid.getVertex(0);
    
    for(Vertex v: grid.hostedVertices){
	grid.reduce(rootVertex, v, std::plus<unsigned>(), send, recv);
    }

    if(grid.peerHostsVertex(rootVertex)){
	for(unsigned receivedElement: recv){
	    BOOST_CHECK_EQUAL(receivedElement, grid.getVertices().size());
	}
    }
    
}


BOOST_AUTO_TEST_CASE( allReduce ){
    grid.setGraph(graybat::pattern::Grid(grid.getPeers().size(), grid.getPeers().size()));
    grid.distribute(graybat::mapping::Consecutive());

    const unsigned nElements = 10;
    
    std::vector<unsigned> send(nElements, 1);
    std::vector<unsigned> recv(nElements, 0);

    for(Vertex v: grid.hostedVertices){
	grid.allReduce(v, std::plus<unsigned>(), send, recv);
    }

    for(unsigned receivedElement: recv){
	BOOST_CHECK_EQUAL(receivedElement, grid.getVertices().size());
    }
    
}

BOOST_AUTO_TEST_CASE( gather ){
    grid.setGraph(graybat::pattern::Grid(3,3));
    grid.distribute(graybat::mapping::Consecutive());

    const unsigned nElements = 10;
    const unsigned testValue = 1;
    const bool reorder = true;

    
    std::vector<unsigned> send(nElements, testValue);
    std::vector<unsigned> recv(nElements * grid.getVertices().size(), 0);

    Vertex rootVertex = grid.getVertex(0);
    
    for(Vertex v: grid.hostedVertices){
	grid.gather(rootVertex, v, send, recv, reorder);
    }

    if(grid.peerHostsVertex(rootVertex)){
	for(unsigned receivedElement: recv){
	    BOOST_CHECK_EQUAL(receivedElement, testValue);
	}
    }
    
    
}

BOOST_AUTO_TEST_CASE( allGather ){
    grid.setGraph(graybat::pattern::Grid(grid.getPeers().size(), grid.getPeers().size()));
    grid.distribute(graybat::mapping::Consecutive());

    const unsigned nElements = 10;
    const unsigned testValue = 1;
    const bool reorder = true;

    
    std::vector<unsigned> send(nElements, testValue);
    std::vector<unsigned> recv(nElements * grid.getVertices().size(), 0);

    for(Vertex v: grid.hostedVertices){
	grid.allGather(v, send, recv, reorder);
    }

    for(unsigned receivedElement: recv){
	BOOST_CHECK_EQUAL(receivedElement, testValue);
    }
        
}

BOOST_AUTO_TEST_CASE( spreadAndCollect ){
    star.setGraph(graybat::pattern::InStar(star.getPeers().size()));
    star.distribute(graybat::mapping::Consecutive());

    const unsigned nElements = 10;
    const unsigned testValue = 1;
    std::vector<Event> events; 
    
    std::vector<unsigned> send(nElements, testValue);


    for(Vertex v: star.hostedVertices){
	v.spread(send, events);
    }

    for(Vertex v: star.hostedVertices){
	std::vector<unsigned> recv(v.nInEdges() * nElements, 0);
	v.collect(recv);
	for(unsigned receivedElement: recv){
	    BOOST_CHECK_EQUAL(receivedElement, testValue);
	}
	
    }

    for(unsigned i = 0; i < events.size(); ++i){
	events.back().wait();
	events.pop_back();
    }
  
}

BOOST_AUTO_TEST_SUITE_END()

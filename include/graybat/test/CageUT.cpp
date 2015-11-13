// STL
#include <array>
#include <vector>
#include <iostream>   /* std::cout, std::endl */
#include <functional> /* std::plus, std::ref */
#include <cstdlib>    /* std::getenv */
#include <string>     /* std::string, std::stoi */

// BOOST
#include <boost/test/unit_test.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/append.hpp>

// ELEGANT-PROGRESSBARS
#include <elegant-progressbars/policyProgressbar.hpp>
#include <elegant-progressbars/all_policies.hpp>

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/mapping/Random.hpp>
#include <graybat/mapping/Consecutive.hpp>
#include <graybat/mapping/Roundrobin.hpp>
#include <graybat/pattern/FullyConnected.hpp>
#include <graybat/pattern/InStar.hpp>
#include <graybat/pattern/Grid.hpp>


/*******************************************************************************
 * CommunicationPolicy configuration
 ******************************************************************************/
struct Config {

    Config() :
	masterUri("tcp://127.0.0.1:5000"),
	peerUri("tcp://127.0.0.1:5001"),
	contextSize(std::stoi(std::getenv("OMPI_COMM_WORLD_SIZE"))){

    }

    std::string const masterUri;
    std::string const peerUri;
    size_t const contextSize;	    

};

Config const config;
size_t const nRuns = 1000;


/*******************************************************************************
 * Progress
 ******************************************************************************/
using namespace ElegantProgressbars;
struct Progress {

    template<typename T_CP>
    Progress(T_CP& cp) :
	isMaster(false){
	
	isMaster = cp.getGlobalContext().getVAddr() == 0 ? true : false;
	
    }

    ~Progress(){
	
    }

    void print(unsigned const total, unsigned const current){
	if(isMaster){
	    std::cerr << policyProgressbar<Label, Spinner<>, Percentage>(total, current);
	    
	}
	
    }

    bool isMaster;

};

/*******************************************************************************
 * Test Suites
 ******************************************************************************/
BOOST_AUTO_TEST_SUITE( graybat_cage_point_to_point_test )


/*******************************************************************************
 * Communication Policies to Test
 ******************************************************************************/
namespace hana = boost::hana;
using ZMQ  = graybat::communicationPolicy::ZMQ;
using BMPI = graybat::communicationPolicy::BMPI;

BMPI bmpiCP(config);
ZMQ zmqCP(config);

auto communicationPolicies = hana::make_tuple(std::ref(bmpiCP),
					      std::ref(zmqCP));


/***************************************************************************
 * Test Cases
 ****************************************************************************/
BOOST_AUTO_TEST_CASE( send_recv ){
    hana::for_each(communicationPolicies, [](auto refWrap){
	    // Test setup
	    using CP      = typename decltype(refWrap)::type;
	    using GP      = graybat::graphPolicy::BGL<>;
	    using Cage    = graybat::Cage<CP, GP>;
	    using Event   = typename Cage::Event;
	    using Vertex  = typename Cage::Vertex;
	    using Edge    = typename Cage::Edge;
	    CP& cp = refWrap.get();
	    Progress progress(cp);

	    // Test run
	    {	
    		const unsigned nElements = 1000;

		Cage cage (cp);
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

		    progress.print(nRuns, run_i);	

		}

	    }
	    
	});

}


BOOST_AUTO_TEST_CASE( multi_cage ){
    hana::for_each(communicationPolicies, [](auto refWrap){
	    // Test setup
	    using CP      = typename decltype(refWrap)::type;
	    using GP      = graybat::graphPolicy::BGL<>;
	    using Cage    = graybat::Cage<CP, GP>;
	    using Event   = typename Cage::Event;
	    using Vertex  = typename Cage::Vertex;
	    using Edge    = typename Cage::Edge;
	    CP& cp = refWrap.get();
	    Progress progress(cp);

	    // Test run
	    {
		
		Cage cage1(cp);
		Cage cage2(cp);

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
	});

    
}

BOOST_AUTO_TEST_CASE( asyncSend_recv ){
    hana::for_each(communicationPolicies, [](auto refWrap){
	    // Test setup
	    using CP      = typename decltype(refWrap)::type;
	    using GP      = graybat::graphPolicy::BGL<>;
	    using Cage    = graybat::Cage<CP, GP>;
	    using Event   = typename Cage::Event;
	    using Vertex  = typename Cage::Vertex;
	    using Edge    = typename Cage::Edge;
	    CP& cp = refWrap.get();
	    Progress progress(cp);

	    // Test run
	    {

		Cage cage(cp);    
    
		cage.setGraph(graybat::pattern::FullyConnected(cage.getPeers().size()));
		cage.distribute(graybat::mapping::Consecutive());

		const unsigned nElements = 1000;

		for(unsigned run_i = 0; run_i < nRuns; ++run_i){
		    std::vector<Event> events; 
		    std::vector<unsigned> send(nElements,0);
		    std::vector<unsigned> recv(nElements,0);

		    for(unsigned i = 0; i < send.size();++i){
			send.at(i) = i;
		    }

		    // Send state to neighbor cells
		    for(Vertex &v : cage.hostedVertices){
			for(Edge edge : cage.getOutEdges(v)){
			    cage.send(edge, send, events);
			}
		    }

		    // Recv state from neighbor cells
		    for(Vertex &v : cage.hostedVertices){
			for(Edge edge : cage.getInEdges(v)){
			    cage.recv(edge, recv);
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

		    progress.print(nRuns, run_i);	

		}

	    }

	});

}





BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE( graybat_cage_collective_test )

// BOOST_AUTO_TEST_CASE( reduce ){
//     grid.setGraph(graybat::pattern::Grid(3,3));
//     grid.distribute(graybat::mapping::Consecutive());

//     const unsigned nElements = 10;
    
//     std::vector<unsigned> send(nElements, 1);
//     std::vector<unsigned> recv(nElements, 0);

//     Vertex rootVertex = grid.getVertex(0);
    
//     for(Vertex v: grid.hostedVertices){
// 	grid.reduce(rootVertex, v, std::plus<unsigned>(), send, recv);
//     }

//     if(grid.peerHostsVertex(rootVertex)){
// 	for(unsigned receivedElement: recv){
// 	    BOOST_CHECK_EQUAL(receivedElement, grid.getVertices().size());
// 	}
//     }
    
// }


// BOOST_AUTO_TEST_CASE( allReduce ){
//     grid.setGraph(graybat::pattern::Grid(grid.getPeers().size(), grid.getPeers().size()));
//     grid.distribute(graybat::mapping::Consecutive());

//     const unsigned nElements = 10;
    
//     std::vector<unsigned> send(nElements, 1);
//     std::vector<unsigned> recv(nElements, 0);

//     for(Vertex v: grid.hostedVertices){
// 	grid.allReduce(v, std::plus<unsigned>(), send, recv);
//     }

//     for(unsigned receivedElement: recv){
// 	BOOST_CHECK_EQUAL(receivedElement, grid.getVertices().size());
//     }
    
// }

// BOOST_AUTO_TEST_CASE( gather ){
//     grid.setGraph(graybat::pattern::Grid(3,3));
//     grid.distribute(graybat::mapping::Consecutive());

//     const unsigned nElements = 10;
//     const unsigned testValue = 1;
//     const bool reorder = true;

    
//     std::vector<unsigned> send(nElements, testValue);
//     std::vector<unsigned> recv(nElements * grid.getVertices().size(), 0);

//     Vertex rootVertex = grid.getVertex(0);
    
//     for(Vertex v: grid.hostedVertices){
// 	grid.gather(rootVertex, v, send, recv, reorder);
//     }

//     if(grid.peerHostsVertex(rootVertex)){
// 	for(unsigned receivedElement: recv){
// 	    BOOST_CHECK_EQUAL(receivedElement, testValue);
// 	}
//     }
    
    
// }

// BOOST_AUTO_TEST_CASE( allGather ){
//     grid.setGraph(graybat::pattern::Grid(grid.getPeers().size(), grid.getPeers().size()));
//     grid.distribute(graybat::mapping::Consecutive());

//     const unsigned nElements = 10;
//     const unsigned testValue = 1;
//     const bool reorder = true;

    
//     std::vector<unsigned> send(nElements, testValue);
//     std::vector<unsigned> recv(nElements * grid.getVertices().size(), 0);

//     for(Vertex v: grid.hostedVertices){
// 	grid.allGather(v, send, recv, reorder);
//     }

//     for(unsigned receivedElement: recv){
// 	BOOST_CHECK_EQUAL(receivedElement, testValue);
//     }
        
// }

// BOOST_AUTO_TEST_CASE( spreadAndCollect ){
//     star.setGraph(graybat::pattern::InStar(star.getPeers().size()));
//     star.distribute(graybat::mapping::Consecutive());

//     const unsigned nElements = 10;
//     const unsigned testValue = 1;
//     std::vector<Event> events; 
    
//     std::vector<unsigned> send(nElements, testValue);


//     for(Vertex v: star.hostedVertices){
// 	v.spread(send, events);
//     }

//     for(Vertex v: star.hostedVertices){
// 	std::vector<unsigned> recv(v.nInEdges() * nElements, 0);
// 	v.collect(recv);
// 	for(unsigned receivedElement: recv){
// 	    BOOST_CHECK_EQUAL(receivedElement, testValue);
// 	}
	
//     }

//     for(unsigned i = 0; i < events.size(); ++i){
// 	events.back().wait();
// 	events.pop_back();
//     }
  
// }



// BOOST_AUTO_TEST_CASE( multi_cage ){

// 	CP communicationPolicy1(masterUri, peerUri, contextSize);          
// 	Cage cage1(communicationPolicy1);
// 	cage1.setGraph(graybat::pattern::FullyConnected(cage1.getPeers().size()));
// 	cage1.distribute(graybat::mapping::Roundrobin());

// 	CP communicationPolicy2(masterUri, peerUri, contextSize);              
// 	Cage cage2(communicationPolicy2);
// 	cage2.setGraph(graybat::pattern::FullyConnected(cage2.getPeers().size()));
// 	cage2.distribute(graybat::mapping::Roundrobin());
	
// }

BOOST_AUTO_TEST_SUITE_END()

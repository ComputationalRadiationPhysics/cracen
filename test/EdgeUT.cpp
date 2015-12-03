// BOOST
#include <boost/test/unit_test.hpp>
#include <boost/hana/tuple.hpp>

// STL
#include <vector>
#include <functional> // std::plus

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/mapping/Consecutive.hpp>
#include <graybat/pattern/Grid.hpp>
#include <graybat/pattern/Chain.hpp>

/***************************************************************************
 * Test Suites
 ****************************************************************************/
BOOST_AUTO_TEST_SUITE( edge )

/*******************************************************************************
 * Communication Policies to Test
 ******************************************************************************/
namespace hana = boost::hana;

using ZMQ        = graybat::communicationPolicy::ZMQ;
using BMPI       = graybat::communicationPolicy::BMPI;
using GP         = graybat::graphPolicy::BGL<>;
using ZMQCage    = graybat::Cage<ZMQ, GP>;
using BMPICage   = graybat::Cage<BMPI, GP>;
using ZMQConfig  = ZMQ::Config;
using BMPIConfig = BMPI::Config;

ZMQConfig zmqConfig = {"tcp://127.0.0.1:5000",
                       "tcp://127.0.0.1:5001",
                       static_cast<size_t>(std::stoi(std::getenv("OMPI_COMM_WORLD_SIZE")))};

BMPIConfig bmpiConfig;

ZMQCage zmqCage(zmqConfig);
BMPICage bmpiCage(bmpiConfig);

auto cages = hana::make_tuple(std::ref(zmqCage),
                              std::ref(bmpiCage) );


/***************************************************************************
 * Test Cases
 ****************************************************************************/

BOOST_AUTO_TEST_CASE( send_recv ){
        hana::for_each(cages, [](auto cageRef){
	    // Test setup
            using Cage    = typename decltype(cageRef)::type;
            using GP      = typename Cage::GraphPolicy;            
	    using Event   = typename Cage::Event;
	    using Vertex  = typename Cage::Vertex;
	    using Edge    = typename Cage::Edge;

	    // Test run
	    {	
    
		std::vector<Event> events;
                auto& grid = cageRef.get();
 
		grid.setGraph(graybat::pattern::Grid<GP>(grid.getPeers().size(),
                                                         grid.getPeers().size()));
		
		grid.distribute(graybat::mapping::Consecutive());

		const unsigned nElements = 10;
		const unsigned testValue = 5;
    
		std::vector<unsigned> send(nElements, testValue);
		std::vector<unsigned> recv(nElements, 0);
    
    
		for(Vertex v : grid.hostedVertices){
		    for(Edge edge : grid.getOutEdges(v)){
			Event e = edge << send;
			events.push_back(e);
		    }
        
		}

		for(Vertex v : grid.hostedVertices){
		    for(Edge edge : grid.getInEdges(v)){
			edge >> recv;
			for(unsigned u : recv){
			    BOOST_CHECK_EQUAL(u, testValue);
			}
		    }
        
		}

		// Wait to finish events
		for(unsigned i = 0; i < events.size(); ++i){
		    events.back().wait();
		    events.pop_back();
		}

	    }
	    
	});
    
}

BOOST_AUTO_TEST_SUITE_END()

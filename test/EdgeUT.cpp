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

/***************************************************************************
 * Test Suites
 ****************************************************************************/
BOOST_AUTO_TEST_SUITE( edge )

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

BOOST_AUTO_TEST_CASE( send_recv){
    hana::for_each(communicationPolicies, [](auto refWrap){
	    // Test setup
	    using CP      = typename decltype(refWrap)::type;
	    using GP      = graybat::graphPolicy::BGL<>;
	    using Cage    = graybat::Cage<CP, GP>;
	    using Event   = typename Cage::Event;
	    using Vertex  = typename Cage::Vertex;
	    using Edge    = typename Cage::Edge;
	    CP& cp = refWrap.get();

	    // Test run
	    {	
		std::vector<Event> events;

		Cage grid(cp);

		grid.setGraph(graybat::pattern::Grid(grid.getPeers().size(),
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

// BOOST
#include <boost/test/unit_test.hpp>

// STL
#include <vector>
#include <functional> // std::plus

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/mapping/Consecutive.hpp>
#include <graybat/pattern/Grid.hpp>
#include <graybat/pattern/Chain.hpp>


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

BOOST_AUTO_TEST_SUITE(edge)

CP communicationPolicy;
Cage grid(communicationPolicy);

BOOST_AUTO_TEST_CASE( send_recv){
    std::vector<Event> events;


    grid.setGraph(graybat::pattern::Grid(grid.getPeers().size(), grid.getPeers().size()));
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




BOOST_AUTO_TEST_SUITE_END()

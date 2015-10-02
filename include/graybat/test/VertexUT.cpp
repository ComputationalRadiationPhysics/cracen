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

BOOST_AUTO_TEST_SUITE(vertex)

CP communicationPolicy;
Cage grid(communicationPolicy);
Cage chain(communicationPolicy);

BOOST_AUTO_TEST_CASE( spread_collect ){
    std::vector<Event> events;


    grid.setGraph(graybat::pattern::Grid(grid.getPeers().size(), grid.getPeers().size()));
    grid.distribute(graybat::mapping::Consecutive());


    
    const unsigned nElements = 10;
    const unsigned testValue = 5;
    
    std::vector<unsigned> send(nElements, testValue);

    
    for(Vertex v : grid.hostedVertices){
        v.spread(send, events);
        
    }

    for(Vertex v : grid.hostedVertices){
        std::vector<unsigned> recv(nElements * v.nInEdges(), 0);
        v.collect(recv);
        for(unsigned r: recv){
            BOOST_CHECK_EQUAL(r, testValue);
            
        }
        
    }

    // Wait to finish events
    for(unsigned i = 0; i < events.size(); ++i){
	events.back().wait();
	events.pop_back();
    }
    
}


BOOST_AUTO_TEST_CASE( accumulate ){
    std::vector<Event> events;

    grid.setGraph(graybat::pattern::Grid(grid.getPeers().size(), grid.getPeers().size()));
    grid.distribute(graybat::mapping::Consecutive());
    
    const unsigned nElements = 1;
    const unsigned testValue = 5;
    
    std::vector<unsigned> send(nElements, testValue);

    
    for(Vertex v : grid.hostedVertices){
        v.spread(send, events);
        
    }

    for(Vertex v : grid.hostedVertices){
        BOOST_CHECK_EQUAL(v.accumulate(std::plus<unsigned>(), 0), testValue * nElements * v.nInEdges());
                
    }

    // Wait to finish events
    for(unsigned i = 0; i < events.size(); ++i){
	events.back().wait();
	events.pop_back();
    }
    
}

BOOST_AUTO_TEST_CASE( forward ){
    std::vector<Event> events;

    chain.setGraph(graybat::pattern::Chain(grid.getPeers().size()));
    chain.distribute(graybat::mapping::Consecutive());
    
    const unsigned nElements = 100;
    const unsigned testValue = 1;
    
    std::vector<unsigned> input(nElements, testValue);
    std::vector<unsigned> output(nElements, 0);
    std::vector<unsigned> intermediate(nElements, 0);

    const Vertex entry = chain.getVertex(0);
    const Vertex exit  = chain.getVertex(chain.getVertices().size()-1);
    
    
    for(Vertex v : chain.hostedVertices){

        if(v == entry){
            v.spread(input, events);
            
        }

        if(v == exit){
            v.collect(output);
            for(unsigned u : output){
                if(chain.getVertices().size() == 1)
                    BOOST_CHECK_EQUAL(u, testValue);
                if(chain.getVertices().size() == 2)
                    BOOST_CHECK_EQUAL(u, testValue);
                if(chain.getVertices().size() > 2)
                    BOOST_CHECK_EQUAL(u, testValue + (1 * chain.getVertices().size() - 2));
                    
            }
            
        }

        if(v != entry and v != exit){
            v.forward(intermediate, [](std::vector<unsigned> &v)->void{for(unsigned &u : v){u++;}});
            
        }
            
        
    }

    // Wait to finish events
    for(unsigned i = 0; i < events.size(); ++i){
	events.back().wait();
	events.pop_back();
    }
    
}


BOOST_AUTO_TEST_SUITE_END()

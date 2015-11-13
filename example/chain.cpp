/**
 * @example chain.cpp
 *
 * @brief Data is send through a chain of compute 
 *        nodes and every node increments the value.
 *
 */

// STL
#include <iostream>   /* std::cout */
#include <vector>     /* std::vector */
#include <array>      /* std::array */
#include <functional> /* std::bind */
#include <cmath>      /* sqrt */
#include <cstdlib>    /* atoi */
#include <numeric>    /* std::accumulate */

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>
#include <graybat/graphPolicy/BGL.hpp>
// GRAYBAT mappings
#include <graybat/mapping/Consecutive.hpp>
#include <graybat/mapping/Random.hpp>
#include <graybat/mapping/Roundrobin.hpp>
// GRAYBAT pattern
#include <graybat/pattern/Chain.hpp>

struct Config {

};

int exp() {
    /***************************************************************************
     * Configuration
     ****************************************************************************/

    // CommunicationPolicy
    typedef graybat::communicationPolicy::BMPI CP;
    
    // GraphPolicy
    typedef graybat::graphPolicy::BGL<>    GP;
    
    // Cage
    typedef graybat::Cage<CP, GP> Cage;
    typedef typename Cage::Event  Event;
    typedef typename Cage::Vertex Vertex;

    /***************************************************************************
     * Initialize Communication
     ****************************************************************************/
    const unsigned nChainLinks = 1000;
    auto inc = [](unsigned &a){a++;};

    // Create GoL Graph
    Config config;
    CP communicationPolicy(config);
    Cage cage(communicationPolicy);

    cage.setGraph(graybat::pattern::Chain(nChainLinks));
    
    // Distribute vertices
    cage.distribute(graybat::mapping::Consecutive());

    /***************************************************************************
     * Run Simulation
     ****************************************************************************/
    std::vector<Event> events;

    std::array<unsigned, 1> input {{0}};
    std::array<unsigned, 1> output {{0}};
    std::array<unsigned, 1> intermediate {{0}};

    const Vertex entry = cage.getVertex(0);
    const Vertex exit  = cage.getVertex(nChainLinks-1);


    for(Vertex v : cage.hostedVertices){

	if(v == entry){
	    v.spread(input, events);
	    std::cout << "Input: " << input[0] << std::endl;
	}

	if(v == exit){
	    v.collect(output);
	    std::cout << "Output: " << output[0] << std::endl;
	}

	if(v != entry and v != exit){
	    v.collect(intermediate);
	    inc(intermediate[0]);
	    std::cout << "Increment: " << intermediate[0] << std::endl;
	    v.spread(intermediate, events);
	    
	}
	
    }

    for(unsigned i = 0; i < events.size(); ++i){
	events.back().wait();
	events.pop_back();
    }
    
    return 0;

}

int main(){
    exp();
    return 0;
}

/**
 * @example forward.cpp
 *
 * @brief Data is forwarded through a chain of compute 
 *        nodes.
 *
 */

// STL
#include <iostream>   /* std::cout */
#include <vector>     /* std::vector */
#include <array>      /* std::array */
#include <functional> /* std::bind */

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>
#include <graybat/graphPolicy/BGL.hpp>
// GRAYBAT mappings
#include <graybat/mapping/Consecutive.hpp>
#include <graybat/mapping/Random.hpp>
#include <graybat/mapping/Roundrobin.hpp>
// GRAYBAT pattern
#include <graybat/pattern/GridDiagonal.hpp>
#include <graybat/pattern/Chain.hpp>

struct Function {
    
    void process(std::array<unsigned, 1> &data){
	foo(std::get<0>(data));
	
    }

    std::function<void (unsigned&)> foo;
};


int exp() {
    /***************************************************************************
     * Configuration
     ****************************************************************************/

    // CommunicationPolicy
    typedef graybat::communicationPolicy::BMPI CP;
    typedef CP::Config                         Config;
    
    // GraphPolicy
    typedef graybat::graphPolicy::BGL<Function>    GP;
    
    // Cage
    typedef graybat::Cage<CP, GP> Cage;
    typedef typename Cage::Event  Event;
    typedef typename Cage::Vertex Vertex;
    typedef typename Vertex::VertexProperty VertexProperty;

    /***************************************************************************
     * Initialize Communication
     ****************************************************************************/
    const unsigned nChainLinks = 1000;
    
    // Create GoL Graph
    Config config;
    Cage cage(config);

    // Set communication pattern
    cage.setGraph(graybat::pattern::Chain<GP>(nChainLinks));

    // Distribute vertices
    cage.distribute(graybat::mapping::Consecutive());

    // Set functions of vertices
    for(Vertex &v : cage.getVertices()){
	v().foo = [v](unsigned &a)->void {a += v.id;};
    }
        

    /***************************************************************************
     * Run Simulation
     ****************************************************************************/
    std::vector<Event> events;

    std::array<unsigned, 1> input {{0}};
    std::array<unsigned, 1> output {{0}};
    std::array<unsigned, 1> intermediate {{0}};

    const Vertex entry = cage.getVertex(0);
    const Vertex exit  = cage.getVertex(nChainLinks-1);
    
    using namespace std::placeholders;
    
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
	    v.forward(intermediate, std::bind(&VertexProperty::process, v(), _1));
	    std::cout << "Intermediate: " << intermediate[0] << std::endl;
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

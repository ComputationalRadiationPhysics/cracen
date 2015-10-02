// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include <graybat/graphPolicy/BGL.hpp>
#include <graybat/mapping/Roundrobin.hpp>
#include <graybat/pattern/BiStar.hpp>

/**
 * @brief Clients start communication with "Hello", Server answers with "World".
 *
 * This example implements the request and reply example of ZMQ[1]
 * in graybat style.
 *
 * [1] http://zguide.zeromq.org/page:all#Ask-and-Ye-Shall-Receive
 *
 */


int main() {
    /***************************************************************************
     * Configuration
     ****************************************************************************/

    // CommunicationPolicy
    typedef graybat::communicationPolicy::ZMQ CP;
    
    // GraphPolicy
    typedef graybat::graphPolicy::BGL<>        GP;
    
    // Cage
    typedef graybat::Cage<CP, GP> Cage;
    typedef typename Cage::Vertex Vertex;
    typedef typename Cage::Edge   Edge;

    /***************************************************************************
     * Initialize Communication
     ****************************************************************************/
    // Create GoL Graph
    Cage cage;

    // Set communication pattern
    cage.setGraph(graybat::pattern::BiStar(cage.getPeers().size()));
    
    // Map vertices to peers
    //cage.distribute(graybat::mapping::Roundrobin());

    /***************************************************************************
     * Run 
     ****************************************************************************/
    Vertex server = cage.getVertex(0);

    std::array<std::string, 1> hello{{"Hello"}};
    std::array<std::string, 1> world{{"World"}};


    // Cage test
    
    
    /*
    
    while(true){

	for(Vertex v: cage.hostedVertices) {

	    // Server
	    if(v == server){
                // Wait for next request from client
                 Edge recvEdge = cage.recv(hello);
                std::cout << "Received " << hello[0] << std::endl;

                // Send reply back to client
                 cage.send(recvEdge.inverse(), world);
                std::cout << "Send " << world[0] << std::endl;			
		
	    }

	    // Clients
	    if(v != server){
		for(Edge sendEdge : cage.getOutEdges(v)){
		    // Send a hello
		     cage.send(sendEdge, hello);
		    std::cout << "Send " << hello[0] << std::endl;
		
		    // Get the reply
		     cage.recv(sendEdge.inverse(), world);
		    std::cout << "Received " << world[0] << std::endl;
		}
		
	    }

	}
	
    }

    */
    
    return 0;
}

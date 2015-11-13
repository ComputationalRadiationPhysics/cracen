// BOOST
#include <boost/test/unit_test.hpp>
#include <boost/hana/tuple.hpp>

// STL
#include <functional> /* std::plus, std::reference_wrapper */
#include <iostream>   /* std::cout, std::endl */
#include <array>      /* std::array */
#include <numeric>    /* std::iota */
#include <cstdlib>    /* std::getenv */
#include <string>     /* std::string, std::stoi */

// ELEGANT-PROGRESSBARS
#include <elegant-progressbars/policyProgressbar.hpp>
#include <elegant-progressbars/all_policies.hpp>

// GRAYBAT
#include <graybat/Cage.hpp>
#include <graybat/communicationPolicy/ZMQ.hpp>
#include <graybat/communicationPolicy/BMPI.hpp>


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
BOOST_AUTO_TEST_SUITE( graybat_communication_policy_point_to_point_test )


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
BOOST_AUTO_TEST_CASE( context ){
    hana::for_each(communicationPolicies, [](auto refWrap){
	    // Test setup
	    using CP      = typename decltype(refWrap)::type;
	    using Context = typename CP::Context;	    
	    CP& cp = refWrap.get();
	    Progress progress(cp);

	    // Test run
	    {	
		Context oldContext = cp.getGlobalContext();
		for(unsigned i = 0; i < nRuns; ++i){
		    Context newContext = cp.splitContext( true, oldContext);
		    oldContext = newContext;
		    progress.print( nRuns, i);
	
		}
		
	    }

	});
    
}


BOOST_AUTO_TEST_CASE( send_recv){
    hana::for_each(communicationPolicies, [](auto refWrap){
	    // Test setup
	    using CP      = typename decltype(refWrap)::type;
	    using Context = typename CP::Context;
	    using Event = typename CP::Event;	    	    
	    CP& cp = refWrap.get();
	    Progress progress(cp);

	    // Test run
	    {
    
		const unsigned nElements = 10;
		const unsigned tag = 99;

		Context context = cp.getGlobalContext();
    
		for(unsigned i = 0; i < nRuns; ++i){
		    std::vector<unsigned> recv (nElements, 0);
		    std::vector<Event> events;

		    for(unsigned vAddr = 0; vAddr < context.size(); ++vAddr){
			std::vector<unsigned> data (nElements, 0);
			std::iota(data.begin(), data.end(), context.getVAddr());
			events.push_back(cp.asyncSend(vAddr, tag, context, data));
        
		    }

		    for(unsigned vAddr = 0; vAddr < context.size(); ++vAddr){
			cp.recv(vAddr, tag, context, recv);

			for(unsigned i = 0; i < recv.size(); ++i){
			    BOOST_CHECK_EQUAL(recv[i], vAddr+i);
            
			}

		    }

		    for(Event &e : events){
			e.wait();
	    
		    }

		    progress.print( nRuns, i);
	
		}

	    }

	});
    
}
	


BOOST_AUTO_TEST_CASE( send_recv_all){
    hana::for_each(communicationPolicies, [](auto refWrap){
	    // Test setup
	    using CP      = typename decltype(refWrap)::type;
	    using Context = typename CP::Context;
	    using Event = typename CP::Event;	    	    
	    CP& cp = refWrap.get();
	    Progress progress(cp);

	    // Test run
	    {
    
		Context context = cp.getGlobalContext();

		const unsigned nElements = 10;
    
		for(unsigned i = 0; i < nRuns; ++i){
		    std::vector<unsigned> recv (nElements, 0);
		    std::vector<Event> events;

		    for(unsigned vAddr = 0; vAddr < context.size(); ++vAddr){
			std::vector<unsigned> data (nElements, 0);
			std::iota(data.begin(), data.end(), context.getVAddr());
			events.push_back(cp.asyncSend(vAddr, 99, context, data));
        
		    }

		    for(unsigned i = 0; i < context.size(); ++i){
			Event e = cp.recv(context, recv);

			unsigned vAddr = e.source();

			for(unsigned i = 0; i < recv.size(); ++i){
			    BOOST_CHECK_EQUAL(recv[i], vAddr+i);
            
			}

		    }


		    for(Event &e : events){
			e.wait();
	    
		    }

		    progress.print(nRuns, i);	

		}
		
	    }
	    
	});

}


BOOST_AUTO_TEST_CASE( send_recv_order ){
    hana::for_each(communicationPolicies, [](auto refWrap){
	    // Test setup
	    using CP      = typename decltype(refWrap)::type;
	    using Context = typename CP::Context;
	    using Event = typename CP::Event;	    	    
	    CP& cp = refWrap.get();
	    Progress progress(cp);

	    // Test run
	    {
    
		Context context = cp.getGlobalContext();

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
			events.push_back( cp.asyncSend(vAddr, tag, context, data1));
			events.push_back( cp.asyncSend(vAddr, tag, context, data2));
			events.push_back( cp.asyncSend(vAddr, tag, context, data3));
        
		    }

		    for(unsigned vAddr = 0; vAddr < context.size(); ++vAddr){
			cp.recv(vAddr, tag, context, recv1);
			cp.recv(vAddr, tag, context, recv2);
			cp.recv(vAddr, tag, context, recv3);

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
	
		    progress.print(nRuns, run_i);	

		}
		
	    }
	    
	});

}

BOOST_AUTO_TEST_SUITE_END()

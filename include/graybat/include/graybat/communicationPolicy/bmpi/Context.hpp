#pragma once

#include <boost/mpi/environment.hpp>

namespace graybat {
    
    namespace communicationPolicy {

        namespace bmpi {

            /**
	     * @Brief A context represents a set of peers which are
	     *        able to communicate with each other.
	     *
	     */
	    class Context {
		typedef unsigned ContextID;
		typedef unsigned VAddr;
	    
	    public:
		Context() :
		    id(0),
		    isValid(false){

		}

		Context(ContextID contextID, boost::mpi::communicator comm) : 
		    comm(comm),
		    id(contextID),
		    isValid(true){
		
		}

		Context& operator=(const Context& otherContext){
		    id            = otherContext.getID();
		    isValid       = otherContext.valid();
		    comm          = otherContext.comm;
		    return *this;

		}

		size_t size() const{
		    return comm.size();
		}

		VAddr getVAddr() const {
		    return comm.rank();
		}

		ContextID getID() const {
		    return id;
		}

		bool valid() const{
		    return isValid;
		}

		boost::mpi::communicator comm;
	
	    private:	
		ContextID id;
		bool      isValid;
	    };

        } // namespace bmpi
        
    } // namespace communicationPolicy
	
} // namespace graybat

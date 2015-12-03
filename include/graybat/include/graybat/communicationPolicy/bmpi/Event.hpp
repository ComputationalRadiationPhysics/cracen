#pragma once

#include <boost/mpi/environment.hpp>

namespace graybat {
    
    namespace communicationPolicy {

        namespace bmpi {


	    /**
	     * @brief An event is returned by non-blocking 
	     *        communication operations and can be 
	     *        asked whether an operation has finished
	     *        or it can be waited for this operation to
	     *        be finished.
	     *
	     */
	    class Event {
                typedef unsigned Tag;                                            
                typedef unsigned VAddr;
                
	    public:
		Event(boost::mpi::request request) : request(request), async(true){

		}

                Event(boost::mpi::status status) : status(status), async(false){

                }


		~Event(){

		}

		void wait(){
                    if(async){
                        request.wait();
                    }
	
		}

		bool ready(){
                    if(async){
                        boost::optional<boost::mpi::status> status = request.test();

                        if(status){
                            return true;
                        }
                        else {
                            return false;
                        }
                    }
                    return true;

		}

                VAddr source(){
                    if(async){
                        status = request.wait();
                    }
                    return status.source();
                }

                Tag getTag(){
                    if(async){
                        status = request.wait();
                    }
                    return status.tag();

                }

	    private:
		boost::mpi::request request;
                boost::mpi::status  status;
                const bool async;

                
                
	    };

        } // namespace bmpi
        
    } // namespace communicationPolicy
	
} // namespace graybat


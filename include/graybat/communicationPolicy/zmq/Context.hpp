#pragma once

#include <graybat/communicationPolicy/Traits.hpp> 

namespace graybat {
    
    namespace communicationPolicy {
    
        namespace zmq {

            /**
             * @brief A context represents a set of peers which are
             *        able to communicate with each other.
             *
             */
            template<typename T_CP>
            class Context {

                using ContextID = typename graybat::communicationPolicy::ContextID<T_CP>;
                using VAddr     = typename graybat::communicationPolicy::VAddr<T_CP>;
                using Tag       = typename graybat::communicationPolicy::Tag<T_CP>;                
                using MsgType   = typename graybat::communicationPolicy::MsgType<T_CP>;
                using MsgID     = typename graybat::communicationPolicy::MsgID<T_CP>;
	    
            public:
                Context() :
                    contextID(0),
                    vAddr(0),
                    nPeers(1),
                    isValid(false){

                }

                Context(ContextID contextID, VAddr vAddr, unsigned nPeers) :
                    contextID(contextID),
                    vAddr(vAddr),
                    nPeers(nPeers),
                    isValid(true){
		
                }

                size_t size() const{
                    return nPeers;
                }

                VAddr getVAddr() const {
                    return vAddr;
                }

                ContextID getID() const {
                    return contextID;
                }

                bool valid() const{
                    return isValid;
                }

            private:	
                ContextID contextID;
                VAddr     vAddr;
                unsigned  nPeers;
                bool      isValid;		
            };


        } // zmq
        
    } // namespace communicationPolicy
	
} // namespace graybat

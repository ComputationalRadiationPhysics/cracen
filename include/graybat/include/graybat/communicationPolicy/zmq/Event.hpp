#pragma once

#include <graybat/communicationPolicy/Traits.hpp> 

namespace graybat {
    
    namespace communicationPolicy {
    
        namespace zmq {
                        
	    /**
	     * @brief An event is returned by non-blocking 
	     *        communication operations and can be 
	     *        asked whether an operation has finished
	     *        or it can be waited for this operation to
	     *        be finished.
	     *
	     */
            template <typename T_CP>
	    class Event {
	    public:

                using ContextID = typename graybat::communicationPolicy::ContextID<T_CP>;
                using VAddr     = typename graybat::communicationPolicy::VAddr<T_CP>;
                using Tag       = typename graybat::communicationPolicy::Tag<T_CP>;                
                using MsgType   = typename graybat::communicationPolicy::MsgType<T_CP>;
                using MsgID     = typename graybat::communicationPolicy::MsgID<T_CP>;
                using Context   = typename graybat::communicationPolicy::Context<T_CP>;                

		Event(MsgID msgID, Context context, VAddr vAddr, Tag tag, T_CP& comm) :
                    msgID(msgID),
                    context(context),
                    vAddr(vAddr),
                    tag(tag),
                    comm(comm){
		}

		void wait(){
                    comm.wait(msgID, context, vAddr, tag);

		}

                bool ready(){
                    comm.ready(msgID, context, vAddr, tag);
                    return true;
                }

                VAddr source(){
		    return vAddr;
		}

                Tag getTag(){
                    return tag;

                }

                MsgID     msgID;
                Context   context;
                VAddr     vAddr;
                Tag       tag;
                T_CP&      comm;

                
	    };

        } // zmq
        
    } // namespace communicationPolicy
	
} // namespace graybat

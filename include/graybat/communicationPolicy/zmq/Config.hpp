#pragma once

namespace graybat {
    
    namespace communicationPolicy {
    
        namespace zmq {

            struct Config {
                std::string masterUri;
                std::string peerUri;
                size_t contextSize;	    
            };

        } // zmq
        
    } // namespace communicationPolicy
	
} // namespace graybat

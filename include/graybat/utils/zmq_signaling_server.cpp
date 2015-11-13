
// CLIB
#include <string.h>   /* strup */

// STL
#include <iostream> /* std::cout, std::endl */
#include <map>      /* std::map */
#include <sstream>  /* std::stringstream */

// ZMQ
#include <zmq.hpp>

// Type defs
typedef unsigned Tag;                                            
typedef unsigned ContextID;
typedef unsigned VAddr;
typedef unsigned MsgType;
typedef std::string Uri;

// Message tags
static const MsgType VADDR_REQUEST   = 0;
static const MsgType VADDR_LOOKUP    = 1;
static const MsgType DESTRUCT        = 2;
static const MsgType RETRY           = 3;
static const MsgType ACK             = 4;
static const MsgType CONTEXT_INIT    = 5;
static const MsgType CONTEXT_REQUEST = 6;

//  Receive 0MQ string from socket and convert into C string
//  Caller must free returned string. Returns NULL if the context
//  is being terminated.
static char * s_recv (zmq::socket_t& socket) {
    zmq::message_t message(256);
    socket.recv(&message);
    if (message.size() == static_cast<size_t>(-1))
	return NULL;
    if (message.size() > 255)
	static_cast<char*>(message.data())[255] = 0;
    return strdup (static_cast<char*>(message.data()));
}

//  Convert C string to 0MQ string and send to socket
static int s_send (zmq::socket_t& socket, const char *string) {
    zmq::message_t message(sizeof(char) * strlen(string));
    memcpy (static_cast<char*>(message.data()), string, sizeof(char) * strlen(string));
    socket.send(message);
    return 0;
}

int main(){
    std::cout << "Start zmq connection manager" << std::endl;

    std::map<ContextID, std::map<VAddr, Uri> > phoneBook;
    std::map<ContextID, VAddr> maxVAddr;
    std::string masterUri = "tcp://127.0.0.1:5000";
    zmq::context_t context(1);
    
    zmq::message_t request;
    zmq::message_t reply;
    zmq::socket_t socket (context, ZMQ_REP);
    const int hwm = 10000;
    socket.setsockopt( ZMQ_RCVHWM, &hwm, sizeof(hwm));
    socket.setsockopt( ZMQ_SNDHWM, &hwm, sizeof(hwm));

    
    socket.bind(masterUri.c_str());

    ContextID maxContextID = 0;
    ContextID maxInitialContextID = maxContextID;    

    while(true){
        std::stringstream ss;
        ss << s_recv(socket);

        Uri srcUri;
        MsgType type;
	ContextID contextID;
	unsigned size;
        ss >> type;

        switch(type){

	case CONTEXT_INIT:
	    {

		ss >> size;
		if(maxVAddr[maxInitialContextID]  == size){
		    maxInitialContextID = ++maxContextID;
		    maxVAddr[maxInitialContextID] = 0;
		}
		s_send(socket, (std::to_string(maxInitialContextID) + " ").c_str());
		std::cout << "CONTEXT INIT [size:" << size << "]: " << maxInitialContextID << std::endl;		
		break;
	    }

	case CONTEXT_REQUEST:
	    {
		ss >> size;
		maxVAddr[++maxContextID] = 0;
		s_send(socket, (std::to_string(maxContextID) + " ").c_str());
		std::cout << "CONTEXT REQUEST [size:" << size << "]: " << maxContextID << std::endl;				
		break;

	    }
	    
        case VADDR_REQUEST:
            {
                // Reply with correct information
		ss >> contextID;
                ss >> srcUri;

                phoneBook[contextID][maxVAddr[contextID]] = srcUri;
                s_send(socket, (std::to_string(maxVAddr[contextID]) + " ").c_str());
		std::cout << "VADDR REQUEST [contextID:" << contextID << "][srcUri:" << srcUri << "]: " << maxVAddr[contextID] << std::endl;
		maxVAddr[contextID]++;		
                break;
            }
                        
        case VADDR_LOOKUP:
            {
                VAddr remoteVAddr;
		ss >> contextID;
                ss >> remoteVAddr;		
		
                std::stringstream sss;

                if(phoneBook[contextID].count(remoteVAddr) == 0){
                    sss << RETRY;
                    s_send(socket, sss.str().c_str());
		    std::cout << "VADDR LOOKUP [contextID:" << contextID << "][remoteVAddr:" << remoteVAddr << "]: " << " RETRY"<< std::endl;		    		    
                }
                else {
                    sss << ACK << " " << phoneBook[contextID][remoteVAddr] << " ";
                    s_send(socket, sss.str().c_str());
		    std::cout << "VADDR LOOKUP [contextID:" << contextID << "][remoteVAddr:" << remoteVAddr << "]: " << phoneBook[contextID][remoteVAddr] << std::endl;
                }

                break;
            }

        case DESTRUCT:
            s_send(socket, " ");
	    std::cout << "DESTRUCT" << std::endl;
            break;
                        
        default:
            // Reply empty message
            s_send(socket, " ");
	    std::cout << "UNKNOWN MESSAGE TYPE" << std::endl;
	    exit(0);
            break;

        };
                    
    }

}

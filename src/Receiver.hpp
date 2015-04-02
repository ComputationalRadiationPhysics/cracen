#ifndef RECEIVER_HPP
#define RECEIVER_HPP

#include <boost/asio.hpp>
#include "Constants.hpp"
#include "Types.hpp"

class Receiver {
private:
	unsigned int port;
	boost::asio::io_service io_service;
	boost::asio::ip::udp::socket socket;
	boost::asio::ip::udp::endpoint sender_endpoint;
  	Chunk chunk;
  	unsigned int receivedPackages;
	const unsigned int maxPackageLength = sizeof(short)*SAMPLE_COUNT;
  	
public:
	Receiver(const unsigned int port);
	int connect();
	Chunk receive();
	int disconnect();
};

#endif

#ifndef SENDER_HPP
#define SENDER_HPP

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <boost/asio.hpp>

class Sender {
private:
	const char* host;
	const char* port;
	boost::asio::io_service io_service;
    boost::asio::ip::udp::socket socket;
    boost::asio::ip::udp::endpoint endpoint;
    unsigned int packagesSent;
public:
	Sender(const char* host, const char* port);
	int connect();
	int transmit(char* data, size_t length);
	int disconnect();
};

#endif

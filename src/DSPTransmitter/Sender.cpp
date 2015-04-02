#include <cstdlib>
#include <cstring>
#include <iostream>
#include <boost/asio.hpp>

#include "Sender.hpp"

using boost::asio::ip::udp;
using boost::asio::ip::basic_resolver_query;

Sender::Sender(const char* host, const char* port) :
	host(host),
	port(port),
	io_service(),
	socket(io_service, udp::endpoint(udp::v4(), 0)),
	packagesSent(0)
{
	io_service.run();
}
int Sender::connect() {
	boost::system::error_code ec;
    udp::resolver resolver(io_service);
    udp::resolver::query query = {udp::v4(), host, port};
	endpoint = *resolver.resolve(query, ec);
	if(ec != 0) {
		std::cerr << "An error while resolving the host." << std::endl;
		std::cerr << "Code: " << ec << std::endl;
		return 1;
	}
	socket.connect(endpoint, ec);
	if(ec == 0) {
		return 0;
	} else {
		std::cerr << "An error occured during connection." << std::endl;
		std::cerr << "Code: " << ec << std::endl;
		return 1;
	}
}
int Sender::transmit(char* data, size_t length) {
	boost::system::error_code ec;
	socket.send(boost::asio::buffer(data, length), NULL, ec);
	if(ec == 0) {
		packagesSent++;
		return 0;
	} else {
		std::cerr << "An error occured during transmittion of data." << std::endl;
		std::cerr << "Code: " << ec << std::endl;
		return 1;
	}
}
int Sender::disconnect() {
	std::cout << "Packages sent: " << packagesSent << std::endl;
	socket.close();
	return 0;
};


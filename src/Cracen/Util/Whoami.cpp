#include "Whoami.hpp"

#include <boost/asio.hpp>

#include <iostream>

using boost::asio::ip::udp;
std::string Whoami(std::string serverIp) {
	try {
		boost::asio::io_service io_service;
		udp::resolver resolver(io_service);
		udp::resolver::query query(udp::v4(), serverIp, "");
		udp::endpoint receiver_endpoint = *resolver.resolve(query);
		
		udp::socket socket(io_service);
		socket.connect(receiver_endpoint);
		return socket.local_endpoint().address().to_string();
	} catch (std::exception& e){
		std::cerr << "Error: Could not resolve own ip address. Maybe the signaling server can not be resolved." << std::endl;
		std::cerr << "Exception: " << e.what() << std::endl;
		return "Error: Could not resolve own ip address.";
	}
}
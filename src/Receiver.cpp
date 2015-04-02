#include <cstdlib>
#include <iostream>
#include <boost/asio.hpp>
#include "Constants.hpp"
#include "Receiver.hpp"

using boost::asio::ip::udp;

Receiver::Receiver(const unsigned int port) :
	port(port),
	io_service(),
	socket(io_service, udp::endpoint(udp::v4(), port)),
	receivedPackages(0)
{
	receive();
	io_service.run();
}

int Receiver::connect() {
	boost::system::error_code ec;
	socket.connect(udp::endpoint(udp::v4(), port), ec);
	if(ec != 0) {
		std::cerr << "An error occured during connection." << std::endl;
		std::cerr << "Error code: " << ec << std::endl;
		return 1;
	}
	return 0;
}
	
Chunk Receiver::receive() {
	Chunk chunk;
	char data[maxPackageLength]; //Allocate new memory for package
	boost::asio::mutable_buffers_1 buffer(data, maxPackageLength);
	boost::system::error_code ec;
	bool eos = false;
	while(!eos) {
		//Collect one complete Chunk of data
		for(unsigned int i = 0; i < CHUNK_COUNT; i++) {
			socket.receive(buffer, 0, ec);
			if(ec == 0) {
				if(boost::asio::buffer_size(buffer) == 0) {
					//Empty payload package signals end of stream
					eos = true;
					break;
				}
				receivedPackages++;
				MeasureType* formattedData = reinterpret_cast<MeasureType*>(data);
				for(unsigned int j = 0; j < SAMPLE_COUNT; j++) {
					chunk[i*SAMPLE_COUNT+j] = static_cast<DATATYPE>(formattedData[j]); //Conversion from meassured type to a more gpu friendly type
				}
			} else {
				std::cerr << "Error receiving a package." << std::endl;
				std::cerr << "Error code:" << ec << std::endl;
			}
		}
		//Save Chunk to Ringbuffer 
	 }
	 return chunk;
}

int Receiver::disconnect() {
	std::cout << "Received Packages: " << receivedPackages << std::endl;
	socket.close();
	return 0;
}

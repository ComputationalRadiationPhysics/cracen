#include <iostream>

#include "Sender.hpp"

int main(int argc, char* argv[]) {
	char* host;
	char* port;
	if(argc > 2) {
		host = argv[1];
		port = argv[2]; 
	} else {
		std::cout << "Usage of Transmitter: ./DSPTransmitter <Host> <Port>" << std::endl;
	}
	
	Sender sender(host, port);
	sender.connect();
	/*
	while(!eof) {
		sender.transmit(data);
	}
	*/
	sender.disconnect();
}

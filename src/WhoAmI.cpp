#include <iostream>
#include "Utility/Whoami.hpp"

int main(int argc, char** argv) {
	if(argc != 2) {
		std::cout << "Usage: WhoAmI <Remote Host>" << std::endl;
		return 0;
	}
	std::string localIp = Whoami(argv[1]);
	
	std::cout << "Local IP " << localIp << std::endl;
	
	return 0;
}
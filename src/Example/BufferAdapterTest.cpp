#include "Util/BufferAdapter.hpp"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring> // std::memcpy
#include <typeinfo>

using namespace Util;

bool failed;

struct TrivialType {
	int x;
	char y;
};

bool operator==(const TrivialType& lhs, const TrivialType& rhs) {
	return lhs.x == rhs.x && lhs.y == rhs.y;
};

bool operator!=(const TrivialType& lhs, const TrivialType& rhs) {
	return !(lhs == rhs);
}

void test(bool expr, std::string error = "") {
	if(!expr) {
		failed = true;
		std::cerr << "Failed:" << error << std::endl;

	}
}

template <class T>
void copyTest(T& source, T& destination) {
	std::cout << "Run copy test on type:" << typeid(T).name() << std::endl;
	test(
		source != destination,
		"Source and destination are initilised with the same value."
	);
	make_buffer_adaptor(source).copyTo(destination);
	test(
		source == destination,
		"Source and destination do not match after the copy operation."
	);
	std::cout << "Test passed" << std::endl;
}
int main() {
	int builtinTypedVariableSource = 5, builtinTypedVariableDestination = 0;
	TrivialType trivialTypedVariableSource {5, 'a'}, trivialTypedVariableDestination {0, ' '};
	std::array<TrivialType, 3> trivialTypedArraySource, trivialTypedArrayDestination;
	trivialTypedArraySource[0] = { 0 , 'a' };
	trivialTypedArraySource[1] = { 1 , 'b' };
	trivialTypedArraySource[2] = { 2 , 'c' };
	std::vector<TrivialType> trivialTypedVectorSource { {0, 'a'}, {1, 'b'}, {2, 'c'}}, trivialTypedVectorDestination;

	copyTest(builtinTypedVariableSource, builtinTypedVariableDestination);
	copyTest(trivialTypedVariableSource, trivialTypedVariableDestination);
	copyTest(trivialTypedArraySource, trivialTypedArrayDestination);
	copyTest(trivialTypedVectorSource, trivialTypedVectorDestination);


	if(!failed) std::cout << "All tests run successfully" << std::endl;
	else std::cout << "Some tests failed." << std::endl;
	return 0;
}

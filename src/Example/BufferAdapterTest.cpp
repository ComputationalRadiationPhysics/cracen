#include "Util/BufferAdapter.hpp"

#include <iostream>
#include "Util/StlAdapter/Vector.hpp"
#include "Util/StlAdapter/String.hpp"
#include "Util/StlAdapter/Deque.hpp"
#include <cstdlib>
#include <cstring> // std::memcpy
#include <typeinfo>
#include <tuple>
#include <bitset>
#include <chrono>


using namespace Util;

bool failed;

struct TrivialType {
	int x;
	char y;
};

struct PrivateType {
private:
	int x;
public:
	int y;
	PrivateType(int x, int y) :
		x(x),
		y(y)
	{}

	friend bool operator==(const PrivateType& lhs, const PrivateType& rhs);

};
bool operator==(const TrivialType& lhs, const TrivialType& rhs) {
	return lhs.x == rhs.x && lhs.y == rhs.y;
};

bool operator!=(const TrivialType& lhs, const TrivialType& rhs) {
	return !(lhs == rhs);
}
bool operator==(const PrivateType& lhs, const PrivateType& rhs) {
	return lhs.x == rhs.x && lhs.y == rhs.y;
};

bool operator!=(const PrivateType& lhs, const PrivateType& rhs) {
	return !(lhs == rhs);
}

bool test(bool expr, std::string error = "") {
	if(!expr) {
		failed = true;
		std::cerr << "Failed:" << error << std::endl;

	}
	return expr;
}

template <class T>
void copyTest(T& source, T& destination) {
	std::cout << "Run copy test on type:" << typeid(T).name() << std::endl;
	bool result = test(
		source != destination,
		"Source and destination are initilised with the same value."
	);
	make_buffer_adaptor(source).copyTo(destination);
	result &= test(
		source == destination,
		"Source and destination do not match after the copy operation."
	);
	if(result) {
		std::cout << "Test passed" << std::endl;
	} else {
		std::cout << "Test failed" << std::endl;
	}
}

struct Foo {
	int* bar;
	Foo() :
		bar(new  int)
	{
		std::cout << "new int" << std::endl;
	}

	Foo(const Foo& cpy) :
		bar(new int)
	{
		*bar = *(cpy.bar);
	}

	~Foo() {
		std::cout << "delete int" << std::endl;
		delete bar;
	}
};

Foo createFoo() {
	return Foo();
}

int main() {
	Foo a;

	a = createFoo();
	*(a.bar) = 5;

	int builtinTypedVariableSource = 5, builtinTypedVariableDestination = 0;
	TrivialType trivialTypedVariableSource {5, 'a'}, trivialTypedVariableDestination {0, ' '};
	PrivateType privateTypeSource(5,15), privateTypeDestination(0,0);
	std::array<TrivialType, 3> trivialTypedArraySource, trivialTypedArrayDestination;
	trivialTypedArraySource[0] = { 0 , 'a' };
	trivialTypedArraySource[1] = { 1 , 'b' };
	trivialTypedArraySource[2] = { 2 , 'c' };
	std::vector<TrivialType> trivialTypedVectorSource { {0, 'a'}, {1, 'b'}, {2, 'c'}}, trivialTypedVectorDestination;
	std::string stringSource = "Hello World", stringDestination;
	std::pair<int, TrivialType> stdPairSource(5, {2, 'g'}), stdPairDestination;
	std::tuple<int, char, float> stdTupleSource {5, 'a', 1.0f}, stdTupleDestination;
	std::deque<int> stdDequeSource { 1, 2, 3, 4, 5} , stdDequeDestination;
	std::bitset<8> stdBitsetSource(42), stdBitsetDestination;
	std::chrono::high_resolution_clock::time_point timePointSource(std::chrono::high_resolution_clock::now()), timePointDestination;
	std::vector<std::string> vectorStringSource {"Hallo", "Welt"}, vectorStringDestination;
	std::vector<std::vector<std::string>>
		vectorVectorStringSource {
			{ "1", "2", "3", "4", "5", "6"},
			{ "a", "b", "c" },
			{ "string", "test" }
		},
		vectorVectorStringDestination;

	copyTest(builtinTypedVariableSource, builtinTypedVariableDestination);
	copyTest(trivialTypedVariableSource, trivialTypedVariableDestination);
	copyTest(privateTypeSource, privateTypeDestination);
	copyTest(trivialTypedArraySource, trivialTypedArrayDestination);
	copyTest(trivialTypedVectorSource, trivialTypedVectorDestination);
	copyTest(stringSource, stringDestination);
	copyTest(stdPairSource, stdPairDestination);
	copyTest(stdTupleSource, stdTupleDestination);
	copyTest(stdDequeSource, stdDequeDestination);
	copyTest(stdBitsetSource, stdBitsetDestination);
	copyTest(timePointSource, timePointDestination);
	copyTest(vectorStringSource, vectorStringDestination);
	copyTest(vectorVectorStringSource, vectorVectorStringDestination);

	if(!failed) std::cout << "All tests run successfully" << std::endl;
	else std::cout << "Some tests failed." << std::endl;
	return 0;
}

#pragma once

#include <type_traits>
#include "Ringbuffer.hpp"

namespace Cracen {

template <class Type, typename enable = void>
class InputBufferEnable;

template <class Type>
class InputBufferEnable<
	Type,
	typename std::enable_if<
		std::is_same<
			Type,
			void
		>::value
	>::type
>
{
public:
	InputBufferEnable(int size, int producer)
	{}
};

template <class Type>
class InputBufferEnable<Type, typename std::enable_if<!std::is_same<Type, void>::value>::type> {
public:
	Ringbuffer<Type> inputBuffer;
	InputBufferEnable(int size, int producer) :
		inputBuffer(size, producer)
	{}
};

template <class Type, typename enable = void>
class OutputBufferEnable;

template <class Type>
class OutputBufferEnable<Type, typename std::enable_if<std::is_same<Type, void>::value>::type> {
public:
	OutputBufferEnable(int size, int producer)
	{}
};

template <class Type>
class OutputBufferEnable<Type,typename std::enable_if<!std::is_same<Type, void>::value>::type> {
public:
	Ringbuffer<Type> outputBuffer;
	OutputBufferEnable(int size, int producer) :
		outputBuffer(size, producer)
	{}
};

}

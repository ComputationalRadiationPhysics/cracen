#pragma once

#include "../BufferAdapter.hpp"
#include <string>

namespace Util {

template <>
struct BufferAdapter<
	std::string
> :
	public MutableBuffer
{

	BufferAdapter(std::string& input) :
		MutableBuffer(
			reinterpret_cast<decltype(MutableBuffer::data)>(&input[0]),
			input.size()*sizeof(char)
		)
	{};

	BufferAdapter(decltype(MutableBuffer::data) data, size_t size) :
		MutableBuffer(
			reinterpret_cast<decltype(MutableBuffer::data)>(data),
			size
		)
	{};

	void copyTo(std::string& destination) {
		destination.resize(size / sizeof(char));
		memcpy(
			&destination[0],
			data,
			size
		);
	}

}; // End of struct BufferAdapter

} // End of namespace Util


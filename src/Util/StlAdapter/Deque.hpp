#pragma once

#include "../BufferAdapter.hpp"
#include <deque>

namespace Util {

template <
	class Type
>
struct BufferAdapter<
	std::deque<Type>,
	typename std::enable_if<
		std::is_trivially_copyable<Type>::value
	>::type
> :
	public MutableBuffer
{

	BufferAdapter(std::deque<Type>& input) :
		MutableBuffer(
			reinterpret_cast<decltype(MutableBuffer::data)>(&input[0]),
			input.size()*sizeof(Type)
		)
	{};

	BufferAdapter(decltype(MutableBuffer::data) data, size_t size) :
		MutableBuffer(
			reinterpret_cast<decltype(MutableBuffer::data)>(data),
			size
		)
	{};

	void copyTo(std::deque<Type>& destination) {
		destination.resize(size / sizeof(Type));
		memcpy(
			&destination[0],
			data,
			size
		);
	}

}; // End of struct BufferAdapter

} // End of namespace Util


#pragma once

#include "../BufferAdapter.hpp"
#include <vector>
#include <cstdint>

namespace Util {

template <
	class Type
>
struct BufferAdapter<
	std::vector<Type>,
	typename std::enable_if<
		linear_memory_check<Type>::value
	>::type
> {

	void* const data;
	const size_t size;

	BufferAdapter(std::vector<Type>& input) :
		data(input.data()),
		size(input.size()*sizeof(Type))
	{};

	BufferAdapter(void* data, size_t size) :
		data(data),
		size(size)
	{};

	void copyTo(std::vector<Type>& destination) {
		destination.resize(size / sizeof(Type));
		memcpy(
			destination.data(),
			data,
			size
		);
	}

}; // End of struct BufferAdapter

template <
	class Type
>
struct BufferAdapter<
	std::vector<Type>,
	typename std::enable_if<
		!linear_memory_check<Type>::value
	>::type
> :
	public MutableBuffer
{
	//static_assert(false, "Not implemented");
	/* Memory Layout
	 *
	 * [size 0][blob 0][size 1][blob 1][size 2][blob 2][size 3][blob 3]...
	 * size is 16 bit unsigned int;
	 * blobs can have sizes up to (2^16 - 1), without the whole package beeing bigger than the max payload of the protocol
	 */

	bool owningData;

	size_t getSize(std::vector<Type>& input) {

		size_t size = 0;

		for(Type& t : input) {
			/* Copy all elements of the vector into a linear space */
			BufferAdapter<Type> ba(t);
			size += sizeof(std::uint16_t);
			size += ba.size;
		}

		return size;
	};

	using PointerType =
		std::remove_pointer<
			std::remove_const<
				decltype(MutableBuffer::data)
			>::type
		>::type;

	BufferAdapter(std::vector<Type>& input) :
		MutableBuffer(
			new PointerType[getSize(input)],
			getSize(input)
		),
		owningData(true)
	{
		auto pos = data;
		for(Type& t : input) {
			/* Copy all elements of the vector into a linear space */
			BufferAdapter<Type> ba(t);
			std::uint16_t size = ba.size;
			std::memcpy(
				pos,
				&size,
				sizeof(size)
			);
			pos += sizeof(size);
			memcpy(
				pos,
				ba.data,
				ba.size
			);
			pos += ba.size;
		}
	};

	BufferAdapter(decltype(MutableBuffer::data) data, size_t size) :
		MutableBuffer(
			data,
			size
		),
		owningData(false)
	{};

	BufferAdapter(BufferAdapter&& copy) :
		MutableBuffer(
			copy.data,
			copy.size
		),
		owningData(copy.owningData)
	{
		copy.owningData = false;
	};

	~BufferAdapter() {
		if(owningData) {
			::operator delete(data);
		}
	}



	void copyTo(std::vector<Type>& destination) {
		destination.clear();
		auto pos = data;
		while(pos < data+size) {
			std::uint16_t subsize = *pos;
			pos += sizeof(subsize);
			BufferAdapter<Type> bs(pos, subsize);
			Type element;
			bs.copyTo(element);
			destination.push_back(element);
			pos += subsize;
		};
	}

}; // End of struct BufferAdapter


} // End of namespace Util

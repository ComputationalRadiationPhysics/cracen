#pragma once

#include <type_traits>
#include <vector>
#include <array>
#include <cstddef> // size_t
#include <cstring> // memcpy

namespace Util {

template <class Type, class enable = void>
struct BufferAdapter; // End of class BufferAdapter

template <
	class Type
>
struct BufferAdapter<
	std::vector<Type>,
	typename std::enable_if<
		std::is_trivially_copyable<Type>::value
	>::type
> {

	const void* data;
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

template <>
struct BufferAdapter<
	std::string
> {

	const void* data;
	const size_t size;

	BufferAdapter(std::string& input) :
		data(input.data()),
		size(input.size()*sizeof(char))
	{};

	BufferAdapter(void* data, size_t size) :
		data(data),
		size(size)
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

template <
	class Type
>
struct BufferAdapter<
	Type,
	typename std::enable_if<
		std::is_trivially_copyable<Type>::value
	>::type
> {
	const void* data;
	const size_t size;

	BufferAdapter(Type& input) :
		data(&input),
		size(sizeof(Type))
	{};

	BufferAdapter(void* data, size_t size) :
		data(data),
		size(size)
	{};

	void copyTo(Type& destination) {
		memcpy(
			&destination,
			data,
			size
		);
	}
}; // End of struct BufferAdapter

template <class Type>
BufferAdapter<Type> make_buffer_adaptor(Type& input) {
	return BufferAdapter<Type>(input);
}

} // End of namespace Util

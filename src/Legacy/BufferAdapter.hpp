// #pragma once
//
// #include <type_traits>
// #include <vector>
// #include <array>
// #include <cstddef> // size_t
// #include <cstring> // memcpy
//
// namespace Util {
//
// template <class Type>
// using  linear_memory_check = std::is_trivially_destructible<Type>;
// //using  linear_memory_check = std::is_pod<Type>;
// //using  linear_memory_check = std::is_trivial<Type>;
// //using  linear_memory_check = std::is_trivially_copyable<Type>;
// //using  linear_memory_check = std::is_standard_layout<Type>;
//
//
// struct MutableBuffer {
// 	std::uint8_t* const data;
// 	const size_t size;
// 	MutableBuffer(std::uint8_t* data, size_t size) :
// 		data(data),
// 		size(size)
// 	{}
// }; // End of struct MutableBuffer
//
// template <class Type, class enable = void>
// struct BufferAdapter; // End of class BufferAdapter
//
//
// template <
// 	class Type
// >
// struct BufferAdapter<
// 	Type,
// 	typename std::enable_if<
// 		linear_memory_check<Type>::value
// 	>::type
// > :
// 	public MutableBuffer
// {
// 	BufferAdapter(Type& input) :
// 		MutableBuffer(
// 			reinterpret_cast<decltype(MutableBuffer::data)>(&input),
// 			sizeof(Type)
// 		)
// 	{};
//
// 	BufferAdapter(decltype(MutableBuffer::data) data, size_t size) :
// 		MutableBuffer(
// 			data,
// 			size
// 		)
// 	{};
//
// 	void copyTo(Type& destination) {
// 		memcpy(
// 			&destination,
// 			data,
// 			size
// 		);
// 	}
// }; // End of struct BufferAdapter
//
// template <class Type>
// BufferAdapter<Type> make_buffer_adaptor(Type& input) {
// 	return BufferAdapter<Type>(input);
// }
//
// } // End of namespace Util

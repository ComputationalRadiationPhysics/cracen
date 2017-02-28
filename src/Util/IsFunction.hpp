#pragma once

#include <type_traits>

namespace Detail {

template <
	class Function,
	class Result,
	class Enable,
	class... Args
>
struct isFunctionImpl {
	static constexpr bool value = false;
};

template <
	class Function,
	Function fn,
	class Result,
	class... Arg
>
struct isFunctionImpl <
	Function,
	Result,
	typename std::enable_if<
		std::is_same<
			Result,
			decltype(std::declval<Function>()(*reinterpret_cast<Arg*>(0)...))
		>::value
	>::type,
	Arg...
> {
	static constexpr bool value = true;
};

} // End of namespace Detail

template <
	class Function,
	class Result,
	class... Args
>
struct isFunction {
	static constexpr bool value = Detail::isFunctionImpl<Function, Result, void, Args...>::value;
};

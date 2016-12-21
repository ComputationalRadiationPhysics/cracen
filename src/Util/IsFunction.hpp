#pragma once

namespace Detail {

template <
	class Function,
	Function fn,
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
	fn,
	Result,
	typename std::enable_if<
		std::is_same<
			Result,
			decltype(fn(*reinterpret_cast<Arg*>(0)...))
		>::value
	>::type,
	Arg...
> {
	static constexpr bool value = true;
};

} // End of namespace Detail

template <
	class Function,
	Function fn,
	class Result,
	class... Args
>
struct isFunction {
	static constexpr bool value = Detail::isFunctionImpl<Function, fn, Result, void, Args...>::value;
};

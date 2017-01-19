#pragma once

template <
	bool enable,
	class Function,
	Function function,
	class ResultType,
	class... Args
>
struct ConditionalCall;


template <
	class Function,
	Function function,
	class ResultType,
	class... Args
>
struct ConditionalCall<
	true,
	Function,
	function,
	ResultType,
	Args...
> {
	ResultType operator()(Args... args) {
		return function(args...);
	};
};

template <
	class Function,
	Function function,
	class ResultType,
	class... Args
>
struct ConditionalCall<
	false,
	Function,
	function,
	ResultType,
	Args...
> {
	ResultType operator()(Args... args) {
		return {};
	};
};

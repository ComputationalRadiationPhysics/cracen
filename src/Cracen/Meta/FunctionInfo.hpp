#pragma once

#include <tuple>
#include <functional>

namespace Cracen {

namespace Meta {

namespace detail {

template <class Funktor, class Result, class... Args>
std::tuple<Args...> paramList(Result(Funktor::*f)(Args...)) {
	return {};
}

template <class Funktor, class Result, class... Args>
Result result(Result(Funktor::*f)(Args...)) {
	return {};
}

} // End of namespace detail

template <class F>
struct FunctionInfo {
	using ParamList = decltype(detail::paramList(&F::operator()));
	using Result = decltype(detail::result(&F::operator()));
};

} // End of namespace Meta

} // End of namespace Cracen

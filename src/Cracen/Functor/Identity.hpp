#pragma once

#include <type_traits>

namespace Cracen {

namespace Functor {

namespace detail {

template <class Type, class enable = void>
struct Identity;

template <class Type>
struct Identity<
	Type,
	typename std::enable_if<!std::is_same<Type,void>::value>::type
>
{
	Type operator()(Type value) {
		return value;
	}
};

template <class Type>
struct Identity<
	Type,
	typename std::enable_if<std::is_same<Type,void>::value>::type
>
{
	Type operator()() {
	}
};

}

template <class Type>
using Identity = detail::Identity<Type>;

} // End of namespace Functor

} // End of namespace Cracen

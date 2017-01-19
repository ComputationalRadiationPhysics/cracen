#pragma once

namespace Cracen {

namespace Meta {

template <bool condition, class Function, class... Args>
struct ConditionalInvoke;

template <class Function, class... Args>
struct ConditionalInvoke<
	true,
	Function,
	Args...
>
{
	using type = Function(Args...);
};

template <class Function, class... Args>
struct ConditionalInvoke<
	false,
	Function,
	Args...
>
{
	using type = Function();
};

} // End of namespace Meta

} // End of namespace Cracen

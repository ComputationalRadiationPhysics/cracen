#pragma once

template <
	class Type,
	bool enabled
>
struct OptionalAttribute;

template <class Type>
struct OptionalAttribute<
	Type,
	false
>
{
	using type = Type;

	template <class... Args>
	OptionalAttribute(Args... args) {

	};

	template <
		class Function,
		Function function,
		class ReturnType,
		class... Args
	>
	ReturnType optionalCall(Args... args) {
		return {};
	}
};

template <class Type>
struct OptionalAttribute<
	Type,
	true
> :
	public Type
{
	using Type::Type;
	using Type::operator=;

	using type = Type;

	template <
		class Function,
		Function function,
		class ReturnType,
		class... Args
	>
	ReturnType optionalCall(Args... args) {
		return (this->*function)(args...);
	}
};

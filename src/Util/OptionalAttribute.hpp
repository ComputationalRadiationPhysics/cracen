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
		class Return,
		class... Args
	>
	Return optionalCall(Return (Type::*function)(Args...), Args... args) {
		return {};
	}

	template <
		class Return,
		class... Args
	>
	Return optionalCall(Return (Type::*function)(Args...) const, Args... args) const {
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
		class Return,
		class... Args
	>
	Return optionalCall(Return (Type::*function)(Args...), Args... args) {
		return (this->*function)(args...);
	}

	template <
		class Return,
		class... Args
	>
	Return optionalCall(Return (Type::*function)(Args...) const, Args... args) const {
		return (this->*function)(args...);
	}
};

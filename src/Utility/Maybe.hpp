#ifndef MAYBE_HPP
#define MAYBE_HPP

template <class Type>
struct Maybe {
	Type value;
	const bool valid;
	
	Maybe(const Type& value) :
		value(value),
		valid(true)
	{}
	
	Maybe(bool valid) :
		value(),
		valid(valid)
	{}
	
	Maybe() :
		value(),
		valid(false)
	{}	
};

#endif
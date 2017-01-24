#ifndef Ringbuffer_HPP
#define Ringbuffer_HPP

#include <vector>
#include <iostream>
#include <semaphore.h>
#include <cassert>
#include <queue>

/*
template <class type>
class Ringbuffer {
	std::queue<type> queue;

public:
	Ringbuffer(const unsigned int bSize, int producer, type defaultItem) {}
	Ringbuffer(const unsigned int bSize, int producer) {}
	int push(type input) { queue.push(input); return 0; };
	//int push(Type &&input) noexcept;
	type pop() { while(queue.size() == 0); type result = queue.front(); queue.pop(); return result;};
	template <class Funktor>
	int popTry(Funktor popFunction) { while(queue.size() == 0); type result = queue.front(); popFunction(result); queue.pop(); return 0;};

	int getSize() const { return queue.size(); };
	bool isEmpty() const { return (queue.size() == 0); };
	bool isFinished() const { return false; };
	void producerQuit() {};
};
*/

/*! Ringbuffer
 *  @brief A ringbuffer template supporting non-host consumers/producers.
 *
 *  Data is written to the head of the buffer and read from the tail.
 *  The buffer will block write attempts if full and block read attempts
 *  if empty.
 *  To enable reading to devices like graphic cards the tail of the
 *  buffer can be reserved. In the reserved state copy operations can
 *  be performed externally. After copying the head needs to be
 *  freed.
 *  The same mechanism is available for writing to the buffer from other
 *  devices.
 *  For data reading/writing from host to host classic write/read methods
 *  are available.
 */


template <class Type>
class Ringbuffer {

private:
	sem_t* mtx;
	sem_t* usage;
	sem_t* space;
	std::vector<Type> buffer;
	unsigned int head, tail;
	unsigned int producer;

public:
	void init();
	Ringbuffer(const unsigned int bSize, int producer, Type defaultItem);
	Ringbuffer(const unsigned int bSize, int producer);
	Ringbuffer(const Ringbuffer& rb) {}
	Ringbuffer& operator=(const Ringbuffer& rb) { return *this; }
	~Ringbuffer() noexcept;
	int push(Type input) noexcept;
	//int push(Type &&input) noexcept;
	Type pop() noexcept;
	template <class Funktor>
	int popTry(Funktor popFunction) noexcept;

	int getSize() const noexcept;
	bool isEmpty() const noexcept;
	bool isFinished() const noexcept;
	void producerQuit() noexcept;

};

template <class Type>
void Ringbuffer<Type>::init()
{
	int semaphoreErrorValue = 0;
	mtx = new sem_t;
	usage = new sem_t;
	space = new sem_t;
	semaphoreErrorValue |= sem_init(mtx, 0, 1);
    semaphoreErrorValue |= sem_init(usage, 0, 0);
    semaphoreErrorValue |= sem_init(space, 0, buffer.size());
	if(semaphoreErrorValue != 0) std::cerr << "Initialization of semaphore failed." << std::endl;
	assert(semaphoreErrorValue == 0);
	//std::cout << "usage << " << usage << std::endl;
	//std::cout << "space << " << space << std::endl;
}

/**
 * Constructor for dynamic size elements.
 *
 * Reserves buffer memory. The buffer holds bSize items. The
 * items consist of itemSize elements of type Type. These elements
 * may be of a dynamic size type but they need to have the same size.
 *
 * \param bSize Amount of items the buffer can hold.
 * \param producer number of producers feeding the buffer.
 * \param defaultItem A default item to store in the buffer. This fixes
 *                    the memory available for variable length types
 *                    like std::vector.
 */
template <class Type>
Ringbuffer<Type>::Ringbuffer(const unsigned int bSize,
                             int producer,
                             Type defaultItem) :
    buffer(bSize, defaultItem),
	head(0),// We write new item to this position
	tail(0),// We read stored item from this position
	producer(producer)
{
	init();
}

/**
 *  Fixed size type Constructor.
 *
 *  For Type with fixed size no defaultItem is needed.
 *
 *  \param bSize Amount of items the buffer can hold.
 *  \param producer Number of producers feeding the buffer.
 */
template <class Type>
Ringbuffer<Type>::Ringbuffer(const unsigned int bSize,
                             int producer) :
	buffer(bSize),
	head(0),// We write new item to this position
	tail(0),// We read stored item from this position
	producer(producer)
{
	init();
}


template <class Type>
Ringbuffer<Type>::~Ringbuffer() noexcept
{
    sem_destroy(mtx);
    sem_destroy(usage);
    sem_destroy(space);
	delete mtx;
	delete usage;
	delete space;
}

/**
 * Write data to the buffer from the host.
 *
 * The call blocks if there is no space available on the buffer or if the
 * buffer is already used by another thread.
 *
 * \param inputOnHost Needs to be on host memory.
 */
template <class Type>
int Ringbuffer<Type>::push(Type input) noexcept
{
	sem_wait(space);   // is there space in buffer?
    sem_wait(mtx);     // lock buffer

    std::swap(buffer.at(head), input);
    head = (head+1) % buffer.size();     // move head

    sem_post(mtx);     // unlock buffer
    sem_post(usage);   // tell them that there is something in buffer

	return 0;
}

/**
 * Write data to the buffer from the host.
 *
 * The call blocks if there is no space available on the buffer or if the
 * buffer is already used by another thread.
 *
 * \param inputOnHost Needs to be on host memory.
 */
/*
template <class Type>
int Ringbuffer<Type>::push(Type &&input) noexcept
{
	sem_wait(space);   // is there space in buffer?
    //sem_wait(mtx);     // lock buffer

    buffer.at(head) = input;
    head = (head+1) % buffer.size();     // move head

    //sem_post(mtx);     // unlock buffer
    sem_post(usage);   // tell them that there is something in buffer

	return 0;
}
*/
/**
 * Read data from the buffer to the host.
 *
 * The call blocks until there is data available in the buffer. The call
 * blocks if the buffer is already used by another thread.
 *
 * \param outputOnHost Pointer to host memory where buffer data is to be
 *        written.
 */
template <class Type>
Type Ringbuffer<Type>::pop() noexcept
{
	Type result;
	sem_wait(usage);   // is there some data in buffer?
	sem_wait(mtx);     // lock buffer

	std::swap(result, buffer.at(tail));
	tail = (tail+1) % buffer.size();     // move tail

	sem_post(mtx);     // unlock buffer
	sem_post(space);   // tell them that there is space in buffer

	return result;
}

template <class Type>
template <class Funktor>
int Ringbuffer<Type>::popTry(Funktor popFunction) noexcept
{
	const int err = sem_trywait(usage);
	if(err == 0) {

		Type result;
		sem_wait(mtx);     // lock buffer

		result.swap(buffer.at(tail));
		tail = (tail+1) % buffer.size();     // move tail

		sem_post(mtx);     // unlock buffer
		sem_post(space);   // tell them that there is space in buffer

		popFunction(result);
		return 0;
	} else {
		return 1;
	}
}
/** Get amount of items stored in buffer.
 * \return Number of items in buffer
 */
template <class Type>
int Ringbuffer<Type>::getSize() const noexcept {
	int full_value;
	sem_getvalue(const_cast<sem_t*>(usage), &full_value);
	return full_value;
}

/** Tell if buffer is empty.
 * \return True if no elements are in buffer. False otherwise.
 */
template <class Type>
bool Ringbuffer<Type>::isEmpty() const noexcept {
	int full_value, empty_value;
	sem_getvalue(const_cast<sem_t*>(usage), &full_value);
	sem_getvalue(const_cast<sem_t*>(space), &empty_value);
	return (full_value == 0) && (empty_value == static_cast<int>(buffer.size()));
}

/** Tell if buffer is empty and will stay empty. This is the case if
 *  all produces ceased to add data and no data is in the buffer.
 * \return True if there are no elements in buffer and all producers
 * announced that they stopped adding elements. False otherwise.
 */
template <class Type>
bool Ringbuffer<Type>::isFinished() const noexcept {
	return (producer==0) && isEmpty();
}

/** Lets a producer announce that it is adding no more elements to the
 *  buffer.
 *  To be called only once per producer. This is not checked.
 */
template <class Type>
void Ringbuffer<Type>::producerQuit() noexcept {
	__sync_sub_and_fetch(&producer,1);
}
#endif

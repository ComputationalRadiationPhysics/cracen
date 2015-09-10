#ifndef Ringbuffer_HPP
#define Ringbuffer_HPP

#include <vector>
#include <iostream>
#include <semaphore.h>

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
    sem_t mtx;
    sem_t usage, space;
    std::vector<Type> buffer;
    unsigned int bufferSize;
    unsigned int head, tail;
    int producer;

public:
    Ringbuffer(const unsigned int bSize, int producer,
               Type defaultItem);
    Ringbuffer(const unsigned int bSize, int producer);
    ~Ringbuffer();
    int writeFromHost(Type &inputOnHost);
    int copyToHost(Type &outputOnHost);
    Type& reserveHead();
    int freeHead();
    Type* reserveTailTry();
    int freeTail();
    int getSize();
    bool isEmpty();
    bool isFinished();
    void producerQuit();
    
    //Not for multithreaded processes! Use with extreme caution.
    std::vector<Type>& getBuffer() {return buffer;}
};

typedef std::vector<float >Chunk;

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
	head(0),// We write new item to this position
	tail(0),// We read stored item from this position
	producer(producer)
{
    buffer.reserve(bSize);
    for (int i=0; i<bSize; i++) {
        // We need to push_back the defaultItem for variable size types 
        // like std::vector. Otherwise the memory is not allocated.
        buffer.push_back(defaultItem);
    }
    sem_init(&mtx, 0, 1);
    sem_init(&usage, 0, 0);
    sem_init(&space, 0, bSize);
    bufferSize = bSize; 
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
    head(0),
    tail(0),
    producer(producer),
    bufferSize(bSize)
{
    buffer.resize(bSize);
    sem_init(&mtx, 0, 1);
    sem_init(&usage, 0, 0);
    sem_init(&space, 0, bSize);
}


template <class Type>
Ringbuffer<Type>::~Ringbuffer()
{
    sem_destroy(&mtx);
    sem_destroy(&usage);
    sem_destroy(&space);
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
int Ringbuffer<Type>::writeFromHost(Type &inputOnHost)
{
    sem_wait(&space);   // is there space in buffer?
    sem_wait(&mtx);     // lock buffer
    
    buffer.at(head) = inputOnHost;
    head = ++head % bufferSize;     // move head

    sem_post(&mtx);     // unlock buffer
    sem_post(&usage);   // tell them that there is something in buffer
    
	return 0;
}

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
int Ringbuffer<Type>::copyToHost(Type &outputOnHost)
{
    sem_wait(&usage);   // is there some data in buffer?
    sem_wait(&mtx);     // lock buffer
    
    outputOnHost = buffer.at(tail);
    tail = ++tail % bufferSize;     // move tail
    
    sem_post(&mtx);     // unlock buffer
    sem_post(&space);   // tell them that there is space in buffer
    
    return 0;
}

/** 
 * Lock head position of buffer to perform write operations externally.
 *
 * The call blocks until there is space available in the buffer.
 *
 * Buffer is blocked until freeHead() is called.
 * \return Pointer to the head of the ringbuffer. One item of <Type> can
 *         be written here.
 */
template <class Type>
Type& Ringbuffer<Type>::reserveHead()
{
    sem_wait(&space);
    sem_wait(&mtx);
    return buffer.at(head);
}

/** Unlock buffer after external write operation (using reserveHead) 
 * finished. All other calls to the buffer will block until freeHead() is 
 * called. 
 * Calling freeHead() wakes up other threads trying to read from an empty 
 * buffer.
 */
template <class Type>
int Ringbuffer<Type>::freeHead()
{
    head = ++head % bufferSize;
    sem_post(&mtx);
    sem_post(&usage);
    return 0;
}

/** Lock tail position of buffer to perform read/copy operation externally.
 * 
 * If there is no data in the buffer it returns NULL. The call blocks if
 * another thread is using the buffer.
 *
 * The buffer will block any other threads until freeTail() is called.
 *
 * \return Pointer to data to be read or NULL if buffer is empty.
 */
template <class Type>
Type* Ringbuffer<Type>::reserveTailTry()
{
    if(sem_trywait(&usage) != 0) {
    	return NULL;
    }
    sem_wait(&mtx);
    return &buffer.at(tail);
}

/**Unlock buffer after external read operation (using reserveTail())
 * finished. All other calls to the buffer will block until freeTail() is
 * called.
 * Calling freeTail() wakes up other blocking threads trying to write to a 
 * full buffer.
 */
template <class Type>
int Ringbuffer<Type>::freeTail()
{
    tail = ++ tail % bufferSize;
    sem_post(&mtx);
    sem_post(&space);
    return 0;
}
/** Get amount of items stored in buffer.
 * \return Number of items in buffer
 */
template <class Type>
int Ringbuffer<Type>::getSize() {
	int full_value;
	sem_getvalue(&usage, &full_value);
	return full_value;
}

/** Tell if buffer is empty.
 * \return True if no elements are in buffer. False otherwise.
 */
template <class Type>
bool Ringbuffer<Type>::isEmpty() {
	int full_value, empty_value;
	sem_getvalue(&usage, &full_value);
	sem_getvalue(&space, &empty_value);
	return (full_value == 0) && (empty_value == bufferSize);
}

/** Tell if buffer is empty and will stay empty. This is the case if
 *  all produces ceased to add data and no data is in the buffer.
 * \return True if there are no elements in buffer and all producers 
 * announced that they stopped adding elements. False otherwise.
 */
template <class Type>
bool Ringbuffer<Type>::isFinished() {
	int full_value;
	sem_getvalue(&usage, &full_value);
	return (producer==0) && (full_value == 0);
}

/** Lets a producer announce that it is adding no more elements to the 
 *  buffer.
 *  To be called only once per producer. This is not checked.
 */
template <class Type>
void Ringbuffer<Type>::producerQuit() {
	__sync_sub_and_fetch(&producer,1);
}
#endif

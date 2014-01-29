#ifndef Ringbuffer_H
#define Ringbuffer_H

#include <vector>
#include <iostream>
#include <semaphore.h>

/*! Ringbuffer
 *  @brief A ringbuffer template supporting non-host consumers/producers.
 *
 *  Data is written to the head of the buffer and read from the tail.
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
    Ringbuffer(unsigned int bSize, int producer);
    ~Ringbuffer();
    int writeFromHost(Type *inputOnHost);
    int copyToHost(Type *outputOnHost);
    Type* reserveHead();
    int freeHead();
    Type* reserveTailTry();
    int freeTail();
    int getSize();
    bool isEmpty();
    bool isFinished();
    void producerQuit();
};

/**
 * Basic Constructor.
 *
 * Reserves buffer memory.
 *
 * \param bSize buffer size in items of 'Type'
 * \param producer Number of producers feeding the buffer. 
 */
template <class Type>
Ringbuffer<Type>::Ringbuffer(unsigned int bSize, int producer) :
	head(0),// We write new data to this position
	tail(0),// We read stored data from this position
	producer(producer)
{
    buffer.reserve(bSize);
    std::cout << "Reserved buffer of size " << bSize << "\n";
    sem_init(&mtx, 0, 1);
    sem_init(&usage, 0, 0);
    sem_init(&space, 0, bSize);
    bufferSize = bSize; 
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
int Ringbuffer<Type>::writeFromHost(Type *inputOnHost)
{
    sem_wait(&space);   // is there space in buffer?
    sem_wait(&mtx);     // lock buffer
    
    buffer[head] = *inputOnHost;
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
int Ringbuffer<Type>::copyToHost(Type *outputOnHost)
{
    sem_wait(&usage);   // is there some data in buffer?
    sem_wait(&mtx);     // lock buffer
    
    *outputOnHost = buffer[tail];
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
Type* Ringbuffer<Type>::reserveHead()
{
    sem_wait(&space);
    sem_wait(&mtx);
    return &buffer[head];
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

/* Lock tail position of buffer to perform read/copy operation externally.
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
    return &buffer[tail];
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

/** Tell if buffer is empty and will stay empty.
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

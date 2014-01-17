#ifndef RINGBUFFERWF_H
#define RINGBUFFERWF_H

#include <vector>
#include <iostream>
#include <semaphore.h>

template <class Type>
class RingbufferWf {
    /*  Ringbuffer for data of type Type
     *  Data is put to head of buffer and read from tail of buffer.
     *
     *  To enable reading to devices like graphic cards the tail of the 
     *  buffer can be reserved. In the reserved state copy operations can
     *  be performed externally. After copying the head needs to be 
     *  freed.
     */

private:
    sem_t mtx;
    sem_t usage, space;
    std::vector<Type> buffer;
    unsigned int bufferSize;
    unsigned int head, tail;

public:
    RingbufferWf(unsigned int bSize);
    ~RingbufferWf();
    int writeFromHost(Type *inputOnHost);
    int copyToHost(Type *outputOnHost);
    Type* reserveHead();
    int freeHead();
    Type* reserveTail();
    int freeTail();
};

template <class Type>
RingbufferWf<Type>::RingbufferWf(unsigned int bSize)
{
    buffer.reserve(bSize);
    std::cout << "Reserved buffer of size " << bSize << "\n";
    sem_init(&mtx, 0, 1);
    sem_init(&usage, 0, 0);
    sem_init(&space, 0, bSize);
    bufferSize = bSize;
    head = 0;   // We write new data to this position
    tail = 0;   // We read stored data from this position
}

template <class Type>
RingbufferWf<Type>::~RingbufferWf()
{
    sem_destroy(&mtx);
    sem_destroy(&usage);
    sem_destroy(&space);
}

template <class Type>
int RingbufferWf<Type>::writeFromHost(Type *inputOnHost)
// Put data on buffer. InputOnHost needs to be on host memory.
{
    sem_wait(&space);   // is there space in buffer?
    sem_wait(&mtx);     // lock buffer
    
    buffer[head] = *inputOnHost;
    head = ++head % bufferSize;     // move head

    sem_post(&mtx);     // unlock buffer
    sem_post(&usage);   // tell them that there is something in buffer
    
    int buffer_usage;
    sem_getvalue(&usage, &buffer_usage);
    //std::cout << "Usage after write:" << buffer_usage << std::endl;

}

template <class Type>
int RingbufferWf<Type>::copyToHost(Type *outputOnHost)
// Get data from buffer. OutputOnHost needs to be on host memory.
{
    sem_wait(&usage);   // is there some data in buffer?
    sem_wait(&mtx);     // lock buffer
    
    *outputOnHost = buffer[tail];
    tail = ++tail % bufferSize;     // move tail
    
    sem_post(&mtx);     // unlock buffer
    sem_post(&space);   // tell them that there is space in buffer
    
    //int buffer_usage;
    //sem_getvalue(&usage, &buffer_usage);
    //std::cout << "Usage after read:" << buffer_usage << std::endl;
}

template <class Type>
Type* RingbufferWf<Type>::reserveHead()
// Lock head position of buffer to perform write operations externally.
// Buffer is blocked until freeHead is called.
{
    sem_wait(&space);
    sem_wait(&mtx);
    return &buffer[head];
}

template <class Type>
int RingbufferWf<Type>::freeHead()
// Free head position after external operation finished.
{
    head = ++head % bufferSize;
    sem_post(&mtx);
    sem_post(&usage);
    return 0;
}

template <class Type>
Type* RingbufferWf<Type>::reserveTail()
// Lock tail position of buffer to perform read/copy operation 
// externally.
// Buffer is blocked until freeTail is called.
{
    sem_wait(&usage);
    sem_wait(&mtx);
    return &buffer[tail];
}

template <class Type>
int RingbufferWf<Type>::freeTail()
// Free tail position after external operation finished.
{
    tail = ++ tail % bufferSize;
    sem_post(&mtx);
    sem_post(&space);
    return 0;
}

template <class Type>
void fill_wform(Type wform, short int fill_value)
{
    for (int i=0; i<SAMPLE_COUNT; i++) {
        wform[i] = fill_value;
    }
}
#endif

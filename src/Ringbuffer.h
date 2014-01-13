#ifndef RINGBUFFER_H
#define RINGEBUGGER_H

#include <semaphore.h>

template <class type>
class Ringbuffer() {
private:
	sem_t mtx;
	sem_t full, empty;
public:
    Ringbuffer(size_t bSize);
    ~Ringbuffer();
    type* reserveHead(unsigned int count);
    int freeHead(type* data, unsigned int count);
    type* reserveTail(unsigned int count);
    int freeTail(type* gpu_data, unsigned int count);
}	

#endif

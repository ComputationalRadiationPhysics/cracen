#ifndef RINGBUFFER_H
#define RINGEBUGGER_H

#include <semaphore.h>

template <class Type, unsigned int size>
class Ringbuffer {
private:
	sem_t mtx;
	sem_t full, empty;
	Type buffer[size];
	
public:
    Ringbuffer();
    ~Ringbuffer();
    Type* reserveHead(unsigned int count);
    int freeHead(Type* data, unsigned int count);
    Type* reserveTail(unsigned int count);
    int freeTail(Type* gpu_data, unsigned int count);
};

template <class Type, unsigned int size>
Ringbuffer<Type, size>::Ringbuffer() {
	sem_init(&full,0,0);
	sem_init(&empty,0,size);

}
template <class Type, unsigned int size>
Ringbuffer<Type, size>::~Ringbuffer() {
	sem_destroy(&full);
	sem_destroy(&empty);
}

template <class Type, unsigned int size>
Type* Ringbuffer<Type, size>::reserveHead(unsigned int count) {
	sem_wait(&empty);
	int value;
	sem_getvalue(&empty, &value);
	return &buffer[size-value]; //Hier bin ich mir nicht sicher
}

template <class Type, unsigned int size>
int Ringbuffer<Type, size>::freeHead(Type* data, unsigned int count) {
	sem_post(&full);
	return 0;
}

template <class Type, unsigned int size>
Type* Ringbuffer<Type, size>::reserveTail(unsigned int count) {
	sem_wait(&full);
	int value;
	sem_getvalue(&full, &value);
	return &buffer[size-value]; //Hier bin ich mir nicht sicher
}

template <class Type, unsigned int size>
int Ringbuffer<Type, size>::freeTail(Type* gpu_data, unsigned int count) {
	sem_post(&empty);
	return 0;
}

#endif

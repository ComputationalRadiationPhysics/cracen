#ifndef TEST_RINGBUFFERWF_H
#define TEST_RINGBUFFERWF_H

#include <pthread.h>
#include <iostream>
#include <cstdlib>      // for 'exit'
#include <unistd.h>     // for 'sleep'
#include "Types.h"
#include "RingbufferWf.h"

#define BSIZE 1000000

typedef RingbufferWf<wform_t> InputBufferWf;

int test_InputBufferWf_creation()
{
    InputBufferWf *testBuffer = new InputBufferWf(BSIZE);
    delete testBuffer;
    return 0;
}

int add_to_buffer(InputBufferWf *rb)
{
    static const short int ini_arr[1000] = {0};
    wform_t input(ini_arr, ini_arr + sizeof(ini_arr) / sizeof(ini_arr[0]));
    int total_copies = 0;
    std::cout << "\tStart writing to buffer." << std::endl;
    while(total_copies < 100) {
        std::cout << "write to buffer no " << total_copies << "\n";
        rb->writeFromHost(&input);
        total_copies++;
        usleep(100);
    }
    return total_copies;
}

int remove_from_buffer(InputBufferWf *rb)
{
    wform_t output;
    int total_copies = 0;
    while(total_copies < 100) {
        std::cout << "read from buffer no " << total_copies << "\n";
        rb->copyToHost(&output);
        total_copies++;
        usleep(325);
    }
    return total_copies;
}
void* buffer_adder(void* buffer)
{
    std::cout << "\tThread 'buffer_adder' created." << std::endl;
    InputBufferWf* rb = (InputBufferWf*) buffer;
    int total_copies;
    total_copies = add_to_buffer(rb);
    return (void*)NULL;
}

void* buffer_remover(void* buffer)
{
    std::cout << "\tThread 'buffer_remover' created." << "\n";
    InputBufferWf* rb = (InputBufferWf*) buffer;
    int total_copies = remove_from_buffer(rb);
    return (void*)NULL;
}

int test_InputBufferWf_twoThreads()
{
    int letRun_add = 1;
    int letRun_remove = 1;

    int rc;
    pthread_t threads[2];

    InputBufferWf* buf = new InputBufferWf(BSIZE);
    
    rc = pthread_create(&threads[0], NULL, buffer_adder,
                        (void *)buf);
    if (rc) {
        std::cout << "\tError: unable to create thread, " << rc << "\n";
        exit(-1);
    }
    
    rc = pthread_create(&threads[1], NULL, buffer_remover,
                        (void *)buf);
    if (rc) {
        std::cout << "\tError: unable to create thread, " << rc << "\n";
        exit(-1);
    }

    void *status;
    for (int i=0; i<2; i++) {
        rc = pthread_join(threads[i], &status);
        if (rc) {
            std::cout << "\tError: unable to join," << rc << "\n";
            exit(-1);
        }
    }
    delete buf;
    std::cout << "\tTwo threads reading and writing from/to buffer \n"
              << "\tcreated, executed and ended without exception.\n";
}

int tests_RingbufferWf()
{
    std::cout << "Testing InputBufferWf...\n------------\n";

    int fails = 0;
    if (test_InputBufferWf_creation()) {
        fails++;
    }
    if (test_InputBufferWf_twoThreads()) {
        fails++;
    }
    return fails;
}

#endif

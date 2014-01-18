#ifndef DATAREADER_H
#define DATAREADER_H

#include <iostream>
#include <fstream> 
#include <string> 
#include <pthread.h>
#include "Types.h"
#include "Constants.h"
#include "Ringbuffer.h"

typedef Ringbuffer<wform_t> InputBufferWf;

class DataReader {
    /*  DataReader is meant as a producer for the Ringbuffer class. It 
     *  reads data from a file *inputFilename*, puts it into the type 
     *  wform_t and tries to write it to the ringbuffer.
     */

private:
    std::string inputFilename;
    InputBufferWf* rb;
    wform_t wfCh1;
    wform_t wfCh2;
    std::vector<short int> channelBuffer;

    int nSamp;
    int nSeg;
    int nWf;
    pthread_t readthread;

public:
    DataReader(std::string filename, InputBufferWf* buffer);
    ~DataReader();
    int _checkFileHeader();
    void readToBufferAsync();
    int isReading();
    void stopReading();

    int get_nSamp() {return nSamp;};
    int get_nSeg() {return nSeg;};
    int get_nWf() {return nWf;};
};
#endif

#ifndef DATAREADER_H
#define DATAREADER_H

#include <iostream>
#include <fstream> 
#include <string> 
#include <pthread.h>
#include "Types.h"
#include "Constants.h"
#include "Ringbuffer.h"


class DataReader {
    /*  DataReader is meant as a producer for the Ringbuffer class. It 
     *  reads data from a file *inputFilename*, puts it into the type 
     *  wform_t and tries to write it to the ringbuffer.
     */

private:
    std::string inputFilename;
    InputBuffer* rb;
    Chunk temp;
    std::vector<MeasureType> channelBuffer;

    int nSamp;
    int nSeg;
    int nWf;
    int nChunk;

public:
    DataReader(const std::string& filename, InputBuffer* buffer,
               int chunksize);
    ~DataReader();
    static int readHeader(const std::string& filename,
                          int &nSample, int &nSegment, int &nWaveform);
    void readToBuffer();

    int get_nSamp() {return nSamp;};
    int get_nSeg() {return nSeg;};
    int get_nWf() {return nWf;};
};
#endif

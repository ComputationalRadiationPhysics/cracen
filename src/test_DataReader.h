#ifndef TEST_DATAREADER_H
#define TEST_DATAREADER_H

#include <iostream>
#include <string>
#include "Types.h"
#include "Ringbuffer.h"
#include "DataReader.h"
#include "Constants.h"

#define BUFFERSIZE 500000

int test_DataReader_ReadFileHeader()
{
    InputBufferWf rb = InputBufferWf(BUFFERSIZE);
    DataReader dr = DataReader(FILENAME_TESTFILE, &rb);
    if (dr.get_nSamp() != SAMPLE_COUNT_TESTFILE or 
        dr.get_nSeg() != SEGMENT_COUNT_TESTFILE or 
        dr.get_nWf() != WAVEFORM_COUNT_TESTFILE) {
        std::cout << "unexpected data: " << dr.get_nSamp() << " " 
                  << dr.get_nSeg() << " "  << dr.get_nWf() << "\n";
        return -1;
    }
    return 0;
}

int test_DataReader_ReadStreamToBuffer()
{   
    InputBufferWf rb = InputBufferWf(BUFFERSIZE);
    DataReader dr = DataReader(FILENAME_TESTFILE, &rb);
    dr.readToBufferAsync();
    return 0;
}

int tests_DataReader()
{
    std::cout << "Testing DataReader...\n---------\n" << std::endl;
    int fails = 0;

    if (test_DataReader_ReadFileHeader()) {
        std::cout << "\tFailed DataReader_ReadFileHeader" << std::endl;
        fails++;
    }
    if (test_DataReader_ReadStreamToBuffer()) {
        std::cout << "\tFailed DataReader_ReadStream" << std::endl;
        fails++;
    }
    //if (test_DataReader_
    std::cout << "\tFailed " << fails << " of 2 tests." << std::endl;

    return fails;
}
#endif

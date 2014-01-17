#include "DataReader.h"

DataReader::DataReader(std::string filename, InputBufferWf* buffer) :
    inputFilename(filename),
    rb(buffer),
    nSamp(-1), nSeg(-1), nWf(-1)
{
    channelBuffer.reserve(2*SAMPLE_COUNT);
    wfCh1.reserve(SAMPLE_COUNT);
    wfCh2.reserve(SAMPLE_COUNT);
    this->_checkFileHeader();
}

DataReader::~DataReader()
{

}

int DataReader::_checkFileHeader()
{
    int errorcode = 0;
    int fsize = 0;
    std::ifstream fs;
    fs.open(inputFilename.c_str(), std::ifstream::in | std::ifstream::binary);
    if (fs) {
        fs.seekg(0, fs.end);    // move cursor to end of file
        fsize = fs.tellg();
        fs.seekg(0, fs.beg);    // move cursor to beginning of file

        fs.read(reinterpret_cast<char*>(&nSamp), sizeof(int));
        fs.read(reinterpret_cast<char*>(&nSeg),  sizeof(int));
        fs.read(reinterpret_cast<char*>(&nWf),   sizeof(int));
    } else {
        std::cout << "Failed to open file." << std::endl;
        if (fs.fail()) {
            std::cout << "Fail() was true." << std::endl;
        }
        if (fs.eof()) {
            std::cout << "EOF() was true." << std::endl;
        }
    }
    //std::cout << "nSamp = " << nSamp << std::endl;
    //std::cout << "nSegments = " << nSeg << std::endl;
    //std::cout << "nWafeforms = " << nWf << std::endl;
    //std::cout << "Filesize = " << fsize << std::endl;
    
    // Check if NSAMPLE of file matches our NSAMPLE
    if (nSamp != SAMPLE_COUNT_TESTFILE) {
        std::cout << "ERROR: SAMPLE_COUNT_TESTFILE in file (" << nSamp 
                  << ") is not what we expected (" << SAMPLE_COUNT_TESTFILE
                  << ")." << std::endl;
        errorcode = 1;
    }
    fs.close();
    return errorcode; 
}

void DataReader::readToBufferAsync()
{
    std::ifstream fs;
    fs.open(inputFilename.c_str(), std::ifstream::in |
                                   std::ifstream::binary);
    if (fs) {
        // move cursor over the first 3 Integers
        fs.seekg(3*sizeof(int), fs.beg);    
        // read waveform data
        while (not fs.eof()) {
            // reading directly into the vector seems legitimate
            // http://stackoverflow.com/questions/2780365/using-read-directly-into-a-c-stdvector
            fs.read(reinterpret_cast<char*>(&channelBuffer[0]), 
                    sizeof(short int) * 2 * nSamp);
           
            // Now we have sampledata of two Waveforms (channel 1 and 
            // channel 2) mixed like
            //      [s1ch1 s1ch2 s2ch1 s2 ch2 s3ch1 s3ch2 ... ]
            // But we want it like
            //      [s1ch1 s2ch1 s3ch1 ... s1ch2 s2ch2 s3ch2 ...]

            for (int i=0; i<nSamp; i++) {
                wfCh1[i] = channelBuffer[2*i];
                wfCh2[i] = channelBuffer[2*i+1];
            }

            rb->writeFromHost(&wfCh1);
            rb->writeFromHost(&wfCh2);
        }
    } else {
        std::cout << "Failed to open file." << std::endl;
    }

    if (fs.is_open()) {
        fs.close();
    }
}

int DataReader::isReading()
{
    return 0;
}

void DataReader::stopReading()
{
    // stop readThread
}

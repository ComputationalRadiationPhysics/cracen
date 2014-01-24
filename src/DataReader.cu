#include "DataReader.h"

DataReader::DataReader(const std::string& filename, InputBuffer* buffer) :
    inputFilename(filename),
    nSamp(-1), nSeg(-1), nWf(-1),
    rb(buffer)
{
    channelBuffer.reserve(2*SAMPLE_COUNT);
    _checkFileHeader();
}

DataReader::~DataReader()
{

}

int DataReader::_checkFileHeader()
{
    int errorcode = 0;
    std::ifstream fs;
    fs.open(inputFilename.c_str(), std::ifstream::in | std::ifstream::binary);
    if (fs) {
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
    if(nSamp != SAMPLE_COUNT) {
    	std::cerr << "Error reading the File:" << std::endl;
    	std::cerr << "SAMPLE_COUNT does not match SAMPLE_COUNT written to the file" << std::endl;
    	std::cerr << SAMPLE_COUNT << " != " << nSamp << std::endl;
    	std::cerr << "Recompile the programm with SAMPLE_COUNT = " << nSamp << " to fix this problem." << std::endl;
    	std::cerr << "Abort." << std::endl;
	  	errorcode = 1;
    }
    
    if(nSeg != 1) {
    	std::cerr << "Number of Segments has to be one. Sequence Mode is not allowed" << std::endl;
    	std::cerr << "Abort." << std::endl;
     	errorcode = 2;
    }
    
    if(nSamp > MAXCOUNTDATA) {
    	std::cerr << "Resolution of the Waveforms exceed the compute capability of you graphics card." << std::endl;
    	std::cerr << "Maximum number of samples is:" << MAXCOUNTDATA << std::endl;
    	std::cerr << "Abort." << std::endl;
	 	errorcode = 3;
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
        int j = 0;
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
                temp[j][i] = static_cast<Precision>(channelBuffer[2*i]);
                temp[j+1][i] = static_cast<Precision>(channelBuffer[2*i+1]);
            }
			j += 2;
            
            if(j >= CHUNK_COUNT) {
            	//Copy data to ring buffer
    			//TODO : Replace the copy thing with a nice function
            	//rb->writeFromHost(&temp); //This can work because of missing copy constructors
            	SampleChunk *buffer = rb->reserveHead();
		        for(int k = 0; k < CHUNK_COUNT; k++) {
		        	for(int i = 0; i < SAMPLE_COUNT; i++) {
		        		(*buffer)[k][i] = temp[k][i];	
		        	}
		        }
            	rb->freeHead();
            	j = 0;
            }
        }
        
        //Fill the last Chunk with 0
        for(j=j; j < CHUNK_COUNT; j++) {
        	for(int i = 0; i < nSamp; i++) {
        		temp[j][i] = static_cast<Precision>(0);
        	}
        }
		//Copy data to ring buffer
		//TODO : Replace the copy thing with a nice function
    	//rb->writeFromHost(&temp); //This can work because of missing copy constructors
    	SampleChunk *buffer = rb->reserveHead();
        for(int k = 0; k < CHUNK_COUNT; k++) {
        	for(int i = 0; i < SAMPLE_COUNT; i++) {
        		(*buffer)[k][i] = temp[k][i];	
        	}
        }
    	rb->freeHead();
    	
    	j = 0;

    } else {
        std::cout << "Failed to open file." << std::endl;
    }
	rb->producerQuit();
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

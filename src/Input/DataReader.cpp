#include "DataReader.hpp"

DataReader::DataReader(const std::string& filename, InputBuffer* buffer,                       int chunksize) :
    inputFilename(filename),
    nSamp(-1), nSeg(-1), nWf(-1), nChunk(chunksize),
    rb(buffer)
{
    readHeader(filename, nSamp, nSeg, nWf);
    channelBuffer.resize(2 * nSamp);
    temp.resize(nSamp * nChunk);
}

DataReader::~DataReader()
{

}

int DataReader::readHeader(const std::string& filename,
                           int &nSample, int &nSegment, int &nWaveform)
{
    int errorcode = 0;
    std::ifstream fs;
    fs.open(filename.c_str(), 
            std::ifstream::in | std::ifstream::binary);
    if (fs) {
        fs.read(reinterpret_cast<char*>(&nSample),   sizeof(int));
        fs.read(reinterpret_cast<char*>(&nSegment),  sizeof(int));
        fs.read(reinterpret_cast<char*>(&nWaveform), sizeof(int));
    } else {
        std::cout << "Failed to open file." << std::endl;
        if (fs.fail()) {
            std::cout << "Fail() was true." << std::endl;
            errorcode = 1;
        }
        if (fs.eof()) {
            std::cout << "EOF() was true." << std::endl;
            errorcode = 2;
        }

    }
    std::cout << "nSample: " << nSample << "\n";
    std::cout << "nSegment: " << nSegment << "\n";
    std::cout << "nWaveform: " << nWaveform << "\n";
    //Check if NSAMPLE of file matches our NSAMPLE
    
    if(nSample != SAMPLE_COUNT) {
    	std::cerr << "Error reading the File:" << std::endl;
    	std::cerr << "SAMPLE_COUNT does not match SAMPLE_COUNT written to the file" << std::endl;
    	std::cerr << SAMPLE_COUNT << " != " << nSample << std::endl;
    	std::cerr << "Recompile the programm with SAMPLE_COUNT = " << nSample << " to fix this problem." << std::endl;
    	std::cerr << "Abort." << std::endl;
	  	errorcode = 3;
    }
    
    if(nSegment != 1) {
    	std::cerr << "Number of Segments has to be one. Sequence Mode is not allowed" << std::endl;
    	std::cerr << "Abort." << std::endl;
     	errorcode = 4;
    }
    
    fs.close();
    return errorcode; 
}

void DataReader::readToBuffer()
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
            /*
            static int xxii = 0;
            xxii++;
            if(xxii>=128) break;
            */
            // reading directly into the vector seems legitimate
            // http://stackoverflow.com/questions/2780365/using-read-directly-into-a-c-stdvector
            fs.read(reinterpret_cast<char*>(&channelBuffer[0]), 
                    sizeof(MeasureType) * 2 * nSamp);
           
            // Now we have sampledata of two Waveforms (channel 1 and 
            // channel 2) mixed like
            //      [s1ch1 s1ch2 s2ch1 s2 ch2 s3ch1 s3ch2 ... ]
            // But we want it like
            //      [s1ch1 s2ch1 s3ch1 ... s1ch2 s2ch2 s3ch2 ...]

            for (int i=0; i<nSamp; i++) {
                temp[ j   *nSamp + i] = 
                        static_cast<DATATYPE>(channelBuffer[2*i]);
                temp[(j+1)*nSamp + i] = 
                        static_cast<DATATYPE>(channelBuffer[2*i+1]);
            }
			j += 2;
            
            if(j >= nChunk) {
            	//Copy data to ring buffer
    			//TODO : Replace the copy thing with a nice function
            	//rb->writeFromHost(&temp); //This can work because of missing copy constructors
            	Chunk *buffer = rb->reserveHead();
		        for(int k = 0; k < nChunk*nSamp; k++) {
		        	//(*buffer)[k] = temp[k];
		        	buffer->at(k) = temp.at(k);	
                }
            	rb->freeHead();
            	j = 0;
            }
        }
        
        //Fill the last Chunk with 0
        for(j=j; j < nChunk; j++) {
        	for(int i = 0; i < nSamp; i++) {
        		temp[j*nSamp +i] = static_cast<DATATYPE>(0);
        	}
        }
		//Copy data to ring buffer
		//TODO : Replace the copy thing with a nice function
    	//rb->writeFromHost(&temp); //This can work because of missing copy constructors
    	Chunk *buffer = rb->reserveHead();
        for(int k = 0; k < nChunk*nSamp; k++) {
            buffer->at(k) = temp.at(k);	
        }
    	rb->freeHead();
    	
    	j = 0;

    } else {
        std::cout << "Failed to open file." << std::endl;
    }
    if (fs.is_open()) {
        fs.close();
    }
	rb->producerQuit();
}

int DataReader::get_nSamp() {
	return nSamp;
};

int DataReader::get_nSeg() {
	return nSeg;
};

int DataReader::get_nWf() {
	return nWf;
};

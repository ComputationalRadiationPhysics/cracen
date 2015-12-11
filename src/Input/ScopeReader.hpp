#ifndef SCOPEREADER_HPP
#define SCOPEREADER_HPP

/*
 *  Based on word done by 2012  Jakub Cizek, Marian Vlcek 
 * 
 *  J. Cizek. M. Vlcek, I. Prochazka, Digital spectrometer
 *  for coincidence measurement of Doppler broadening of positron
 *  annihilation radiation, Nuclear Instruments and Methods
 *  in Physics Research, Section A 623, 982-994 (2010)
 *
 */
 
 #ifndef ScopeReader_HPP
#define ScopeReader_HPP

#include <iostream>
#include <fstream> 
#include <string> 
#include <pthread.h>
#include <memory>
#include "../Config/Types.hpp"
#include "../Config/Constants.hpp"
#include "AgMD1.h"
#include "AgMD1Fundamental.h"
#include "AgMD1FundamentalErrorCodes.h"
#include "AgMD1FundamentalDataTypes.h"

/*! ScopeReader
 *  @brief Is meant as a producer for the Ringbuffer class. 
 *  It reads data from the digital scope.
 *
 */

class ScopeReader {
public:
	/**
	 * A simple struct that holds all parameters to configure the scope.
	 */
	struct ScopeParameter {
		ViInt32 nbrSamples;
		ViInt32 nbrSegments;
		ViInt32 nbrSessions;
		ViInt32 nbrWaveforms;
		ViReal64 sampInterval;
		ViReal64 delayTime; 
		ViInt32 coupling0;
		ViInt32 coupling1; 
		ViInt32 bandwidth0;
		ViInt32 bandwidth1; 
		ViReal64 fullScale0;
		ViReal64 fullScale1;
		ViReal64 offset0;
		ViReal64 offset1;
		ViInt32 trigType;
		ViInt32 trigCoupling_int;
		ViInt32 trigCoupling_ext;
		ViInt32 trigSlope;
		ViReal64 trigLevel;
		ViInt32 timeout;
		ViInt32 i_Simulation;
		
		ScopeParameter();
		ScopeParameter(const std::string& filename);	
	};

private:
    InputBuffer* rb;
    std::unique_ptr<Chunk> temp;
    std::vector<MeasureType> channelBuffer;
	ScopeParameter param;
    int nSamp;
    int nSeg;
    int nWf;
    int nChunk;
	ViSession InstrumentID;
	ViStatus status;
	
	//TODO: Convert raw pointer to objects
	AqReadParameters *readPar;
	AqDataDescriptor *dataDesc;
	AqSegmentDescriptor *segDesc;


	void initilizeDevice();
	void calibrateDevice();
	void getInstrumentInfo();
	void configureDigitizer();
	void setDeviceTrigger();
	void readDeviceConfig();
	
public:	
    /**
     * Basic constructor
     *
     * Creates the ScopeReader for the given filename and connects it
     * to the Ringbuffer buffer. Multiple signals are read and written
     * to the buffer in one chunk.
     *
     * \param filename The file to be read. It needs to follow the
     *                 datastructure as produced by DCDB_Gen.
     * \param buffer The ringbuffer to be filled with the data.
     * \param chunksize Sets the number of signals in one chunk.
     */
    ScopeReader(const ScopeParameter& scopeParameter, InputBuffer* buffer,
               int chunksize);
    ~ScopeReader();

    /** 
     * Start acquire data with the scope and write it to the buffer.
	 *
     */
    void readToBuffer();

    /** Return number of samples per signal as given by the header of
     *  the file.
     *  \return Number of samples per signal
     */
    int get_nSamp() {return nSamp;};

    /** Return number of signals per segment as given by the header of
     *  the file. At the moment only nSeg==1 is supported.
     *  \return Number of signals per segment.
     */
    int get_nSeg() {return nSeg;};

    /** Return number of signals (waveforms) in the datafile as given
     *  by the header of the file.
     *  \return Number of signals in the datafile.
     */
    int get_nWf() {return nWf;};
};
#endif
 
 #endif

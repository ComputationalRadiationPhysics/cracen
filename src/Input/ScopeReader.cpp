/*
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *  Big parts of this code is based on work by Jakub Cizek, Marian Vlcek.
 *
 *  J. Cizek. M. Vlcek, I. Prochazka, Digital spectrometer
 *  for coincidence measurement of Doppler broadening of positron
 *  annihilation radiation, Nuclear Instruments and Methods
 *  in Physics Research, Section A 623, 982-994 (2010)
 */


#include "ScopeReader.hpp"

void check_stat(ViSession InstrumentID,ViStatus status, const char* info)
{
	ViChar message[512];
	Acqrs_errorMessage(InstrumentID,status,message,512);
	if(status!=VI_SUCCESS) std::cerr << info << " " << message << std::endl;
}

ScopeReader::ScopeReader(const ScopeReader::ScopeParameter& param, InputBuffer* buffer, int chunksize) :
    param(param),
    nSamp(-1), nSeg(-1), nWf(-1), nChunk(chunksize),
    rb(buffer)
{
	initilizeDevice();
	calibrateDevice();
	getInstrumentInfo();
	configureDigitizer();
	setDeviceTrigger();
	readDeviceConfig();
}


ScopeReader::~ScopeReader()
{
	delete readPar;
	delete dataDesc;
	delete segDesc;
}

void ScopeReader::initilizeDevice() {
	std::string dev = "PCI::INSTR0";
	std::string simulate = "";
	if(param.i_Simulation==1) {
		dev = "PCI::DC440";
		simulate = "simulate=TRUE";
		status = Acqrs_setSimulationOptions("64K");
	} 
	
	/* Copy string values to a char array to get a non const c-string. */
	std::vector<char> dev_buff(dev.length() + 1, '\0');
	std::copy(dev.begin(), dev.end(), dev_buff.begin());
	char* dev_cstr = &dev_buff[0];
	std::vector<char> simulate_buff(simulate.length() + 1, '\0');
	std::copy(simulate.begin(), simulate.end(), simulate_buff.begin());
	char* simulate_cstr = &simulate_buff[0];
	
	status = Acqrs_InitWithOptions(dev_cstr,VI_FALSE,VI_FALSE,simulate_cstr,&InstrumentID);
	check_stat(InstrumentID,status,"InitWithOptions");
	
	ViChar InstrumentName[32];
	ViInt32 SerialNbr,BusNbr,SlotNbr;
	status = Acqrs_getInstrumentData(InstrumentID,InstrumentName,&SerialNbr,&BusNbr,&SlotNbr);
	check_stat(InstrumentID,status,"getInstrumentData");
	if(status==VI_SUCCESS) std::cout << InstrumentName << " serial no: " << SerialNbr << ", bus no: " << BusNbr << ", slot no: " << SlotNbr << std::endl;
	
	ViInt32 nbrChannels;	
	status = Acqrs_getNbrChannels(InstrumentID,&nbrChannels);
	check_stat(InstrumentID, status, "getNbrChannels");
	if(status==VI_SUCCESS) std::cout << "Number of Channels: " << nbrChannels << std::endl;
}

void ScopeReader::calibrateDevice() {
	status = Acqrs_calibrate(InstrumentID);
	check_stat(InstrumentID,status,"calibrate");
	if(status==VI_SUCCESS) std::cout << "calibration O.K." << std::endl;
}

void ScopeReader::getInstrumentInfo() {
	ViInt32 nbrBits;
	status = Acqrs_getInstrumentInfo(InstrumentID,"NbrADCBits",&nbrBits);
	check_stat(InstrumentID,status,"getInstrumentInfo>NbrADCBits");
	if(status==VI_SUCCESS) std::cout << "Number of bits per sample: " << nbrBits << std::endl;
	
	ViInt32 nbrExtTrigs;
	status = Acqrs_getInstrumentInfo(InstrumentID,"NbrExternalTriggers",&nbrExtTrigs);
	check_stat(InstrumentID,status,"getInstrumentInfo>NbrExternalTriggers");
	if(status==VI_SUCCESS) std::cout << "Number of external triggers: " << nbrExtTrigs << std::endl;
	
	ViInt32 nbrIntTrigs;
	status = Acqrs_getInstrumentInfo(InstrumentID,"NbrInternalTriggers",&nbrIntTrigs);
	check_stat(InstrumentID,status,"getInstrumentInfo>NbrInternalTriggers");
	if(status==VI_SUCCESS) std::cout << "Number of internal triggers: " << nbrIntTrigs << std::endl;
	
	std::cout << "Trigger level range:" << std::endl;
	ViReal64 trigRange0;
	status = Acqrs_getInstrumentInfo(InstrumentID,"TrigLevelRange 1",&trigRange0);
	check_stat(InstrumentID,status,"getInstrumentInfo>TrigLevelRange, channel 1");
	if(status==VI_SUCCESS) std::cout << "channel 1: " << trigRange0 << std::endl;

	ViReal64 trigRange1;
	status = Acqrs_getInstrumentInfo(InstrumentID,"TrigLevelRange 2",&trigRange1);
	check_stat(InstrumentID,status,"getInstrumentInfo>TrigLevelRange, channel 2");	
	if(status==VI_SUCCESS) std::cout << "channel 2: " << trigRange1 << std::endl;
	
	ViReal64 temperature;
	status = Acqrs_getInstrumentInfo(InstrumentID,"Temperature 2",&temperature);
	check_stat(InstrumentID,status,"getInstrumentInfo>Temperature");
	if(status==VI_SUCCESS) std::cout << "Temperature: " << temperature << " C" << std::endl;
}

void ScopeReader::configureDigitizer() {
	std::cout << "Configuring digitizer..." << std::endl;
	// Configure timebase
	status=AcqrsD1_configHorizontal(InstrumentID,param.sampInterval*1.0e-9,param.delayTime*1.0e-9);
	check_stat(InstrumentID,status,"ConfigHorizontal");
	status=AcqrsD1_configMemory(InstrumentID,param.nbrSamples,param.nbrSegments);
	check_stat(InstrumentID,status,"ConfigMemory");
	// Configure vertical settings of channel 1
	status=AcqrsD1_configVertical(InstrumentID,1,param.fullScale0,param.offset0,param.coupling0,param.bandwidth0);
	check_stat(InstrumentID,status,"ConfigVertical channel 1");
	// Configure vertical settings of channel 2
	status=AcqrsD1_configVertical(InstrumentID,2,param.fullScale1,param.offset1,param.coupling1,param.bandwidth1);
	check_stat(InstrumentID,status,"ConfigVertical channel 2");
}
void ScopeReader::setDeviceTrigger() {

	switch(param.trigType) {
		case 1:
			status=AcqrsD1_configTrigClass(InstrumentID,0,0x00000001,0,0,0.0,0.0); //internal trigger ch 1
			check_stat(InstrumentID,status,"ConfigTrigClass");
			status=AcqrsD1_configTrigSource(InstrumentID,1,param.trigCoupling_int,param.trigSlope,param.trigLevel,0.0);
			check_stat(InstrumentID,status,"ConfigTrigSource");
			break;
		case 2:
			status=AcqrsD1_configTrigClass(InstrumentID,0,0x00000002,0,0,0.0,0.0); //internal trigger ch 2
			check_stat(InstrumentID,status,"ConfigTrigClass");
			status=AcqrsD1_configTrigSource(InstrumentID,2,param.trigCoupling_int,param.trigSlope,param.trigLevel,0.0);
			check_stat(InstrumentID,status,"ConfigTrigSource");
			break;
		default:
			status=AcqrsD1_configTrigClass(InstrumentID,0,0x80000000,0,0,0.0,0.0); //external trigger 1
			check_stat(InstrumentID,status,"ConfigTrigClass");
			status=AcqrsD1_configTrigSource(InstrumentID,-1,param.trigCoupling_ext,param.trigSlope,param.trigLevel,0.0);
			check_stat(InstrumentID,status,"ConfigTrigSource");
	}


	std::cout << std::endl;
	std::cout << "CHECK: Reading from digitizer" << std::endl;
	ViReal64 r_delay_time,r_sample_interval;
	ViInt32 r_ipom1,r_ipom2;
	ViReal64 r_pom1,r_pom2;
	status=AcqrsD1_getHorizontal(InstrumentID,&r_sample_interval,&r_delay_time);
	check_stat(InstrumentID,status,"getTrigSource");
	std::cout << "Sample interval(ns): " << r_sample_interval*1e9 << std::endl;
	std::cout << "Delay time (ns): " << r_delay_time*1e9 << std::endl;

	ViReal64 r_full_scale0, r_offset0;
	ViInt32 r_coupling0;
	status=AcqrsD1_getVertical(InstrumentID,1,&r_full_scale0,&r_offset0,&r_coupling0,&r_ipom1);
	check_stat(InstrumentID,status,"getTrigVertical, channel 1");
	std::cout << "Channel 1:" << std::endl;
	std::cout << " full scale(V): " << r_full_scale0 << "    offset(V): " << r_offset0 << "    coupling: " << r_coupling0 << std::endl; 

	ViReal64 r_full_scale1, r_offset1;
	ViInt32 r_coupling1;
	status=AcqrsD1_getVertical(InstrumentID,2,&r_full_scale1,&r_offset1,&r_coupling1,&r_ipom1);
	check_stat(InstrumentID,status,"getTrigVertical, channel 2");
	std::cout << "Channel 2:" << std::endl;
	std::cout << " full scale(V): " << r_full_scale1 << "    offset(V): " << r_offset1 << "    coupling: " << r_coupling1 << std::endl; 


	ViInt32 nbrModules;
	ViInt32 nbrIntTrigs, nbrExtTrigs;
	status=Acqrs_getInstrumentInfo(InstrumentID,"NbrInternalTriggers",&nbrIntTrigs);
	check_stat(InstrumentID,status,"getInstrumentInfo>NbrInternalTriggers");
	status=Acqrs_getInstrumentInfo(InstrumentID,"NbrExternalTriggers",&nbrExtTrigs); 
	check_stat(InstrumentID,status,"getInstrumentInfo>NbrExternalTriggers");
	status=Acqrs_getInstrumentInfo(InstrumentID,"NbrModulesInInstrumentTriggers",&nbrModules);
	check_stat(InstrumentID,status,"getInstrumentInfo>NbrModulesInInstrumentTriggers");
	std::cout << "Number of trigers:" << std::endl;
	std::cout << "    internal: " << nbrIntTrigs << std::endl;
	std::cout << "    external: " << nbrExtTrigs << std::endl;
	std::cout << "Number of modules: " << nbrModules << std::endl;

	ViInt32 r_trig_class,r_source_pattern;
	ViInt32 r_trig_coupling, r_trig_slope;
	ViReal64 r_trig_level;
	status=AcqrsD1_getTrigClass(InstrumentID,&r_trig_class,&r_source_pattern,&r_ipom1,&r_ipom2,&r_pom1,&r_pom2);
	check_stat(InstrumentID,status,"getTrigClass");
	
	switch(param.trigType) {
		case 1:
			status=AcqrsD1_getTrigSource(InstrumentID,1,&r_trig_coupling,&r_trig_slope,&r_trig_level,&r_pom2);
			break;
		case 2:
			status=AcqrsD1_getTrigSource(InstrumentID,2,&r_trig_coupling,&r_trig_slope,&r_trig_level,&r_pom2);
			break;
		default:
			status=AcqrsD1_getTrigSource(InstrumentID,-1,&r_trig_coupling,&r_trig_slope,&r_trig_level,&r_pom2);
	}

	check_stat(InstrumentID,status,"getTrigSource");
	std::cout << "Trigger class: " << r_trig_class << std::endl;
	std::cout << "Source pattern: "  << r_source_pattern << std::endl;
	std::cout << "Trigger coupling: " << r_trig_coupling;
	if(r_trig_coupling==0) std::cout << " (DC) " << std::endl;
	if(r_trig_coupling==1) std::cout << " (AC) " << std::endl;
	if(r_trig_coupling==3) std::cout << " (DC 50 Ohm - external) " << std::endl;
	if(r_trig_coupling==4) std::cout << " (AC 50 Ohm - external) " << std::endl;
	std::cout << "Trigger slope: " << r_trig_slope;
	if(r_trig_slope==0) std::cout << " (positive)" << std::endl;
	if(r_trig_slope==1) std::cout << " (negative)" << std::endl;
	std::cout << "Trigger level: " << r_trig_level << std::endl;
}

void ScopeReader::readDeviceConfig() {
	readPar=new AqReadParameters;
	dataDesc=new AqDataDescriptor;
	segDesc=new AqSegmentDescriptor[param.nbrSegments];
	
	readPar->dataType=1;
	readPar->readMode=1;
	readPar->nbrSegments=param.nbrSegments;
	readPar->firstSampleInSeg=0;
	readPar->segmentOffset=param.nbrSamples;
	readPar->firstSegment=0;
	readPar->nbrSamplesInSeg=param.nbrSamples;
	readPar->flags=0;
	readPar->reserved=0;
	readPar->reserved2=0.0;
	readPar->reserved3=0.0;     

	ViInt32 currentSegmentPad;
	ViInt32 nbrSegmentsNom, nbrSamplesNom;
	status=Acqrs_getInstrumentInfo(InstrumentID,"TbSegmentPad",&currentSegmentPad);
	check_stat(InstrumentID,status,"getInstrumentInfo>TbSegmentPad");
	status=AcqrsD1_getMemory(InstrumentID,&nbrSamplesNom,&nbrSegmentsNom);
	check_stat(InstrumentID,status,"getMemory");
	std::cout << "nbrSamplesNom: "<< nbrSamplesNom << ", nbrSegmentsNom: " << nbrSegmentsNom << std::endl;
	
	readPar->dataArraySize=8*sizeof(short)*(nbrSamplesNom+currentSegmentPad)*(1+param.nbrSegments);
	readPar->segDescArraySize=sizeof(AqSegmentDescriptor)*param.nbrSegments;
}

void ScopeReader::readToBuffer()
{
	//measurement loop
	for(int i_Session=1;i_Session<=param.nbrSessions;i_Session++) {
		std::cout << "session " << i_Session << std::endl;
		//time(&rawtime);
		//timeinfo=localtime(&rawtime);

		//const char* asct = asctime(timeinfo);
		int data_good = 0;
		for(int i_Waveform=1;i_Waveform<=param.nbrWaveforms;i_Waveform++) {
			std::cout << "Waveform" << i_Waveform << std::endl;
			//acquisition
			data_good=0;
			do {
				AcqrsD1_acquire(InstrumentID);
				status=AcqrsD1_waitForEndOfAcquisition(InstrumentID,param.timeout);
				if(status==VI_SUCCESS) data_good=1;
				check_stat(InstrumentID,status,"waitForEndOfAcquisition");
				status=AcqrsD1_stopAcquisition(InstrumentID);
				check_stat(InstrumentID,status,"stopAcquisition");
			} while(data_good!=1);

			//readout
	    	Chunk* buffer = rb->reserveHead();
			status=AcqrsD1_readData(InstrumentID,1,readPar,&((*buffer)[0]),dataDesc,segDesc);
			rb->freeHead();
			check_stat(InstrumentID,status,"readData: channel 1");
			buffer = rb->reserveHead();
			status=AcqrsD1_readData(InstrumentID,2,readPar,&((*buffer)[0]),dataDesc,segDesc);
			rb->freeHead();
			check_stat(InstrumentID,status,"readData: channel 2");
			std::cout << "session " << i_Session << ": " << 100.0 * static_cast<double>(i_Waveform) / static_cast<double>(param.nbrWaveforms) << " %% done" << std::endl; 

		

		}

		//stop_session=clock();
		//t_session=stop_session-start_session;
		//std::cout << "session " << i_Session << ":  " << 100.0*(double)i_Waveform/(double)nbrWaveforms << "%% done  " << std::endl;// << nbrWaveforms/t_session*1000 << " 1/s" << std::endl;

	}

	std::cout << "returned Samples in Segment: " << dataDesc->returnedSamplesPerSeg << ", Index First Point " << dataDesc->indexFirstPoint << std::endl;


	rb->producerQuit();
}

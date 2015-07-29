/*
 *  Copyright (C) 2012  Jakub Cizek, Marian Vlcek 
 *
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
 *  If you publish any work which uses this program and/or data
 *  produced by this program, please cite following paper: 
 * 
 *  J. Cizek. M. Vlcek, I. Prochazka, Digital spectrometer
 *  for coincidence measurement of Doppler broadening of positron
 *  annihilation radiation, Nuclear Instruments and Methods
 *  in Physics Research, Section A 623, 982-994 (2010)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "AcqirisImport.h"
#include "AcqirisD1Import.h"

FILE *inifile; //ini file


/*zjisti jestli soubor existuje*/
/*kdyz ano 1, kdyz ne 0*/
int fexist(char fname[])
{
	FILE *f;
	if((f=fopen(fname,"r"))==NULL){
				printf("error: file %s does not exist.\n", fname);
				return(0);
			}
	fclose(f);
	return(1);
}


void Inifile_Comments(void)
{
	char c;
	do{
	 c=getc(inifile);
	 printf("%c",c);
	}while(c!='=');
} 

void check_stat(ViSession InstrumentID,ViStatus status, ViChar info[])
{
	ViChar message[512];

	Acqrs_errorMessage(InstrumentID,status,message,512);
	if(status!=VI_SUCCESS) printf("%s: %s\n",info,message);
}

void main()
{
	 

	 FILE *fout;  /*output file with measured waveforms*/
	 FILE *fptc;
	 char target_path[80];
	 char file_name[80];
	 char fptc_name[80];
	 char suffix[5];

	 ViInt32 i,j;

	 ViSession InstrumentID;
	 ViStatus status;
	 

	 ViInt32 nbrWaveforms;
	 ViInt32 nbrSamples;
	 ViInt32 nbrSegments;
	 ViInt32 nbrSessions;
	 ViInt32 nbrInstruments;
	 ViInt32 nbrChannels;
	 //ViInt32 segmentPad;
	 ViInt32 nbrBits;
	 //ViInt32 dataArraySize;
	 
	 ViReal64 sampInterval;
	 ViReal64 delayTime; 
	 ViReal64 fullScale0,fullScale1;
	 ViReal64 offset0,offset1;
	 ViInt32 coupling0,coupling1; 
	 ViInt32 bandwidth0,bandwidth1; 
	 
	 ViInt32 trigType;
	 ViInt32 trigCoupling_int;
	 ViInt32 trigCoupling_ext;
	 ViInt32 trigSlope;
	 ViReal64 trigLevel;


	 
	 ViChar InstrumentName[32];
	 ViInt32 SerialNbr,BusNbr,SlotNbr;
	 ViInt32 nbrExtTrigs,nbrIntTrigs;
	 ViReal64 trigRange0,trigRange1;
	 ViInt32 temperature;

	 ViInt32 timeOut;
	 ViInt32 i_Simulation;

	 short *dataArrayP1;
	 short *dataArrayP2;
	 ViInt32 currentSegmentPad;
	 ViInt32 nbrSamplesNom; 
	 ViInt32 nbrSegmentsNom; 
	 
	 long i_Waveform;  //counter for Waveforms
	 long i_Session; //counter for Sessions
	 int write_mode;
	 short data_good; 

	 //time
	 time_t rawtime;
	 struct tm *timeinfo;
	 clock_t start_session,stop_session;
	 double t_session;
	 
	 printf("Dacqn - acquisition program for digital measurement of Doppler broadening\n");
	 printf("version 1.00 8.8.2012\n");
	 printf("copyright Jakub Cizek\n");
	 printf("Faculty of Mathematics and Physics, Charles University in Prague\n");
	 time(&rawtime);
	 timeinfo=localtime(&rawtime);
	 printf("%s", asctime(timeinfo));
	 printf("\n");


		//reading ini-file
		printf("Reading inifile dacqn.ini...\n");
		if (fexist("dacqn.ini")==1) inifile=fopen("dacqn.ini", "r");
		else exit(0);
	
		Inifile_Comments();
		fscanf(inifile,"%ld", &nbrSegments);
		printf("%ld",nbrSegments);

		Inifile_Comments();
		fscanf(inifile,"%ld", &nbrWaveforms);
		printf("%ld",nbrWaveforms);

		Inifile_Comments();
		fscanf(inifile,"%ld", &nbrSamples);
		printf("%ld",nbrSamples);

		Inifile_Comments();
		fscanf(inifile,"%ld", &nbrSessions);
		printf("%ld",nbrSessions);

		Inifile_Comments();
		fscanf(inifile,"%lf", &sampInterval);
		printf("%.2lf",sampInterval);

		Inifile_Comments();
		fscanf(inifile,"%lf", &delayTime);
		printf("%.2lf",delayTime);

		Inifile_Comments();
		fscanf(inifile,"%ld,%ld", &coupling0,&coupling1);
		printf("%ld,%ld",coupling0,coupling1);

		Inifile_Comments();
		fscanf(inifile,"%ld,%ld", &bandwidth0,&bandwidth1);
		printf("%ld,%ld",bandwidth0,bandwidth1);

		Inifile_Comments();
		fscanf(inifile,"%lf,%lf", &fullScale0,&fullScale1);
		printf("%.2lf,%.2lf",fullScale0,fullScale1);
		fullScale0=fullScale0/1e3; /*from mV to V*/
		fullScale1=fullScale1/1e3; /*from mV to V*/

		Inifile_Comments();
		fscanf(inifile,"%lf,%lf", &offset0,&offset1);
		printf("%.2lf,%.2lf",offset0,offset1);
		offset0=offset0/1e3; /*from mV to V*/
		offset1=offset1/1e3; /*from mV to V*/

		Inifile_Comments();
		fscanf(inifile,"%d", &trigType);
		printf("%d",trigType);

		Inifile_Comments();
		fscanf(inifile,"%ld,%ld", &trigCoupling_int,&trigCoupling_ext);
		printf("%ld,%ld",trigCoupling_int,trigCoupling_ext);

		Inifile_Comments();
		fscanf(inifile,"%ld", &trigSlope);
		printf("%ld",trigSlope);

		Inifile_Comments();
		fscanf(inifile,"%lf", &trigLevel);
		if(trigType==1)
		{
			printf("%.2lf\n",trigLevel);
			trigLevel=100*(trigLevel/1e3+offset0)/fullScale0;
			printf("  converted to %.2lf %% of fullscale:",trigLevel);
		}
		if(trigType==2)
		{
			printf("%.2lf",trigLevel);
			trigLevel=100*(trigLevel/1e3+offset1)/fullScale1;
			printf("  converted to %.2lf %% of fullscale:",trigLevel);
		} 
		if(trigType==-1) printf("%.2lf mV",trigLevel);

		Inifile_Comments();
		fscanf(inifile,"%d",&timeOut);
		printf("%d",timeOut);

		Inifile_Comments();
		fscanf(inifile,"%s",suffix);
		printf("%s",suffix);

		Inifile_Comments();
		fscanf(inifile,"%d",&write_mode);
		printf("%d",write_mode);

		Inifile_Comments();
		fscanf(inifile,"%s",target_path);
		printf("%s",target_path);

		Inifile_Comments();
		fscanf(inifile,"%d",&i_Simulation);
		printf("%d\n",i_Simulation);

		fclose(inifile); 

	 printf("\n");
	 printf("Scanning for devices...\n");
	 status=Acqrs_getNbrInstruments(&nbrInstruments);
	 printf("Number of instruments:  %ld\n",nbrInstruments);

	 //initialization
	 
	 if(i_Simulation==1) 
		 {
			 status=Acqrs_setSimulationOptions("64K");
			 status=Acqrs_InitWithOptions("PCI::DC440",VI_FALSE,VI_FALSE,"simulate=TRUE",&InstrumentID);
	 }
	 else 
	 {
		 status=Acqrs_InitWithOptions("PCI::INSTR0",VI_FALSE,VI_FALSE,"",&InstrumentID);
	 }
	 check_stat(InstrumentID,status,"InitWithOptions");

	 status=Acqrs_getInstrumentData(InstrumentID,InstrumentName,&SerialNbr,&BusNbr,&SlotNbr);
	 check_stat(InstrumentID,status,"getInstrumentData");
	 if(status==VI_SUCCESS) printf("%s, serial no: %ld, bus no: %ld, slot no: %ld \n",InstrumentName,SerialNbr,BusNbr,SlotNbr);
	 status=Acqrs_getNbrChannels(InstrumentID,&nbrChannels);
	 check_stat(InstrumentID,status,"getNbrChannels");
	 if(status==VI_SUCCESS) printf("Number of Channels:     %ld\n",nbrChannels);
	 
	 //calibration
	 status=Acqrs_calibrate(InstrumentID);
	 check_stat(InstrumentID,status,"calibrate");
	 if(status==VI_SUCCESS) printf("calibration O.K.\n");
	 
	 //getInstrumentInfo
	 status=Acqrs_getInstrumentInfo(InstrumentID,"NbrADCBits",&nbrBits);
	 check_stat(InstrumentID,status,"getInstrumentInfo>NbrADCBits");
	 if(status==VI_SUCCESS) printf("Number of bits per sample: %ld\n",nbrBits);
	 status=Acqrs_getInstrumentInfo(InstrumentID,"NbrExternalTriggers",&nbrExtTrigs);
	 check_stat(InstrumentID,status,"getInstrumentInfo>NbrExternalTriggers");
	 if(status==VI_SUCCESS) printf("Number of external triggers: %ld\n",nbrExtTrigs);
	 status=Acqrs_getInstrumentInfo(InstrumentID,"NbrInternalTriggers",&nbrIntTrigs);
	 check_stat(InstrumentID,status,"getInstrumentInfo>NbrInternalTriggers");
	 if(status==VI_SUCCESS) printf("Number of internal triggers: %ld\n",nbrIntTrigs);
	 printf("Trigger level range:\n");
	 status=Acqrs_getInstrumentInfo(InstrumentID,"TrigLevelRange 1",&trigRange0);
	 check_stat(InstrumentID,status,"getInstrumentInfo>TrigLevelRange, channel 1");
	 if(status==VI_SUCCESS) printf("channel 1: %lf\n",trigRange0);
	 status=Acqrs_getInstrumentInfo(InstrumentID,"TrigLevelRange 2",&trigRange1);
	 check_stat(InstrumentID,status,"getInstrumentInfo>TrigLevelRange, channel 2");
	 if(status==VI_SUCCESS) printf("channel 2: %lf\n",trigRange1);
	 status=Acqrs_getInstrumentInfo(InstrumentID,"Temperature 2",&temperature);
	 check_stat(InstrumentID,status,"getInstrumentInfo>Temperature");
	 if(status==VI_SUCCESS) printf("Temperature: %ld C\n",temperature);


	printf("Configuring digitizer...\n");
	// Configure timebase
	status=AcqrsD1_configHorizontal(InstrumentID,sampInterval*1.0e-9,delayTime*1.0e-9);
	check_stat(InstrumentID,status,"ConfigHorizontal");
	status=AcqrsD1_configMemory(InstrumentID,nbrSamples,nbrSegments);
	check_stat(InstrumentID,status,"ConfigMemory");
	// Configure vertical settings of channel 1
	status=AcqrsD1_configVertical(InstrumentID,1,fullScale0,offset0,coupling0,bandwidth0);
	check_stat(InstrumentID,status,"ConfigVertical channel 1");
	// Configure vertical settings of channel 2
	status=AcqrsD1_configVertical(InstrumentID,2,fullScale1,offset1,coupling1,bandwidth1);
	check_stat(InstrumentID,status,"ConfigVertical channel 2");

	// Configure trigger
	if(trigType==1) 
	{
	  status=AcqrsD1_configTrigClass(InstrumentID,0,0x00000001,0,0,0.0,0.0); //internal trigger ch 1
	  check_stat(InstrumentID,status,"ConfigTrigClass");
	  status=AcqrsD1_configTrigSource(InstrumentID,1,trigCoupling_int,trigSlope,trigLevel,0.0);
	  check_stat(InstrumentID,status,"ConfigTrigSource");
	 }
	if(trigType==2) 
	{
	  status=AcqrsD1_configTrigClass(InstrumentID,0,0x00000002,0,0,0.0,0.0); //internal trigger ch 2
	  check_stat(InstrumentID,status,"ConfigTrigClass");
	  status=AcqrsD1_configTrigSource(InstrumentID,2,trigCoupling_int,trigSlope,trigLevel,0.0);
	  check_stat(InstrumentID,status,"ConfigTrigSource");
	 }
	if(trigType==-1) 
	{
	  status=AcqrsD1_configTrigClass(InstrumentID,0,0x80000000,0,0,0.0,0.0); //external trigger 1
	  check_stat(InstrumentID,status,"ConfigTrigClass");
	  status=AcqrsD1_configTrigSource(InstrumentID,-1,trigCoupling_ext,trigSlope,trigLevel,0.0);
	  check_stat(InstrumentID,status,"ConfigTrigSource");
	}


	printf("\n");
	printf("CHECK: Reading from digitizer\n");
	ViReal64 r_delay_time,r_sample_interval;
	ViInt32 r_ipom1,r_ipom2;
	ViReal64 r_pom1,r_pom2;
	status=AcqrsD1_getHorizontal(InstrumentID,&r_sample_interval,&r_delay_time);
	check_stat(InstrumentID,status,"getTrigSource");
	printf("Sample interval(ns): %f\n",r_sample_interval*1e9);
	printf("Delay time (ns): %f\n",r_delay_time*1e9);

	ViReal64 r_full_scale0, r_offset0;
	ViInt32 r_coupling0;
	status=AcqrsD1_getVertical(InstrumentID,1,&r_full_scale0,&r_offset0,&r_coupling0,&r_ipom1);
	check_stat(InstrumentID,status,"getTrigVertical, channel 1");
	printf("Channel 1:\n");
	printf(" full scale(V): %f   ",r_full_scale0);
	printf("offset(V): %f   ",r_offset0);
	printf("coupling: %ld\n",r_coupling0); 

	ViReal64 r_full_scale1, r_offset1;
	ViInt32 r_coupling1;
	status=AcqrsD1_getVertical(InstrumentID,2,&r_full_scale1,&r_offset1,&r_coupling1,&r_ipom1);
	check_stat(InstrumentID,status,"getTrigVertical, channel 2");
	printf("Channel 2:\n");
	printf(" full scale(V): %f   ",r_full_scale1);
	printf("offset(V): %f   ",r_offset1);
	printf("coupling: %ld\n",r_coupling1); 
		

	//  trigers
	ViInt32 nbrModules;
	status=Acqrs_getInstrumentInfo(InstrumentID,"NbrInternalTriggers",&nbrIntTrigs);
	check_stat(InstrumentID,status,"getInstrumentInfo>NbrInternalTriggers");
	status=Acqrs_getInstrumentInfo(InstrumentID,"NbrExternalTriggers",&nbrExtTrigs); 
	check_stat(InstrumentID,status,"getInstrumentInfo>NbrExternalTriggers");
	status=Acqrs_getInstrumentInfo(InstrumentID,"NbrModulesInInstrumentTriggers",&nbrModules);
	check_stat(InstrumentID,status,"getInstrumentInfo>NbrModulesInInstrumentTriggers");
	printf("Number of trigers:\n internal: %ld\n external: %ld\n",nbrIntTrigs,nbrExtTrigs);
	printf("Number of modules: %ld\n",nbrModules);
	
	ViInt32 r_trig_class,r_source_pattern;
	ViInt32 r_trig_coupling, r_trig_slope;
	ViReal64 r_trig_level;
	status=AcqrsD1_getTrigClass(InstrumentID,&r_trig_class,&r_source_pattern,&r_ipom1,&r_ipom2,&r_pom1,&r_pom2);
	check_stat(InstrumentID,status,"getTrigClass");
	if(trigType==1) status=AcqrsD1_getTrigSource(InstrumentID,1,&r_trig_coupling,&r_trig_slope,&r_trig_level,&r_pom2);
	if(trigType==2) status=AcqrsD1_getTrigSource(InstrumentID,2,&r_trig_coupling,&r_trig_slope,&r_trig_level,&r_pom2);
	if(trigType==-1) status=AcqrsD1_getTrigSource(InstrumentID,-1,&r_trig_coupling,&r_trig_slope,&r_trig_level,&r_pom2);

	check_stat(InstrumentID,status,"getTrigSource");
	printf("Trigger class: %ld\n",r_trig_class);
	printf("Source pattern: %#010x\n",r_source_pattern);
	printf("Trigger coupling: %ld ",r_trig_coupling);
	if(r_trig_coupling==0) printf(" (DC) \n");
	if(r_trig_coupling==1) printf(" (AC) \n");
	if(r_trig_coupling==3) printf(" (DC 50 Ohm - external) \n");
	if(r_trig_coupling==4) printf(" (AC 50 Ohm - external) \n");
	printf("Trigger slope: %ld ",r_trig_slope);
	if(r_trig_slope==0) printf(" (positive) \n");
	if(r_trig_slope==1) printf(" (negative) \n");
	printf("Trigger level: %7.2lf %%\n",r_trig_level);


	//read par config
	AqReadParameters *readPar=new AqReadParameters;
	AqDataDescriptor *dataDesc=new AqDataDescriptor;
	AqSegmentDescriptor *segDesc=new AqSegmentDescriptor[nbrSegments];
	readPar->dataType=1;
	readPar->readMode=1;
	readPar->nbrSegments=nbrSegments;
	readPar->firstSampleInSeg=0;
	readPar->segmentOffset=nbrSamples;
	readPar->firstSegment=0;
	readPar->nbrSamplesInSeg=nbrSamples;
	readPar->flags=0;
	readPar->reserved=0;
	readPar->reserved2=0.0;
	readPar->reserved3=0.0;     

	status=Acqrs_getInstrumentInfo(InstrumentID,"TbSegmentPad",&currentSegmentPad);
	check_stat(InstrumentID,status,"getInstrumentInfo>TbSegmentPad");
	status=AcqrsD1_getMemory(InstrumentID,&nbrSamplesNom,&nbrSegmentsNom);
	check_stat(InstrumentID,status,"getMemory");
	printf("nbrSamplesNom: %ld, nbrSegmentsNom: %ld\n",nbrSamplesNom,nbrSegmentsNom);
	readPar->dataArraySize=8*sizeof(short)*(nbrSamplesNom+currentSegmentPad)*(1+nbrSegments);
	dataArrayP1=(short*)malloc(readPar->dataArraySize);
	dataArrayP2=(short*)malloc(readPar->dataArraySize);
	readPar->segDescArraySize=sizeof(AqSegmentDescriptor)*nbrSegments;

	sprintf(fptc_name,"%s.ptc",target_path); 
	fptc=fopen(fptc_name,"w");

	//measurement loop
	for(i_Session=1;i_Session<=nbrSessions;i_Session++)
	{
		time(&rawtime);
		timeinfo=localtime(&rawtime);
		fprintf(fptc,"session %5ld: %s", i_Session,asctime(timeinfo));
		start_session=clock();
		sprintf(file_name,"%s-%ld.%s",target_path,i_Session,suffix);
		if(write_mode==0) 
		{
			fout=fopen(file_name,"wb");
			fwrite(&nbrSamples,sizeof(nbrSamples),1,fout);
			fwrite(&nbrSegments,sizeof(nbrSegments),1,fout);
			fwrite(&nbrWaveforms,sizeof(nbrWaveforms),1,fout);
		}
		if(write_mode==1) fout=fopen(file_name,"w");
		for(i_Waveform=1;i_Waveform<=nbrWaveforms;i_Waveform++)
		{
			//acquisition
			data_good=0;
			do
			{
			  AcqrsD1_acquire(InstrumentID);
			  status=AcqrsD1_waitForEndOfAcquisition(InstrumentID,timeOut);
			  if(status==VI_SUCCESS) data_good=1;
			  check_stat(InstrumentID,status,"waitForEndOfAcquisition");
			  status=AcqrsD1_stopAcquisition(InstrumentID);
			  check_stat(InstrumentID,status,"stopAcquisition");
			  } while(data_good!=1);

			//readout
		
			  status=AcqrsD1_readData(InstrumentID,1,readPar,dataArrayP1,dataDesc,segDesc);
			  check_stat(InstrumentID,status,"readData: channel 1");
			  status=AcqrsD1_readData(InstrumentID,2,readPar,dataArrayP2,dataDesc,segDesc);
			  check_stat(InstrumentID,status,"readData: channel 2");
			  printf("session %5ld: %6.2lf %% done\r",i_Session,100.0*(double)i_Waveform/(double)nbrWaveforms); 
		
			//write
			if(write_mode==0)
			{
				for(i=0;i<nbrSegments*nbrSamples;i++) 
				{
					fwrite(&dataArrayP1[i],sizeof(dataArrayP1[i]),1,fout);
					fwrite(&dataArrayP2[i],sizeof(dataArrayP2[i]),1,fout);
				}
			}//if
			if(write_mode==1)
			{
				for(j=0;j<nbrSegments;j++)
				{
					for(i=0;i<(nbrSamplesNom);i++) fprintf(fout,"%8ld %8ld %8ld %8ld\n",i+1,j*(nbrSamples)+i+1,dataArrayP1[j*(nbrSamples)+i],dataArrayP2[j*(nbrSamples)+i]);
					fprintf(fout,"\n");
				}
			}//if

		} //i_Waveform

		fclose(fout);
		stop_session=clock();
		t_session=stop_session-start_session;
		printf("session %5ld: %6.2lf %% done  %4.2lf 1/s\n",i_Session,100.0*(double)i_Waveform/(double)nbrWaveforms,nbrWaveforms/t_session*1000);
		fprintf(fptc,"session %5ld: %6.2lf %% done  %4.2lf 1/s\n",i_Session,100.0*(double)i_Waveform/(double)nbrWaveforms,nbrWaveforms/t_session*1000);
	} //i_Session
	fclose(fptc);
	//printf("\n");

	printf("returned Samples in Segment: %ld, Index First Point %ld\n",dataDesc->returnedSamplesPerSeg,dataDesc->indexFirstPoint);

}

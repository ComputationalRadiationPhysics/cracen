#ifndef _AGMD1FUNDAMENTALDATATYPES_H
#define _AGMD1FUNDAMENTALDATATYPES_H

//////////////////////////////////////////////////////////////////////////////////////////
//
//  AgMD1FundamentalDatatypes.h:    Agilent 'MD1' Data Types
//
//----------------------------------------------------------------------------------------
//  Copyright (C) 2000, 2001-2013 Agilent Technologies, Inc.
//
//////////////////////////////////////////////////////////////////////////////////////////

#include <vpptype.h>
#include "AcqirisIvi.h"


//////////////////////////////////////////////////////////////////////////////////////////
// Data Structures for '...readData' and '...writeData' functions

typedef struct 
{
    ViInt32 dataType;               //!< 0 = 8-bit (char)  1 = 16-bit (short) 2 = 32-bit (integer)  3 = 64-bit (real)
                                    //!< or use 'enum AqReadType' defined below
    ViInt32 readMode;               //!< Use 'enum AqReadDataMode' defined below
    ViInt32 firstSegment;
    ViInt32 nbrSegments;
    ViInt32 firstSampleInSeg;
    ViInt32 nbrSamplesInSeg;
    ViInt32 segmentOffset;
    ViInt32 dataArraySize;          //!< In bytes            
    ViInt32 segDescArraySize;       //!< In bytes            
    ViInt32 flags;                  //!< Use 'enum AqReadDataFlags' defined below
    ViInt32 reserved;
    ViReal64 reserved2;
    ViReal64 reserved3;
} AqReadParameters;

typedef struct 
{
    ViInt32 dataType;               //!< 0 = 8-bit (char)  1 = 16-bit (short) 2 = 32-bit (integer)  3 = 64-bit (real)
                                    //!< or use 'enum ReadType' defined below
    ViInt32 writeMode;              //!< Use 'enum AqWriteDataMode' defined below
    ViInt32 firstSegment;
    ViInt32 nbrSegments;
    ViInt32 firstSampleInSeg;
    ViInt32 nbrSamplesInSeg;
    ViInt32 segmentOffset;
    ViInt32 dataArraySize;          //!< In bytes            
    ViInt32 segDescArraySize;       //!< In bytes            
    ViInt32 flags;                  //!< Use 'enum AqReadDataFlags' defined below
    ViInt32 reserved;
    ViReal64 reserved2;
    ViReal64 reserved3;
} AqWriteParameters;

typedef struct
{
    ViReal64 horPos;
    ViUInt32 timeStampLo;           //!< Timestamp 
    ViUInt32 timeStampHi;
} AqSegmentDescriptor;

//! For 'readMode = ReadModeSeqRawW' only.
typedef struct
{
    ViReal64 horPos;
    ViUInt32 timeStampLo;           //!< Timestamp 
    ViUInt32 timeStampHi;
    ViUInt32 indexFirstPoint;       //!< Pointer to the first sample of this segment, 
                                    //!< into the array passed to 'readData'.
    ViUInt32 actualSegmentSize;     //!< In samples, for the circular buffer wrapping.
    ViInt32  reserved;
} AqSegmentDescriptorSeqRaw;

typedef struct
{
    ViReal64 horPos;
    ViUInt32 timeStampLo;           //!< Timestamp 
    ViUInt32 timeStampHi;
    ViUInt32 actualTriggersInSeg;   //!< Number of actual triggers acquired in this segment                
    ViInt32  avgOvfl;               //!< Acquisition overflow (avg)
    ViInt32  avgStatus;             //!< Average status (avg)
    ViInt32  avgMax;                //!< Max value in the sequence (avg)
    ViUInt32 flags;                 //!< Bits 0-3: markers (avg)
    ViInt32  reserved;
} AqSegmentDescriptorAvg;

//! For backward compatibility.
#define addrFirstPoint indexFirstPoint

typedef struct
{
    ViInt32     returnedSamplesPerSeg;
    ViInt32     indexFirstPoint;    //!< 'data[desc.indexFirstPoint]' is the first valid point. 
                                    //!< Note: Not valid for 'readMode = ReadModeSeqRawW'.
    ViReal64    sampTime;
    ViReal64    vGain;
    ViReal64    vOffset;
    ViInt32     returnedSegments;   //!< When reading multiple segments in one waveform
    ViInt32     nbrAvgWforms;        
    ViUInt32    actualTriggersInAcqLo;
    ViUInt32    actualTriggersInAcqHi;
    ViUInt32    actualDataSize;
    ViInt32     reserved2;    
    ViReal64    reserved3;
} AqDataDescriptor;

typedef struct     
{
    ViInt32 GatePos;
    ViInt32 GateLength;
} AqGateParameters;

//! Threshold parameters structure for AcqrsD1_configSetupArray with the U1084A SSR (Zero-Suppress) mode.
typedef struct
{
    ViReal64 threshold;         //!< Threshold value in Volts.
    ViUInt32 nextThreshSample;  //!< Sample index at which to switch to the next threshold
    ViInt32  reserved;          //!< Reserved; set to 0.
} AqThresholdGateParametersU1084A;

enum AqReadType 
{ 
    ReadInt8 = 0, 
    ReadInt16, 
    ReadInt32, 
    ReadReal64, 
    ReadRawData,
    nbrAqReadType,
};    

enum AqReadDataMode 
{
    ReadModeStdW = 0,   //!< Standard waveform
    ReadModeSeqW,       //!< Sequenced waveform
    ReadModeAvgW,       //!< Averaged waveform
    ReadModeGateW,      //!< Gated waveform
    ReadModePeak,       //!< Peaks of a waveform 
    ReadModeShAvgW,     //!< Standard short averaged
    ReadModeSShAvgW,    //!< Short shifted averaged waveform 
    ReadModeSSRW,       //!< Sustained sequential recording
    ReadModeZsW,        //!< Zero suppressed waveform
    ReadModeHistogram,  //!< Histogram
    ReadModePeakPic,    //!< Peak picture
    ReadModeSeqRawW,    //!< Raw Sequenced waveform (no unwrap)
    nbrAqReadDataMode 
};

enum AqReadDataFlags 
{
    AqIgnoreTDC          = 0x0001, //!< If set, the TDC value (if any) will be ignored.
    AqIgnoreLookUp       = 0x0002, //!< If set, the lookup table (if any) will not be applied on data.
    AqSkipClearHistogram = 0x0004, //!< If set, the histogram will be not be zero-ed during read
    AqSkipCircular2Linear= 0x0008, //!< If set, the memory image will be transferred in ReadModeSeqW
    AqDmaToCompanion     = 0x0010, //!< If set, a VX407c device will transfer data to its companion device.
};


//////////////////////////////////////////////////////////////////////////////////////////
// Constants and data types for D1 streaming
enum AqStreamBlockDescriptorVersion
{
    AqStreamBlockDescriptorVersion1 = 1,

    AqStreamBlockDescriptorVersionCurrent = AqStreamBlockDescriptorVersion1
};


typedef struct
{
    //! Version number of this structure.
    /*! Identifies the layout of this structure. Must be initialized to AqStreamBlockDescriptorVersionCurrent. */
    ViUInt32 descriptorVersion;
    
    //! Total number of bytes written to the data buffer.
    ViUInt32 nbrBytesReturned;
    
    //! Total number of samples included in this block.
    ViUInt32 nbrSamplesInBlock;
    
    //! Bitfield reporting the occurrence of various events.
    /*! See also enum AqStreamingBlockFlags. */
    ViUInt32 dataFlags;
    
    //! Reserved for future use.
    ViUInt64 timeStamp;
    
    //! Index of the first sample in this block relative to acquisition start.
    ViUInt64 firstSample;
    
    //! Index of the trigger sample relative to acquisition start.
    /*! This field is only valid if the 'AqStreamingFlagTrigger' bit is set in the 'flags' field.
        Otherwise, the value of this field is undefined. See also enum AqStreamingBlockFlags. */
    ViUInt64 triggerPos;
    
    //! Index of the first valid sample in case of a buffer overrun, relative to the beginning of this block.
    /*! This field is only valid if the 'AqStreamingFlagOverrun' bit is set in the 'flags' field.
        Otherwise, the value of this field is undefined. See also enum AqStreamingBlockFlags. */
    ViUInt64 overrunPos;
    
    //! Reserved.
    ViUInt64 reserved2;
    
    //! Reserved.
    ViUInt64 reserved3;
} AqStreamBlockDescriptor;


//! Constants for interpreting the 'flags' bit field in the AqStreamBlockDescriptor structure.
enum AqStreamingFlags
{
    AqStreamingFlagTrigger = (1 << 0),		//!< If set, indicates that a trigger has occurred in this block or a previous one.
    AqStreamingFlagOverrun = (1 << 1),		//!< If set, indicates that a buffer overrun has occurred in this block.
    AqStreamingFlagAdcOverload = (1 << 2),	//!< If set, indicated that an ADC overload has occurred during the current acquisition.
};


//////////////////////////////////////////////////////////////////////////////////////////
// Constants for D1 configMode

enum AqAcqMode
{
    AqAcqModeDigitizer      = 0,    //!< Normal Digitizer mode
    AqAcqModeRepeat         = 1,    //!< Continous acquisition and streaming to DPU (for ACs / SCs)
    AqAcqModeAverager       = 2,    //!< Averaging mode (for real-time averagers only)
    AqAcqModePingPong       = 3,    //!< Buffered acquisition (AP201 / AP101 only)
    AqAcqModePeakTDC        = 5,    //!< Peak detection
    AqAcqModeFreqCounter    = 6,    //!< Frequency counter
    AqAcqModeSSR            = 7,    //!< AP235 / AP240 SSR mode
    AqAcqModeDownConverter  = 12,   //!< Digital Down Conversion mode
    AqAcqModeBaseDesign     = 13,   //!< FDK Base Design (U1084A with CFW option only)
    AqAcqModeCustom         = 14,   //!< FDK Custom firmware (U1084A with CFW option only)
    AqAcqModeUserFDK        = 15    //!< User FDK mode, for M9703A FDK
};

//////////////////////////////////////////////////////////////////////////////////////////
// Constants for power

enum AqPowerState
{
    AqPowerOff = 0,
    AqPowerOn = 1,
    nbrAqPowerState,
};


//////////////////////////////////////////////////////////////////////////////////////////
// Constants for 'getAcqStatus'.
enum AqAcqStatus    
{    
    AqAcqDone = 0,      //!< Acquisition successfuly done. Data is available.
    AqAcqStopped,       //!< Acquisition stopped. Data is available, but might be invalid.
    AqAcqPretrigRun,    //!< Acquisition starting, but some pretrigger points still have to be acquired (Pretrigger run).
    AqAcqArmed,         //!< Acquisition running, ready to be triggered.
    AqAcqAcquiring,     //!< Acquisition running, acquiring data after the trigger occurred (Posttrigger run).
    AqAcqInvalid,       //!< Initial state, no operation has been done. No data available.
    AqAcqIOError,       //!< An IO error has been detected. The actual instrument state is unknown. No data available.
    AqAcqReady,         //!< Acquisition ready to be armed (pretrigger done, trigger disabled)
    nbrAqAcqStatus 
};


//////////////////////////////////////////////////////////////////////////////////////////
//  AcqrsT3Interface structure definitions

typedef struct
{
    ViAddr dataArray;           //!< Pointer to user allocated memory
    ViUInt32 dataSizeInBytes;   //!< Size of user allocated memory in bytes
    ViInt32 nbrSamples;         //!< Number of samples requested
    ViInt32 dataType;           //!< Use 'enum AqReadType' defined above
    ViInt32 readMode;           //!< Use 'enum AqT3ReadModes' defined below
    ViInt32 reserved3;          //!< Reserved, must be 0
    ViInt32 reserved2;          //!< Reserved, must be 0
    ViInt32 reserved1;          //!< Reserved, must be 0
} AqT3ReadParameters;

typedef struct
{
    ViAddr dataPtr;             //!< Pointer to returned data (same as user buffer if aligned)
    ViInt32 nbrSamples;         //!< Number of samples returned
    ViInt32 sampleSize;         //!< Size in bytes of one data sample
    ViInt32 sampleType;         //!< Type of the returned samples, see AqT3SampleType below
    ViInt32 flags;              //!< Readout flags
    ViInt32 reserved3;          //!< Reserved, will be 0
    ViInt32 reserved2;          //!< Reserved, will be 0
    ViInt32 reserved1;          //!< Reserved, will be 0
} AqT3DataDescriptor;

enum AqT3ReadModes
{
    AqT3ReadStandard,           //!< Standard read mode
    AqT3ReadContinuous,         //!< Continuous read mode
    nbrAqT3ReadModes,
};

enum AqT3SampleType
{
    AqT3SecondReal64,           //!< Double value in seconds
    AqT3Count50psInt32,         //!< Count of 50 ps
    AqT3Count5psInt32,          //!< Count of 5 ps
    AqT3Struct50ps6ch,          //!< Struct on 32 bits with (LSB to MSB)
                                //!<  27 bits count of 50 ps, 3 bits channel number, 1 bit overflow
    nbrAqT3SampleType,
};

//////////////////////////////////////////////////////////////////////////////////////////
// Device type. It determines the API interface that will be used.
enum AqDevType
{
    AqDevTypeInvalid = 0,   //!< Invalid, or 'ALL', depending on the context.
    AqD1 = 1,               //!< Digitizer.
    AqG2 = 2,               //!< Generator first generation (RC2xx).
    AqD1G2 = 3,             //!< Digitizer+Generator.
    AqT3 = 4,               //!< Digital timer/tester.
    AqG4 = 5,               //!< Generator second generation (GVMxxx).
    AqD1G4 = 6,             //!< Digitizer+GeneratorNG (RVMxxxx).
    AqP5 = 7,               //!< Processing board.
    nbrAqDevType,
};

/////////////////////////////////////////////////////////////
// Declarations for the Attribute based system
typedef enum 
{
    AqUnknownType = 0,
    AqLong,
    AqDouble, 
    AqString,
    AqInt64,
    nbrAqAttrType
} AqAttrType;

typedef enum 
{
    AqUnknownAccess = 0,
    AqRO,                    //!< Read Only
    AqWO,                    //!< Write Only
    AqRW,                    //!< Read/Write
    nbrAqAttrAccess
} AqAttrAccess;

typedef enum 
{
    AqUnknownCategory = 0, 
    AqInstrument,          //!< Attribute is at instrument level
    AqChannel,             //!< Attribute is at channel level
    AqControlIO,           //!< Attribute is for Control I/O
    AqAverager,            //!< Attribute is for Instrument averager capability (if any)
    AqLogicalDevice,       //!< Attribute is related to a Logical Device, (for ie an FPGA)
    AqInternal,            //!< Attribute is a driver internal value (private)
    AqHardwareRegister,    //!< Attribute is a hardware register (private)
    AqEepromData,          //!< Attribute is data in a Eeprom (private)
    AqCalibration,         //!< Attribute is calibration enable/disable (private).
    AqProfiling,           //!< Attribute is a profiling dedicated attribute (private)
    AqInternalHW,          //!< Attribute is a driver internal value (private) related to hardware. It does not have consistency.
    nbrAqAttrCategory
} AqAttrCategory;

//! The order is determinant. More visibles must have highest number.
typedef enum 
{
    AqUnknownVisibility = 0, 
    AqPrivate,
    AqPublic,
    nbrAqAttrVisibility
} AqAttrVisibility;

typedef struct AqAttrDescStruct
{
    ViConstString           name;               //!< Attribute's name. Use this one for setting or getting an attribute.        
    ViSession               instrumentId;       //!< Instrument id who own this attribute
    AqAttrType              type;               //!< ie: Long, Double, String.
    AqAttrAccess            access;             //!< ie: GetOnly, GetSet.
    AqAttrCategory          category;           //!< ie: Instrument, Channel, System... etc.
    AqAttrVisibility        visibility;         //!< ie: Private, Public.
    ViInt32                 channel;            //!< Channel that own this attribute. If negative this number indicates 
                                                //!< an external channel. For ie: 1 = ch1, -1 = ext1.
                                                //!< If the attribute does not belong to a channel, this value is 0.
    ViInt32                 numberInCategory;   //!< Distinguish an attribute in same category. 
                                                //!< For category HardwareRegister, numberInCategory is the address
    ViInt32                 reserved1;          //!< Unused.
    ViInt32                 reserved2;          //!< Unused.
    ViBoolean               reserved3;          //!< Unused.
    ViBoolean               isStringCompatible; //!< Return '1' if this attribute can be manipulated through
                                                //!< 'setAttributeString/getAttributeString'.
    ViBoolean               hasARangeTable;     //!< Return '1' if this attribute has a range table (see getAttributeRangeTable())
    ViBoolean               isLocked;           //!< Return '1' if the driver's value is locked. (see setAttributexxxx())
    struct AqAttrDescStruct*    nextAttribute;  //!< Pointer to the next attribute. 
                                                //!< NULL if there isn't any next attribute. For ie we have reach 
                                                //!< the last.
} AqAttrDescriptor;
#endif // sentry


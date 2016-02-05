///////////////////////////////////////////////////////////////////////////////////////////////////
//
//  AgMD1Fundamental.h:    Digitizer Driver Function Declarations
//
//----------------------------------------------------------------------------------------
//  Copyright (C) Agilent Technologies, Inc. 2010, 2011-2013
//
//  Purpose:    Declaration of AgMD1Fundamental device driver API
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#include <vpptype.h>
#include "AgMD1FundamentalDataTypes.h"
#include "AgMD1FundamentalErrorCodes.h"

// Calling convention used: __cdecl by default (note: _VI_FUNC is synonymous to __stdcall)
#if !defined( __vxworks ) && !defined( _LINUX )
#define ACQ_CC __stdcall
#else
#define ACQ_CC
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////      
// General comments about the function prototypes:
//
// - All function calls require the argument 'instrumentID' in order to identify
//   the Agilent Technologies Digitizer card to which the call is directed.
//   The only exceptions are the initialization functions 'AcqrsD1_getNbrInstruments',
//   'AcqrsD1_setSimulationOptions', 'AcqrsD1_init' and 'AcqrsD1_InitWithOptions'.
//   The last two functions actually return instrument identifiers at initialization time,
//   for subsequent use in the other function calls.
//
// - All function calls return a status value of type 'ViStatus' with information about
//   the success or failure of the call.
//   The Agilent Technologies specific values are defined in the AgMD1FundamentalErrorCodes.h file.
//   The generic VISA ones are listed in the header file 'vpptype.h'.
//
// - If important parameters supplied by the user (e.g. an invalid instrumentID) are found
//   to be invalid, most functions do not execute and return an error code of the type
//   VI_ERROR_PARAMETERi, where i = 1, 2, ... corresponds to the argument number.
//
// - If the user attempts (with a function $prefixD1$_configXXXX) to set a digitizer
//   parameter to a value which is outside of its acceptable range, the function
//   typically adapts the parameter to the closest available value and returns
//   ACQIRIS_WARN_SETUP_ADAPTED. The digitizer parameters actually set can be retrieved
//   with the 'query' functions $prefixD1$_getXXXX.
//
// - All calls to an instrument which was previously suspended using 'AcqrsD1_suspendControl' will return
//   ACQIRIS_ERROR_INVALID_HANDLE, until 'AcqrsD1_resumeControl' is called.
//
// - Data are always returned through pointers to user-allocated variables or arrays.
//
///////////////////////////////////////////////////////////////////////////////////////////////////      

#ifdef __cplusplus

// Declare the functions as being imported...
#if !defined( __vxworks ) && !defined( _LINUX ) && !defined( _ETS )
#define ACQ_DLL __declspec(dllimport)
#else
#define ACQ_DLL
#endif

// ...and declare C linkage for the imported functions if in C++
    extern "C" {

#else

// In C, simply declare the functions as being 'extern' and imported
#if !defined( __vxworks ) && !defined( _LINUX ) && !defined( _ETS )
#define ACQ_DLL extern __declspec(dllimport)
#else
#define ACQ_DLL 
#endif

#endif

///////////////////////////////////////////////////////////////////////////////////////////////////      
// Generic functions
///////////////////////////////////////////////////////////////////////////////////////////////////      

//! Performs an auto-calibration of the instrument.
/*! Equivalent to Acqrs_calibrateEx with 'calType' = 0 */
ACQ_DLL ViStatus ACQ_CC Acqrs_calibrate(ViSession instrumentID); 



//! Interrupts the calibration and return.
/*! If a calibration is run by another thread, this other thread will be interrupted immediately and it
    will get the error 'ACQIRIS_ERROR_OPERATION_CANCELLED'.

Returns one of the following ViStatus values:
    VI_SUCCESS                           Always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_calibrateCancel(ViSession instrumentID);



//! Performs a (partial) auto-calibration of the instrument.
/*! 'calType'=  0    calibrate the entire instrument (equivalent to 'Acqrs_calibrate')
                1    calibrate only the current channel configuration, 
                     as set by 'Acqrs_configChannelCombination'
                2    calibrate external clock timing. Requires operation in External Clock 
                     (Continuous), i.e. 'clockType' = 1, as set with 'Acqrs_configExtClock'
                3    calibrate only at the current frequency (only instruments with fine resolution 
                     time bases)
                4    calibrate fast, only at the current settings (if supported). 
                     Note: In this mode, any change on 'fullscale', 'bandwidth limit', 'channel 
                     combine', 'impedance' or 'coupling' will require a new calibration.
                      
    'modifier'    
             ['calType' = 0]    currently unused, set to zero
             ['calType' = 1]    currently unused, set to zero
             ['calType' = 2]    currently unused, set to zero
             ['calType' = 3]  0    = All channels
                             >0    = Only specified channel (for i.e. 1 = calibrate only channel 1)
             ['calType' = 4]  0    = All channels
                             >0    = Only specified channel (for i.e. 1 = calibrate only channel 1)

    'flags'            currently unused, set to zero 

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_INSTRUMENT_RUNNING   if the instrument is currently running.
    ACQIRIS_ERROR_CALIBRATION_FAILED   if the requested calibration failed.
    ACQIRIS_ERROR_NOT_SUPPORTED        if the requested calibration is not supported by the 
                                       instrument.
    ACQIRIS_ERROR_CANNOT_READ_THIS_CHANNEL if the requested channel is not available.
    ACQIRIS_ERROR_COULD_NOT_CALIBRATE  if the requested frequency is invalid ('calType' = 2 only).
    ACQIRIS_ERROR_ACQ_TIMEOUT          if an acquisition timed out during the calibration 
                                       (e.g. no clocks provided).
    ACQIRIS_ERROR_CANCELLED            if the calibration has been cancelled.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_calibrateEx(ViSession instrumentID, ViInt32 calType,
    ViInt32 modifier, ViInt32 flags); 



//! Load from 'filePathName' binary file all calibration values and info.
/*! 'filePathName'     name of the file,path or 'NULL' (see 'flags').
    'flags'            = 0, default filename. Calibration values will be loaded from the snXXXXX_calVal.bin
                            file in the working directory. 'filePathName' MUST be NULL or (empty String).
                       = 1, specify path only. Calibration values will be loaded from the snXXXXX_calVal.bin
                            file in the speficied directory. 'filePathName' MUST be non-NULL.
                       = 2, specify filename. 'filePathName' represents the filename (with or without path),
                            and MUST be non-NULL and non-empty.
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_CAL_FILE_CORRUPTED         if the file is corrupted.
    ACQIRIS_ERROR_CAL_FILE_VERSION           if the file has been generated with a different driver version (major and minor).
    ACQIRIS_ERROR_CAL_FILE_SERIAL            if the file does not correspond to this (multi)instrument.
    ACQIRIS_ERROR_FILE_NOT_FOUND             if the file is not found.
    ACQIRIS_ERROR_NOT_SUPPORTED              if the instrument does not support this feature. 
    VI_SUCCESS                               otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_calLoad(ViSession instrumentID, ViConstString filePathName, ViInt32 flags);



//! Query about the desirability of a calibration for the current requested configuration.
/*! 'chan '      = 0, the query will be done on all channels i.e. isRequiredP = VI_TRUE if at least 1 channel needs to be calibrated.
                 = n, the query will be done on channel n

    Returns 'isRequiredP':
                 VI_TRUE if the channel number 'chan' of the instrument needs to be calibrated.
                         This is the case if it has been calibrated more than two hours ago,
                         or if the temperature deviation since the last calibration is more than 5 degrees.
                 VI_FALSE otherwise.

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED             if the instrument does not support this feature. 
    VI_SUCCESS                              otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_calRequired(ViSession instrumentID, ViInt32 channel, ViBoolean* isRequiredP);



//! Save in 'filePathName' binary file all calibration values and info.
/*! 'filePathName'     name of the file,path or 'NULL' (see 'flags').
    'flags'            = 0, default filename. Calibration values will be saved in the snXXXXX_calVal.bin
                            file in the working directory. 'filePathName' MUST be NULL or (empty String).
                       = 1, specify path only. Calibration values will be saved in the snXXXXX_calVal.bin
                            file in the speficied directory. 'filePathName' MUST be non-NULL.
                       = 2, specify filename. 'filePathName' represents the filename (with or without path),
                            and MUST be non-NULL and non-empty.


    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED             if the instrument does not support this feature.
    ACQIRIS_ERROR_NO_ACCESS                 if the access to the file is denied.
    VI_SUCCESS                              otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_calSave(ViSession instrumentID, ViConstString filePathName, ViInt32 flags);



//! Close specified instrument. 
/*! Once closed, this instrument is not available anymore and
    need to be reenabled using 'InitWithOptions' or 'init'.
    Note: For freeing properly all resources, 'closeAll' must still be called when
    the application close, even if 'close' was called for each instrument.

    Returns one of the following ViStatus values:
    VI_SUCCESS always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_close(ViSession instrumentID); 



//! Closes all instruments and prepares for closing of application.
/*! 
    Returns one of the following ViStatus values:
    VI_SUCCESS always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_closeAll(void); 



//! Load/Clear/SetFilePath of the on-board logic devices. 
/*! ONLY useful for a module with on-board programmable logic devices 
   (SCxxx, ACxxx, APxxx, and 12-bit Digitizers).

   'flags'= operation
            = 0 program device, using strictly the specified path/file name
            = 1 clear device
            = 2 clear and program devices using appropriate files from specified path.
            = 3 program device, but also look for file path in the 'AgMD1Fundamental.ini' file, if the file 
                was not found. It is sufficient to specify the file name (without explicit path), 
                if the file is either in the current working directory or in the directory pointed 
                to by 'FPGAPATH' in 'AgMD1Fundamental.ini'.

   'deviceName'   Identifies which device to program. 
      ['flags' = 0 or 3] = device to program.
          This string must be "Block1Devx", with 'x' = is the FPGA number to be programmed. 
          For instance, in the SC240, it must be "Block1Dev1". 
      ['flags' = 1] = device to clear, must be "Block1DevAll". 
      ['flags' = 2] = unused.

   'filePathName'  
      ['flags' = 0 or 3] = file path and file name. 
      ['flags' = 1] = unused. 
      ['flags' = 2] = path (no file name !) where all .bit files can be found.

   Note: Most users do not need to call this function. Check the manual for further details.

   Returns one of the following ViStatus values:
   ACQIRIS_ERROR_PARAM_STRING_INVALID    if 'deviceName' is invalid.
   ACQIRIS_ERROR_FILE_NOT_FOUND          if an FPGA file could not be found.
   ACQIRIS_ERROR_NO_DATA                 if an FPGA file did not contain expected 
                                         data.
   ACQIRIS_ERROR_FIRMWARE_NOT_AUTHORIZED if the instrument is not authorized to use the 
                                         specified 'filePathName' file.
   ACQIRIS_ERROR_EEPROM_DATA_INVALID     if the CPLD 'filePathName' is invalid.
   ACQIRIS_ERROR_FPGA_y_LOAD             if an FPGA could not be loaded, 
                                         where 'y' = FPGA nbr. 
   ACQIRIS_WARN_SETUP_ADAPTED            if one of the parameters has been adapted.
   VI_SUCCESS                            otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_configLogicDevice(ViSession instrumentID, 
    ViConstString deviceNameP, ViConstString filePathNameP, ViInt32 flags);



//! Translates an error code into a human readable form. 
/*! For file errors, the returned message will also contain the file name and the 
    original 'ansi' error string.

   'errorCode'          = Error code (returned by a function) to be translated
   'errorMessage'       = Pointer to user-allocated character string (suggested size 512 bytes),
                          into which the error message text will be copied.
   'errorMessageSize'   = size of 'errorMessage', in bytes.

   NOTE: 'instrumentID' can be VI_NULL.

   Returns one of the following ViStatus values:
   ACQIRIS_ERROR_BUFFER_OVERFLOW    if 'errorMessageSize' is too small.
   VI_SUCCESS                       otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_errorMessage(ViSession instrumentID, ViStatus errorCode, 
    ViChar errorMessage[], ViInt32 errorMessageSize);



//! Return through "attrDescriptorP" the value of the attribute named "name".
/*! 'channel'   = 0   for instrument related attributes.
                = x   for channel related attributes, where 'x' is the channel number.

    'name'      = name of the attribute.

    'attrDescriptorP'    = the pointer where the value will be written.

   Returns one of the following ViStatus values:

    ACQIRIS_ERROR_ATTR_NOT_FOUND        if the attribute is not found.
    VI_SUCCESS                          otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getAttributeDescriptor(ViSession instrumentID, ViInt32 channel, ViConstString name, AqAttrDescriptor* attrDescriptorP); 



//! Return through "valueP" the value of the attribute named "name".
/*! 'channel'   = 0   for instrument related attributes.
                = x   for channel related attributes, where 'x' is the channel number.

    'name'      = name of the attribute.

    'valueP'    = the pointer where the value will be written.

   Returns one of the following ViStatus values:

    ACQIRIS_ERROR_ATTR_NOT_FOUND        if the attribute is not found. 
    ACQIRIS_ERROR_ATTR_WRONG_TYPE       if the attribute is found but not of the expected type.
    VI_SUCCESS                          otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getAttributeDouble(ViSession instrumentID, ViInt32 channel, ViConstString name, ViReal64* valueP); 



//! Return through "valueP" the value of the attribute named "name".
/*! 'channel'   = 0   for instrument related attributes.
                = x   for channel related attributes, where 'x' is the channel number.

    'name'      = name of the attribute.

    'valueP'    = the pointer where the value will be written.

   Returns one of the following ViStatus values:

    ACQIRIS_ERROR_ATTR_NOT_FOUND        if the attribute is not found. 
    ACQIRIS_ERROR_ATTR_WRONG_TYPE       if the attribute is found but not of the expected type.
    VI_SUCCESS                          otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getAttributeLong(ViSession instrumentID, ViInt32 channel, ViConstString name, ViInt32* valueP); 



//! Return through "rangeTableP" the range table of the attribute named "name".
/*! 'channel'   = 0   for instrument related attributes.
                = x   for channel related attributes, where 'x' is the channel number.

    'name'      = name of the attribute.

    'rangeTableP'    = the pointer where the value will be written.

   Returns one of the following ViStatus values:

    ACQIRIS_ERROR_ATTR_NOT_FOUND        if the attribute is not found. 
    ACQIRIS_ERROR_ATTR_WRONG_TYPE       if the attribute is found but not of the expected type.
    ACQIRIS_ERROR_NOT_SUPPORTED         if found but not implemented.
    VI_SUCCESS                          otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getAttributeRangeTable(ViSession instrumentID, ViInt32 channel, ViConstString name, AqAttrRangeTablePtr rangeTableP); 



//! Return through "string" a copy of the string expression of the attribute named "name".
/*! 'channel'   = 0   for instrument related attributes.
                = x   for channel related attributes, where 'x' is the channel number.

    'name'      = name of the attribute.

    'string'    = the string where the value will be written.

    'bufferSize'= the size of the 'string' buffer in bytes.

   Returns one of the following ViStatus values:

    ACQIRIS_ERROR_ATTR_NOT_FOUND        if the attribute is not found. 
    ACQIRIS_ERROR_ATTR_WRONG_TYPE       if the attribute is found but not of the expected type.
    ACQIRIS_ERROR_BUFFER_OVERFLOW       if 'bufferSize' is too small.
    VI_SUCCESS                          otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getAttributeString(ViSession instrumentID, ViInt32 channel, ViConstString name, ViString string, ViInt32 bufferSize); 



//! Returns the API interface type appropriate to this 'instrumentID' and 'channel'.
/*! 'devTypeP'          = The device type. See 'AqDevType'.

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getChanDevType(ViSession instrumentID, ViInt32 channel, ViInt32* devTypeP);



//! Returns the device type of the specified 'instrumentID'.
/*! 'devTypeP'          = The device type. See 'AqDevType'.

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getDevType(ViSession instrumentID, ViInt32* devTypeP);



//! Returns the device type of the instrument specified by 'deviceIndex'.
/*! 'devTypeP'          = The device type. See 'AqDevType'.

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getDevTypeByIndex(ViInt32 deviceIndex, ViInt32* devTypeP);



//! Returns the device type of the instrument deduced by the resource string.
/*! 'devTypeP'          = The device type. See 'AqDevType'.

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getDevTypeByResourceString(ViConstString resourceP, ViInt32* devTypeP);



//! Returns some basic data about a specified instrument.
/*! Values returned by the function:

    'name'            = pointer to user-allocated string, into which the
                        model name is returned (length < 32 characters).
    'serialNbr'       = serial number of the instrument.
    'busNbr'          = bus number where the instrument is located.
    'slotNbr'         = slot number where the instrument is located.

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getInstrumentData(ViSession instrumentID,
    ViChar name[], ViInt32* serialNbr, ViInt32* busNbrP, ViInt32* slotNbrP);



//! Returns some basic data about a specified instrument.
/*! Values returned by the function:

    'modelNumberP'      = pointer to user-allocated string, into which the
                          model number is returned (length < 32 characters)
    'modelNumberSize'   = size of the 'modelNumber' buffer.
    'serialP'           = pointer to user-allocated string, into which the 
                          serial number of the instrument is returned (length < 32 characters).
    'serialSize'        = size of the 'serial' buffer.
    'busNbrP'           = bus number where the instrument is located
    'slotNbrP'          = slot number where the instrument is located
    'crateNbrP'         = crate where the instrument is located.
    'posInCrateP'       = position in the crate where the instrument is located.

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getInstrumentDataEx(ViSession instrumentID,
    ViChar modelNumberP[], ViUInt32 modelNumberSize, ViChar serialP[], ViUInt32 serialSize, 
    ViUInt32* busNbrP, ViUInt32* slotNbrP, ViUInt32* crateNbrP, ViUInt32* posInCrateP);



//! Returns general information about a specified instrument.
/*! The following value must be supplied to the function:

    'parameterString'  = character string specifying the requested parameter
                         Please refer to the manual for the accepted parameter strings

    Value returned by the function:

    'infoValue'        = value of requested parameter
                         The type of the returned value depends on the parameter string
                         Please refer to the manual for the required  data type as a
                         function of the accepted parameter strings
                         NOTE to C/C++ programmers: 'ViAddr' resolves to 'void*'

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_PARAM_STRING_INVALID if 'parameterString' is invalid.
    ACQIRIS_ERROR_NOT_SUPPORTED        if the requested value is not supported by the 
                                       instrument.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getInstrumentInfo(ViSession instrumentID, 
    ViConstString parameterString, ViAddr infoValue);



//! Returns the number of channels on the specified instrument.
/*! Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getNbrChannels(ViSession instrumentID, ViInt32* nbrChannels);



//! Returns the number of supported physical devices found on the computer.
/*! Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getNbrInstruments(ViInt32* nbrInstruments);



//! Returns version numbers associated with a specified instrument / current device driver.
/*! The following value must be supplied to the function:

    'versionItem'    =   1        for version of Kernel-Mode Driver 
                         2        for version of EEPROM Common Section 
                         3        for version of EEPROM Digitizer Section
                         4        for version of CPLD firmware
 
    Value returned by the function:

    'version'        = version number.

    For drivers, the version number is composed of 2 parts. The upper 2 bytes represent
    the major version number, and the lower 2 bytes represent the minor version number. 

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_getVersion(ViSession instrumentID,
    ViInt32 versionItem, ViInt32* version);



//! Initializes an instrument.
/*! See remarks under 'Acqrs_InitWithOptions' */
ACQ_DLL ViStatus ACQ_CC Acqrs_init(ViRsrc resourceName, ViBoolean IDQuery, 
    ViBoolean resetDevice, ViSession* instrumentID);
 


//! Initializes an instrument with options.
/*! The following values must be supplied to the function:

    'resourceName'   = an ASCII string which identifies the instrument to be initialized
                       See manual for a detailed description of this string.
    'IDQuery'        = currently ignored
    'resetDevice'    = if set to 'TRUE', resets the instrument after initialization
    'optionsString'  = an ASCII string which specifies options. Currently, we support
                       "CAL=False" to suppress calibration during the initialization
                       "DMA=False" to inhibit (for diagnostics) the use of scatter-gather DMA for 
                       data transfers
                       "Simulate=True" for the use of simulated instruments during application 
                       development. 
    Values returned by the function:

    'instrumentID'   = identifier for subsequent use in the other function calls.

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_INIT_STRING_INVALID        if the 'resourceName' is invalid.
    ACQIRIS_ERROR_INTERNAL_DEVICENO_INVALID  if the 'resourceName' contains values that are not 
                                             within the expected ranges (e.g. wrong serial number).
    ACQIRIS_ERROR_TOO_MANY_DEVICES           if there are too many devices installed.
    ACQIRIS_ERROR_KERNEL_VERSION             if the instrument require a newer kernel driver.
    ACQIRIS_ERROR_HW_FAILURE                 if the instrument doesn't answer properly to 
                                             defined requests.
    ACQIRIS_ERROR_HW_FAILURE_CHx             if one of the channels doesn't answer properly to 
                                             defined requests, where 'x' = channel number.
    ACQIRIS_ERROR_FILE_NOT_FOUND             if a required FPGA file could not be found.
    ACQIRIS_ERROR_NO_DATA                    if a required FPGA file did not contain expected data.
    VI_SUCCESS                               otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_InitWithOptions(ViRsrc resourceName, ViBoolean IDQuery, 
    ViBoolean resetDevice, ViConstString optionsString, ViSession* instrumentID);



//! Reads/writes data values from programmable logic devices.
/*! Reads/writes a number of 32-bit data values from/to a user-defined register in the
    logic device identified by 'deviceName[]'. 
    ONLY useful for a instrument with on-board programmable logic devices.
    Currently ONLY for SCxxx and ACxxx!

    The following values must be supplied to the function:

    'deviceName'       = an ASCII string which identifies the device
                         Must be of the form "BlockMDevN", where M = 1..4 and N = 1..number
                         of logical devices in the device block M.
                         In the AC/SC Analyzers & the RC200, this string must be "Block1Dev1"
                         See manual for a detailed description of this string.
    'registerID'       = 0..to (nbrRegisters-1)
    'nbrValues'        = number of 32-bit values to be transferred
    'dataArray[]'      = user-supplied buffer for data values
    'readWrite'        = 0 for reading (from logic device to application)
                         1 for writing (from application to logic device)
                         1002 to read core version numbers. registerID is used as CoreID.
                              dataArray[0] will contain raw major/minor on 16/16 bits.
                              dataArray[1...] will contain the version string.
    'flags'            : bit31=1 forces not to use DMA transfer to FPGA

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_PARAM_STRING_INVALID   if the 'deviceName' is not valid.
    ACQIRIS_ERROR_NO_ACCESS              if the operation is not authorized.
    ACQIRIS_ERROR_NOT_SUPPORTED          if the operation is not supported by the instrument,  
                                             or if 'registerID' is outside the expected values.
    ACQIRIS_ERROR_INSTRUMENT_RUNNING     if the instrument was not stopped beforehand.
    ACQIRIS_ERROR_DATA_ARRAY             if 'dataArray' is NULL.
    ACQIRIS_ERROR_IO_WRITE               if a 'write' verify failed.
    VI_SUCCESS                           otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_logicDeviceIO(ViSession instrumentID,
    ViConstString deviceName,   ViInt32 registerID,           ViInt32 nbrValues,
    ViInt32 dataArray[],        ViInt32 readWrite,            ViInt32 flags);



//! Forces all instruments to prepare entry into or return from the system power down state.
/*! Typically, this function is called by a 'Power Aware' application, 
    when it catches a 'system power down' event, such as 'hibernate'. 

    If 'state == 0' (AqPowerOff), it will suspend all other calling threads. If a thread
    is performing a long operation which cannot be completed within milliseconds, 
    such as 'calibrate', it will be interrupted immediately and will get the status 
    'ACQIRIS_ERROR_OPERATION_CANCELLED'. Note that if an acquisition is still running
    when 'powerSystem(0, 0)' is called, it might be incomplete or corrupted.
   
    If 'state == 1' (AqPowerOn), it will re-enable the instruments in the same state as they
    were before 'powerSystem(0, 0)'. Threads which were suspended will be resumed.
    However, interrupted operations which returned an error 
    'ACQIRIS_ERROR_OPERATION_CANCELLED' have to be redone.

    The following values must be supplied to the function:

   'state'          = 0 (=AqPowerOff) : prepare for power down.
                    = 1 (=AqPowerOn)  : re-enable instruments after power down.

   'flags'          = Unused, must be 0.

   Returns one of the following ViStatus values:
   VI_SUCCESS                           always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_powerSystem(ViInt32 state, ViInt32 flags);



//! Resets an instrument.
/*! Returns one of the following ViStatus values:
   VI_SUCCESS                           always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_reset(ViSession instrumentID);


//! Resets the device memory to a known default state. 
/*! Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED         if the instrument does not support this feature. 
    ACQIRIS_ERROR_INSTRUMENT_RUNNING    if the instrument is already running. 
    VI_SUCCESS                          otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_resetMemory(ViSession instrumentID);



//! Resume control of an instrument.
/*! This function reacquires the driver lock of the instrument and allows calls to it from 
    the current process (Windows only). After successfully calling 'Acqrs_resumeControl', the module will be
    set to a default hardware state. It will have no valid data and the timestamp will be set
    to 0. When the next acquisition is started, the module will be configured with all of the 
    unmodified settings from before the 'Acqrs_suspendControl' was invoked.
   
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_DEVICE_ALREADY_OPEN       if the instrument is already used by another process
    VI_SUCCESS                              otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_resumeControl(ViSession instrumentID);

//! Causes the instrument to perform a self test.
/*! Self Test waits for the instrument to complete the test. 
    It then queries the instrument for the results of the self test 
    and returns the results to the user.

    When calling the Self Test function through a C interface, 
    the user should pass a buffer with at least 256 bytes for the TestMessage parameter.

    'testResult'            = Returns the numeric result from the self test operation (0 = no error, e.g. the test passed)
    'testMessage'           = Returns the self test status message.
    'testMessageBufferSize' = Tells the size of the provided buffer.
  
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED
    VI_SUCCESS                              otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_selfTest(ViSession instrumentID, ViInt16* testResult, ViChar testMessage[], ViInt16 testMessageBufferSize);

//! Set through "value" the value of the attribute named "name". 
/*! 'channel'   = 0   for instrument related attributes.
                = x   for channel related attributes, where 'x' is the channel number.

    'name'      = name of the attribute.

    'value'     = value of the attribute.

   Returns one of the following ViStatus values:

    ACQIRIS_ERROR_ATTR_NOT_FOUND        if the attribute is not found. 
    ACQIRIS_ERROR_ATTR_WRONG_TYPE       if the attribute is found but not of the expected type.
    ACQIRIS_ERROR_ATTR_IS_READ_ONLY     if the attribute is found but not writable.
    VI_SUCCESS                          otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_setAttributeDouble(ViSession instrumentID, ViInt32 channel, ViConstString name, ViReal64 value); 



//! Set through "value" the value of the attribute named "name". 
/*! 'channel'   = 0   for instrument related attributes.
                = x   for channel related attributes, where 'x' is the channel number.

    'name'      = name of the attribute.

    'value'     = value of the attribute.

   Returns one of the following ViStatus values:

    ACQIRIS_ERROR_ATTR_NOT_FOUND        if the attribute is not found. 
    ACQIRIS_ERROR_ATTR_WRONG_TYPE       if the attribute is found but not of the expected type.
    ACQIRIS_ERROR_ATTR_IS_READ_ONLY     if the attribute is found but not writable.
    VI_SUCCESS                          otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_setAttributeLong(ViSession instrumentID, ViInt32 channel, ViConstString name, ViInt32 value); 



//! Set through 'value' the value of the attribute named 'name'.
/*! 'channel'            = 1...Nchan (as returned by 'Acqrs_getNbrChannels' ) or
                           0 if it is an attribute related to the instrument 
                           itself (i.e. not to a channel).
    'name'               = specify the name of the attribute to change.
                           Please refer to the manual for the accepted names.
    'value'              = specify the value in which the attribute will be set.
                           Please refer to the manual for the accepted values.

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_ATTR_NOT_FOUND         if not found or if a wrong 'channel' is specified. 
    ACQIRIS_ERROR_ATTR_WRONG_TYPE        if found but not of the expected type.
    ACQIRIS_ERROR_ATTR_INVALID_VALUE     if 'value' is not valid.
    ACQIRIS_ERROR_ATTR_IS_READ_ONLY      if found but not writable.
    VI_SUCCESS                           otherwise. */
ACQ_DLL ViStatus ACQ_CC Acqrs_setAttributeString(ViSession instrumentID, ViInt32 channel, 
    ViConstString name, ViConstString value); 



//! Sets the front-panel LED to the desired color.
/*! 'color' = 0        OFF (returns to normal 'acquisition status' indicator)
              1        Green
              2        Red
              3        Yellow
             -1        Blinking (not supported on all families)

    Returns one of the following ViStatus values:
   VI_SUCCESS                           always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_setLEDColor(ViSession instrumentID, ViInt32 color);



//! Set simulation options.
/*! Sets one or several options which will be used by the function 'Acqrs_InitWithOptions',
    provided that the 'optionString' supplied to 'Acqrs_InitWithOptions' contains the
    string 'simulate=TRUE' (or similar).
    Refer to the manual for the accepted form of 'simOptionString'.
    The simulation options are reset to none by setting 'simOptionString' to an empty string "".

   Returns one of the following ViStatus values:
   VI_SUCCESS                           always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_setSimulationOptions(ViConstString simOptionString);



//! Supend Control of an instrument.
/*! This function releases the driver lock of the instrument and prevents all further calls
    from the current process (Windows only). Use Acqrs_resumeControl to reacquire the 
    control of the instrument. Once suspended, this instrument can be used from another process.
    However, if this is the first time this other process is used, all desired acquisition settings
    must be defined and a calibration will be needed.

    Returns one of the following ViStatus values:
   VI_SUCCESS                           always. */
ACQ_DLL ViStatus ACQ_CC Acqrs_suspendControl(ViSession instrumentID);



///////////////////////////////////////////////////////////////////////////////////////////////////
// D1 Digitizers Functions
///////////////////////////////////////////////////////////////////////////////////////////////////

//! Checks if the acquisition has terminated.
/*! Returns 'done' = VI_TRUE if the acquisition is terminated

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_IO_READ    if a link error has been detected (e.g. PCI link lost).
    VI_SUCCESS otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_acqDone(ViSession instrumentID, ViBoolean* done);
      


//! Starts an acquisition. 
/*! This function is equivalent to 'acquireEx' with 'acquireMode = 0, 
    acquireFlags = 0'

    Common return values:
    ACQIRIS_ERROR_INSTRUMENT_RUNNING    if the instrument is already running. 
    ACQIRIS_ERROR_INCOMPATIBLE_MODE     if acquire is not available in the current mode.
    VI_SUCCESS otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_acquire(ViSession instrumentID);



//! Starts an acquisition.
/*! The following values must be supplied to the function:

    'acquireMode'      = 0     Normal, start an acquisition and return immediately (equivalent to 
                               function 'acquire').
                       = 2     Averagers only! Sets continuous accumulation and starts an 
                               acquisition.
              
    'acquireFlags'     = 0     No flags.
                       = 4     Reset timestamps (if supported). 
    'acquireParams'    Unused, must be set to 0.
    'reserved'         Unused, must be set to 0.

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_INSTRUMENT_RUNNING if the instrument is already running. 
    ACQIRIS_ERROR_NOT_SUPPORTED      if the requested mode or flag is not supported by the  
                                     instrument.
    ACQIRIS_ERROR_INCOMPATIBLE_MODE  if acquireEx is not available in the current mode.
    VI_SUCCESS                       otherwise.
*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_acquireEx(ViSession instrumentID, ViInt32 acquireMode, 
        ViInt32 acquireFlags, ViInt32 acquireParams, ViInt32 reserved);



//! Helper function to ease the instrument configuration.
/*! Returns maximum nominal number of samples which fits into the available memory.

    Values returned by the function:
 
    'nomSamples'        = maximum number of data samples available
    
    NOTE: When using this method, make sure to use '$prefixD1$_configHorizontal' and 
            '$prefixD1$_configMemory' beforehand to set the sampling rate and the number of
            segments to the desired values ('nbrSamples' in '$prefixD1$_configMemory' may be 
            any number!). '$prefixD1$_bestNominalSamples' depends on these variables.
 
    Returns one of the following ViStatus values:
    VI_SUCCESS                            when a good solution has been found. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_bestNominalSamples(ViSession instrumentID, ViInt32* nomSamples);



//! Helper function to ease the instrument configuration.
/*! Returns the best possible sampling rate for an acquisition which covers the 'timeWindow'
    with no more than 'maxSamples', taking into account the current state of the instrument,
    in particular the requested channel combination and the number of segments.
    In addition, this routine returns the 'real' nominal number of samples which can
    be accommodated (it is computed as timeWindow/sampInterval !).
 
    The following values must be supplied to the function:
 
    'maxSamples'       = maximum number of samples to be used
    'timeWindow'       = time window in seconds to be covered
 
    Values returned by the function:
 
    'sampInterval'     = recommended sampling interval in seconds
    'nomSamples'       = recommended number of data samples
 
    NOTE: This function DOES NOT modify the state of the digitizer at all. It simply returns
            a recommendation that the user is free to override.
    NOTE: When using this method, make sure to use '$prefixD1$_configMemory' beforehand to set 
            the number of segments to the desired value. ('nbrSamples' may be any 
            number!). '$prefixD1$_bestSampInterval' depends on this variable.
    NOTE: The returned 'recommended' values for the 'sampInterval' and the nominal number
            of samples 'nomSamples' are expected to be used for configuring the instrument
            with calls to '$prefixD1$_configMemory' and '$prefixD1$_configHorizontal'. Make sure
            to use the same number of segments in this second call to '$prefixD1$_configMemory'
            as in the first one.
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_SETUP_NOT_AVAILABLE when the available memory is too short, and the longest
                                      available sampling interval too short. The returned 
                                      sampling interval is the longest one possible.
    VI_SUCCESS                        when a good solution has been found. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_bestSampInterval(ViSession instrumentID, ViInt32 maxSamples,
    ViReal64 timeWindow, ViReal64* sampInterval, ViInt32* nomSamples);



//! Configures Averagers and Analyzers.
/*! Configures parameter in the channel dependent averager/analyzer configuration 'channel'
    'channel'        = 1...Nchan
                     = 0 (selecting channel 1) is supported for backwards compatibility
 
    'parameterString'= character string specifying the requested parameter
                       Please refer to the manual for the accepted parameter strings
    'value'          = value to set
                       The type of the value depends on 'parameterString'
                       Please refer to the manual for the required  data type as a
                       function of the accepted parameters.
                       NOTE to C/C++ programmers: 'ViAddr' resolves to 'void*'
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED        if this function is not supported by the instrument.
    ACQIRIS_ERROR_PARAM_STRING_INVALID if 'parameterString' is invalid.
    ACQIRIS_WARN_SETUP_ADAPTED         if 'value' has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configAvgConfig(ViSession instrumentID, 
    ViInt32 channel, ViConstString parameterString, ViAddr value);



//! Configures Averagers and Analyzers.
/*! Configures a parameter in the channel dependent averager/analyzer configuration 'channel'.
    This function should be used for 'ViInt32' typed parameters.
    'channel'        = 1...Nchan
                     = 0 (selecting channel 1) is supported for backwards compatibility
 
    'parameterString'= character string specifying the requested parameter.
                       Please refer to the manual for the accepted parameter strings.
                       Use this function ONLY for a 'parameterString' which sets a
                       ViInt32 type parameter!
    'value'          = value to set
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED        if this function is not supported by the instrument.
    ACQIRIS_ERROR_PARAM_STRING_INVALID if 'parameterString' is invalid.
    ACQIRIS_WARN_SETUP_ADAPTED         if 'value' has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configAvgConfigInt32(ViSession instrumentID, 
    ViInt32 channel, ViConstString parameterString, ViInt32 value);



//! Configures Averagers and Analyzers.
/*! Configures parameter in the channel dependent averager/analyzer configuration 'channel'
    This function should be used for 'ViReal64' typed parameters.
    'channel'        = 1...Nchan
                     = 0 (selecting channel 1) is supported for backwards compatibility
 
    'parameterString'= character string specifying the requested parameter.
                       Please refer to the manual for the accepted parameter strings.
                       Use this function ONLY for a 'parameterString' which sets a
                       ViReal64 type parameter!
    'value'          = value to set
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED        if this function is not supported by the instrument.
    ACQIRIS_ERROR_PARAM_STRING_INVALID if 'parameterString' is invalid.
    ACQIRIS_WARN_SETUP_ADAPTED         if 'value' has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configAvgConfigReal64(ViSession instrumentID, 
    ViInt32 channel, ViConstString parameterString, ViReal64 value);


//! Configures combined operation of multiple channels.
/*! 'nbrConvertersPerChannel'    = 1        all channels use 1 converter each (default)
                                 = 2        half of the channels use 2 converters each
                                 = 4        1/4  of the channels use 4 converters each
 
    'usedChannels'        bit-field indicating which channels are used (LSB = channel 1)
    The acceptable values for 'usedChannels' depend on 'nbrConvertersPerChannel' 
    and on the number of available channels in the digitizer:
    A) If 'nbrConvertersPerChannel' = 1, 'usedChannels' must reflect the fact that
    ALL channels are available for use. It accepts a single value for a given digitizer:
    'usedChannels'        = 0x00000001    if the digitizer has 1 channel
                          = 0x00000003    if the digitizer has 2 channels
                          = 0x0000000f    if the digitizer has 4 channels
    B) If 'nbrConvertersPerChannel' = 2, 'usedChannels' must reflect the fact that
    only half of the channels may be used:
    'usedChannels'        = 0x00000001    use channel 1 on a 2-channel digitizer
                            0x00000002    use channel 2 on a 2-channel digitizer
                            0x00000003    use channels 1+2 on a 4-channel digitizer
                            0x00000005    use channels 1+3 on a 4-channel digitizer
                            0x00000009    use channels 1+4 on a 4-channel digitizer
                            0x00000006    use channels 2+3 on a 4-channel digitizer
                            0x0000000a    use channels 2+4 on a 4-channel digitizer
                            0x0000000c    use channels 3+4 on a 4-channel digitizer
    C) If 'nbrConvertersPerChannel' = 4, 'usedChannels' must reflect the fact that
    only 1 of the channels may be used:
    'usedChannels'        = 0x00000001    use channel 1 on a 4-channel digitizer
                            0x00000002    use channel 2 on a 4-channel digitizer
                            0x00000004    use channel 3 on a 4-channel digitizer
                            0x00000008    use channel 4 on a 4-channel digitizer
    NOTE: Digitizers which don't support channel combination, always use the default
          'nbrConvertersPerChannel' = 1, and the single possible value of 'usedChannels'
    NOTE: If digitizers are combined with ASBus, the channel combination applies equally to
          all participating digitizers.    
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configChannelCombination(ViSession instrumentID, 
    ViInt32 nbrConvertersPerChannel, ViInt32 usedChannels);



//! Configures Control-IO connectors.
/*! Typically, only a few (or no) IO connectors are present on a single digitizer
   
    'connector'        = 1        Front Panel I/O A (MMCX connector)
                       = 2        Front Panel I/O B (MMCX connector)
                       = 3        Front Panel I/O C (MMCX connector, if available)
                       = 9        Front Panel Trigger Out (MMCX connector)
                       = 11       PXI Bus 10 MHz (if available)
                       = 12       PXI Bus Star Trigger (if available)
    'signal'           = value depends on 'connector', refer to manual for definitions.
    'qualifier1',      = value depends on 'connector', refer to manual for definitions.
    'qualifier2'       = If trigger veto functionality is available (if available), 
                         accepts values between 30 ns and 1.0 sec. 
                         The trigger veto values given will be rounded off to steps of 33 ns. 
                         A value of 0.0 means that no holdoff is required and no Prepare for 
                         Trigger signal will be needed.
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configControlIO(ViSession instrumentID, ViInt32 connector,
    ViInt32 signal, ViInt32 qualifier1, ViReal64 qualifier2);



//! Configures the external clock of the digitizer.
/*! 'clockType'        = 0        Internal Clock (default at start-up)
                       = 1        External Clock (continuous operation)
                       = 2        External Reference (10 MHz)
                       = 4        External Clock (start/stop operation)
    'inputThreshold'   = input threshold for external clock or reference in mV
    'delayNbrSamples'  = number of samples to acquire after trigger (for 'clockType' = 1 ONLY!)
    'inputFrequency'   = frequency, in Hz, of applied clock input signal
    'sampFrequency'    = frequency, in Hz, of requested sampling
 
    NOTE: When 'clockType' is set to 1 or 4, the values 'sampInterval' and 'delayTime' in the 
          function '$prefixD1$_configHorizontal' are ignored.
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configExtClock(ViSession instrumentID, ViInt32 clockType,
    ViReal64 inputThreshold, ViInt32 delayNbrSamples, 
    ViReal64 inputFrequency, ViReal64 sampFrequency);


//! Configures the frequency counter.
/*! 'signalChannel'    = 1...Nchan for signal input channel
    'typeMes'          = 0        Frequency
                       = 1        Period
                       = 2        Totalize by Time, counts input pulses during interval defined by 
                                  'apertureTime'
                       = 3        Totalize by Gate, counts input pulses during interval defined by 
                                  I/O A or B input
    'targetValue'      = estimate of expected result (set to 0.0, if no estimate available)
    'apertureTime'     = minimum measurement time for Frequency and Period modes
                       = time gate for Totalize by Time mode
    'reserved', 'flags' currently unused (set to zero!)
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configFCounter(ViSession instrumentID, ViInt32 signalChannel, 
         ViInt32 typeMes, ViReal64 targetValue, ViReal64 apertureTime, ViReal64 reserved, 
        ViInt32 flags);


//! Configures the horizontal control parameters of the digitizer.
/*! 'sampInterval'    = sampling interval in seconds
    'delayTime'       = trigger delay time in seconds, with respect to the 
                        beginning of the record. 
                        A positive number corresponds to trigger BEFORE the beginning 
                        of the record (post-trigger recording).
                        A negative number corresponds to pre-trigger recording. It
                        cannot be smaller than (- sampInterval * nbrSamples), which
                        corresponds to 100% pre-trigger.
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configHorizontal(ViSession instrumentID, 
    ViReal64 sampInterval, ViReal64 delayTime);



//! Configures the memory control parameters of the digitizer.
/*! 'nbrSamples'        = nominal number of samples to record (per segment!)
    'nbrSegments'       = number of segments to acquire
                          1 corresponds to the normal single-trace acquisition mode.
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configMemory(ViSession instrumentID, ViInt32 nbrSamples, 
    ViInt32 nbrSegments);



//! Configures the memory control parameters of the digitizer.
/*! 'nbrSamplesHi'      = reserved for future use, must be set to 0.
    'nbrSamplesLo'      = nominal number of samples to record (per segment!).
    'nbrSegments'       = number of segments to acquire per bank
                          1 corresponds to the normal single-trace acquisition mode.
    'nbrBanks'          = number of banks in which the memory will be split, 
                          for buffered reading (SAR).
                          1 corresponds to the normal acquisition mode.
    'flags'             = 0 no flags. 
                        = 1 force use of internal memory (for digitizers with extended 
                          memory options only).
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configMemoryEx(ViSession instrumentID, ViUInt32 nbrSamplesHi, 
    ViUInt32 nbrSamplesLo, ViInt32 nbrSegments, ViInt32 nbrBanks, ViInt32 flags);



//! Configures the operational mode of the digitizer.
/*! 'mode'            = operational mode
                      = 0        normal acquisition
                      = 1        stream data to DPU (only in SC2x0/AC2x0 Data Streamers)
                      = 2        averaging mode (only in APxxx with the Avgr option, U1084A)
                      = 3        dual-memory mode (only in AP101, AP201)
                      = 5        PeakTDC mode (only for PeakTDC Analyzers like AP240, U1084A)
                      = 6        frequency counter mode
                      = 7        SSR mode (only for APxxx and U1084A with the SSR option)
                      = 12       Down Converter mode (only for M9202A with DDC option)
                      = 13       Base Design mode (only U1084A with CFW option)
                      = 14       Custom firmware mode (only U1084A with CFW option)

    'modifier' = not used, set to 0

    'flags'    ['mode' = 0]    = 0        normal
                               = 1        'Start-on-Trigger' mode (if available)
                               = 2        'Sequence-wrap' mode (use 'nbrSegments' > 1) 
                               = 10       'SAR' mode (use 'nbrBanks' > 1)
               ['mode' = 2]    = 0        normal
                               = 10       'SAR' mode (only for U1084A Avg; use 'nbrBanks' = 2)
               ['mode' = 1]    = 0        normal
                               = 1        autoRestart
               ['mode' = 3]    = 0        acquire into memory bank 0
                               = 1        acquire into memory bank 1
               ['mode' = 7]    = 0        normal (only for APxxx)
                               = 10       'SAR' mode (only for U1084A with Zero Suppress)
               otherwise  unused, set to 0

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_FILE_NOT_FOUND       if an FPGA file could not be found.
    ACQIRIS_ERROR_NO_DATA              if an FPGA file did not contain expected data.
    ACQIRIS_ERROR_FPGA_x_LOAD          if an FPGA could not be loaded, 
                                       where 'x' = FPGA nbr. 
    ACQIRIS_ERROR_INSTRUMENT_RUNNING   if the instrument is currently running.
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configMode(ViSession instrumentID, ViInt32 mode, 
    ViInt32 modifier, ViInt32 flags);



//! Configures the multi-module synchronization
/*! This function defines a multi-module synchronization. Upon success, all slaves modules
 *  will be synchronized with the master
 *
 * \param instrumentID Module to be used as master
 * \param nbrSlaves Number of modules to be used as slave
 * \param slaves Array of modules to be used as slave
 */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configModuleSync(ViSession instrumentID,
        ViInt32 nbrSlaves, ViSession slaves[]);



//! Configures the input multiplexer on a channel
/*! 'input'            = 0        set to input connection A
                       = 1        set to input connection B

    NOTE: This function is only of use for instruments with an input multiplexer (i.e. more
            than 1 input per channel, e.g. DP211). On the "normal" instruments with a single
            input per channel, this function may be ignored.
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configMultiInput(ViSession instrumentID, ViInt32 channel, 
    ViInt32 input);



//! Configures an array of setup data (typically for on-board processing)
/*! 'setupType'        = indicates the object type (data structure) of which the setup data is 
                         composed. Some objects might be simple elements, e.g. ViInt32
 
    'nbrSetupObj'      = number of configuration objects contained in configSetupData
 
    'setupData'        = pointer to the setup data array
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED        if this function is not supported by the instrument.
    ACQIRIS_ERROR_BUFFER_OVERFLOW      if 'nbrSetupObj' exceeds the maximum allowed value.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configSetupArray(ViSession instrumentID, ViInt32 channel,
     ViInt32 setupType, ViInt32 nbrSetupObj, void* setupData);



//! Configures the trigger class control parameters of the digitizer.
/*! 'trigClass'          = 0             edge trigger
                         = 1             TV trigger (12-bit-FAMILY External only)
                         = 3             OR (if available)
                         = 4             NOR (if available)
                         = 5             AND (if available)
                         = 6             NAND (if available)
    'sourcePattern'      = 0x000n0001    Enable Channel 1
                         = 0x000n0002    Enable Channel 2
                         = 0x000n0004    Enable Channel 3
                         = 0x000n0008    Enable Channel 4    etc.
                         = 0x800n0000    Enable External Trigger 1
                         = 0x400n0000    Enable External Trigger 2 (if available) etc.
                           where n is 0 for single instruments, or the module number for
                           MultiInstruments (ASBus operation). When 'trigClass' = 3,4,5 or 6,
                           the 'sourcePattern' can be a combination of different sources.
                           See manual for a detailed description of 'sourcePattern'.
    'validatePattern'      Unused, set to 0.
    'holdoffType'        = 0             Holdoff by time (if available)
    'holdoffTime'        Holdoff time, in units of seconds.
    'reserved'           Unused, set to 0.0.
 
    Note: The detailed TV trigger configuration is set with the function '$prefixD1$_configTrigTV'
    Note2: trigClass = 3,4,5 or 6 features are only supported within a single instrument, or
      within a single module in an AS Bus configuration (if available).
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configTrigClass(ViSession instrumentID, ViInt32 trigClass, 
     ViInt32 sourcePattern, ViInt32 validatePattern, ViInt32 holdType, ViReal64 holdoffTime,
     ViReal64 reserved);  



//! Configures the trigger source control parameters for a specified channel in the digitizer.
/*! 'channel'        = 1... (Number of IntTrigSources) for internal trigger sources
                     = -1..-(Number of ExtTrigSources) for external trigger sources
    'trigCoupling'   = 0        DC
                     = 1        AC
                     = 2        HFreject (if available)
                     = 3        DC, 50 Ohms (ext. trigger only, if available)
                     = 4        AC, 50 Ohms (ext. trigger only, if available)
    'trigSlope'      = 0        Positive
                     = 1        Negative
                     = 2        Window, transition out of window
                     = 3        Window, transition into window
                     = 4        HFdivide (by factor 4)
                     = 5        SpikeStretcher (if available)
    'trigLevel1'    (internal)    in % of Vertical Full Scale of the channel settings
                    (external)    in mV
    'trigLevel2'    (internal)    in % of Vertical Full Scale of the channel settings
                    (external)    in mV
                                  'trigLevel2' is only used when Window Trigger is selected
    NOTE: Some of the possible states may be unavailable in some digitizers.
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configTrigSource(ViSession instrumentID, ViInt32 channel, 
     ViInt32 trigCoupling, ViInt32 trigSlope, ViReal64 trigLevel1, ViReal64 trigLevel2);      



//! Configures the TV trigger control parameters for a specified channel in the digitizer.
/*! 'channel'        = -1..-(Number of ExtTrigSources) for external trigger sources
                         NOTE: the TV trigger option is only available on the External Trigger input
    'standard'       = 0        625 lines per frame / 50 Hz
                     = 2        525 lines per frame / 60 Hz
    'field'          = 1        field 1 (odd)
                     = 2        field 2 (even)
    'line'           = line number, depends on the 'standard' and 'field' selection:
                         1 to 263    for 'standard' = 525/60Hz and 'field' = 1
                         1 to 262    for 'standard' = 525/60Hz and 'field' = 2
                         1 to 313    for 'standard' = 625/50Hz and 'field' = 1
                       314 to 625    for 'standard' = 625/50Hz and 'field' = 2
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    ACQIRIS_ERROR_NOT_SUPPORTED        if this 'channel' does not support the TV trigger.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configTrigTV(ViSession instrumentID, ViInt32 channel, 
     ViInt32 standard, ViInt32 field, ViInt32 line);      



//! Configures the vertical control parameters for a specified channel in the digitizer.
/*! 'channel'        = 1...Nchan
                     = -1  for Ext. Trigger Input of digitizers with programmable Trigger Full Scale
    'fullScale'      = in Volts
    'offset'         = in Volts
    'coupling'       = 0        Ground (Averagers ONLY)
                     = 1        DC, 1 MOhm
                     = 2        AC, 1 MOhm
                     = 3        DC,    50 Ohms
                     = 4        AC, 50 Ohms
    'bandwidth'      = 0        no bandwidth limit (default)
                     = 1        bandwidth limit =  25 MHz
                     = 2        bandwidth limit = 700 MHz
                     = 3        bandwidth limit = 200 MHz
                     = 4        bandwidth limit =  20 MHz
                     = 5        bandwidth limit =  35 MHz
                     = 6        bandwidth limit = 400 MHz
                     = 7        bandwidth limit = 600 MHz
    NOTE: Not all bandwidth limits are available on a single instrument. In some, there is no
            bandwidth limiting capability at all. In this case, use 'bandwidth' = 0.
 
    Returns one of the following ViStatus values:
    ACQIRIS_WARN_SETUP_ADAPTED         if one of the parameters has been adapted.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_configVertical(ViSession instrumentID, ViInt32 channel, 
     ViReal64 fullScale, ViReal64 offset, ViInt32 coupling, ViInt32 bandwidth);



//! Translates an error code into a human readable form 
/*! 'errorCode'        = Error code (returned by a function) to be translated
    'errorMessage'     = Pointer to user-allocated character string (minimum size 256),
                         into which the error-message text is returned
 
    NOTE: 'instrumentID' can be VI_NULL.
 
    Returns one of the following ViStatus values:
    VI_SUCCESS always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_errorMessage(ViSession instrumentID, ViStatus errorCode,
    ViChar errorMessage[]);



//! Translates an error code into a human readable form. 
/*! For file errors, the returned message will also contain the file name and the 
    original 'ansi' error string.
 
    'errorCode'          = Error code (returned by a function) to be translated
    'errorMessage'       = Pointer to user-allocated character string (suggested size 512 bytes),
                           into which the error message text will be copied.
    'errorMessageSize'   = size of 'errorMessage', in bytes.
 
    NOTE: 'instrumentID' can be VI_NULL.
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_BUFFER_OVERFLOW    if 'errorMessageSize' is too small.
    VI_SUCCESS                       otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_errorMessageEx(ViSession instrumentID, ViStatus errorCode, 
    ViChar errorMessage[], ViInt32 errorMessageSize);



//! Forces a 'manual' trigger. 
/*! The function returns immediately after initiating
    a trigger. One must therefore wait until this acquisition has terminated
    before reading the data, by checking the status with the '$prefixD1$_acqDone'
    function. Equivalent to $prefixD1$_forceTrigEx with 'forceTrigType' = 0 */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_forceTrig(ViSession instrumentID);



//! Forces a 'manual' trigger. 
/*! The function returns immediately after initiating
    a trigger. One must therefore wait until this acquisition has terminated
    before reading the data, by checking the status with the '$prefixD1$_acqDone'
    or the '$prefixD1$_waitForEndOfAcquisition' functions.
 
    'forceTrigType'=       0    Sends a software trigger to end the (entire) acquisition. 
                             In multisegment mode, the current segment is acquired, the acquisition 
                             is terminated and the data and timestamps of subsequent segments are 
                             invalid. The 'trigOut' Control IO will NOT generate a trigger output.
                             Equivalent to '$prefixD1$_forceTrig'.
                           1    Send a software trigger similar to a hardware trigger. 
                             In multisegment mode, the acquisition advances to the next segment and 
                             then waits again for a trigger. If no valid triggers are provided to 
                             the device, the application must call '$prefixD1$_forceTrigEx' as many 
                             times as there are segments. In this mode, 'trigOut' Control IO will    
                             generate a trigger output on each successful call. Every acquired 
                             segment will be valid. This mode is only supported for single 
                             (i.e. non-ASBus-connected) instruments.
    'modifier'               currently unused, must be zero
    'flags'                  currently unused, must be zero 
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_INSTRUMENT_STOPPED          if the instrument is already stopped. 
    ACQIRIS_ERROR_PRETRIGGER_STILL_RUNNING    if the requested data before trigger is being 
                                              acquired.
    ACQIRIS_ERROR_NOT_SUPPORTED               if this function is not supported by the current 
                                              mode (e.g. mode Average on APxxx).
    VI_SUCCESS                                otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_forceTrigEx(ViSession instrumentID, ViInt32 forceTrigType,
    ViInt32 modifier, ViInt32 flags);



//! Free current bank during SAR acquisitions. 
/*! Calling this function indicates to the driver that
    the current SAR bank has been read and can be reused for a new acquisition. 
    
    'reserved'         Unused, must be set to 0.
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NO_DATA                if there is no data available, no bank to unlock. 
    ACQIRIS_ERROR_SETUP_NOT_AVAILABLE    if the SAR mode is not available, or not activated.
    ACQIRIS_ERROR_INSTRUMENT_STOPPED     was not started using 'acquire' beforehand, or was stopped.
    VI_SUCCESS                           otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_freeBank(ViSession instrumentID, ViInt32 reserved);



//! Return the acquisition status of the specified instrument. 
/*! 
    acqStatus                    =    Current status of the acquisition. See 'AqAcqStatus' declaration.
    reserved1                    =    Unused, but must be a valid pointer.
    reserved2                    =    Unused, but must be a valid pointer.

    The normal sequence of values for 'AqAcqStatus' is shown below (Digitizer State Machine):

    Initial state    =>AqAcqDone (AqAcqStopped) =>AqAcqPretrigRun=>AqAcqArmed=>AqAcqAcquiring=>    AqAcqDone
                    (init)                       (acquire)

    If 'stopAcquisition' is called, the state machine goes directly to 'AqAcqStopped' at any time 
    during the acquisition. 'AqAcqStopped' is similar to 'AqAcqDone', the state machine will go to 
    'AqAcqPretrigRun' if an acquisition is started.

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED if this function is not supported by the current mode (e.g. Analyser mode on APxxx).
    ACQIRIS_ERROR_IO_READ       if a link error has been detected (e.g. PCI link lost).
    VI_SUCCESS                           otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getAcqStatus(ViSession instrumentID, ViInt32* acqStatus, ViBoolean* reserved1, ViInt32* reserved2);



//! Returns a parameter from the averager/analyzer configuration
/*! See remarks under '$prefixD1$_configAvgConfig'

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_PARAM_STRING_INVALID if 'parameterString' is invalid.
    ACQIRIS_ERROR_NOT_SUPPORTED        if this function is not supported by the  
                                       instrument.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getAvgConfig(ViSession instrumentID, 
    ViInt32 channel, ViConstString parameterString, ViAddr value);



//! Returns a parameter from the averager/analyzer configuration
/*! See remarks under '$prefixD1$_configAvgConfig'

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_PARAM_STRING_INVALID if 'parameterString' is invalid.
    ACQIRIS_ERROR_NOT_SUPPORTED        if this function is not supported by the  
                                       instrument.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getAvgConfigInt32(ViSession instrumentID, 
    ViInt32 channel, ViConstString parameterString, ViInt32 *valueP);



//! Returns a parameter from the averager/analyzer configuration
/*! See remarks under '$prefixD1$_configAvgConfig'

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_PARAM_STRING_INVALID if 'parameterString' is invalid.
    ACQIRIS_ERROR_NOT_SUPPORTED        if this function is not supported by the  
                                       instrument.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getAvgConfigReal64(ViSession instrumentID, 
    ViInt32 channel, ViConstString parameterString, ViReal64 *valueP);



//! Returns parameters of combined operation of multiple channels
/*! See remarks under '$prefixD1$_configChannelCombination'
 
    Returns one of the following ViStatus values:
    VI_SUCCESS                         always.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getChannelCombination(ViSession instrumentID, 
    ViInt32* nbrConvertersPerChannel, ViInt32* usedChannels);



//! Returns the state of Control-IO connectors
/*! See remarks under '$prefixD1$_configControlIO'

    SPECIAL CASE: If 'connector' = 0 is specified, the returned value of 'signal'
    is the bit-coded list of the 'connectors' which are available in the digitizer.
    E.g. If the connectors 1 (I/O A) and 9 (TrigOut) are present, the bits 1 and 9 of
    'signal' are set, where bit 0 is the LSB and 31 is the MSB. 
    Thus, 'signal' would be equal to 0x202.
 
    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getControlIO(ViSession instrumentID, ViInt32 connector,
    ViInt32* signal, ViInt32* qualifier1, ViReal64* qualifier2);



//! Returns the (external) clock parameters of the digitizer
/*! See remarks under '$prefixD1$_configExtClock'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getExtClock(ViSession instrumentID, ViInt32* clockType,
    ViReal64* inputThreshold, ViInt32* delayNbrSamples,
    ViReal64* inputFrequency, ViReal64* sampFrequency);



//! Returns the current settings of the frequency counter
/*! See remarks under '$prefixD1$_configFCounter'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getFCounter(ViSession instrumentID, ViInt32* signalChannel, 
         ViInt32* typeMes, ViReal64* targetValue, ViReal64* apertureTime, ViReal64* reserved, 
        ViInt32* flags);



//! Returns the current horizontal control parameters of the digitizer.
/*! See remarks under '$prefixD1$_configHorizontal'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getHorizontal(ViSession instrumentID, 
    ViReal64* sampInterval, ViReal64* delayTime);



//! Returns the current memory control parameters of the digitizer.
/*! See remarks under '$prefixD1$_configMemory'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getMemory(ViSession instrumentID, 
    ViInt32* nbrSamples, ViInt32* nbrSegments);
                   


//! Returns the current memory control parameters of the digitizer.
/*! See remarks under '$prefixD1$_configMemoryEx'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getMemoryEx(ViSession instrumentID, 
    ViUInt32* nbrSamplesHi, ViUInt32* nbrSamplesLo, ViInt32* nbrSegments, ViInt32* nbrBanks, 
    ViInt32* flags);



//! Returns the current operational mode of the digitizer.
/*! See remarks under '$prefixD1$_configMode'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getMode(ViSession instrumentID, ViInt32* mode,
    ViInt32* modifier, ViInt32* flags);



//! Retreives the multi-module synchronization
/*! This function retrieves a multi-module synchronization. Upon success, slavesArray will
 *  contain the slaves modules currently synchronized with this master.
 *
 * \param instrumentID Module to be used as master
 * \param slavesArraySize Number of slaves that can be written to slavesArray
 * \param slavesArray Array that will be filled with slave modules
 * \param ActualSlaves Actual number of slave modules returned
 */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getModuleSync(ViSession instrumentID, ViInt32 slavesArraySize,
        ViSession slavesArray[], ViInt32 *ActualSlaves);


//! Returns the multiple input configuration on a channel
/*! See remarks under '$prefixD1$_configMultiInput'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getMultiInput(ViSession instrumentID, ViInt32 channel, 
    ViInt32* input);



//! Returns setup data array (typically used for on-board processing).
/*! See remarks under '$prefixD1$_configSetupArray'

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED        if this function is not supported by the instrument.
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getSetupArray(ViSession instrumentID, ViInt32 channel,
     ViInt32 setupType, ViInt32 nbrSetupObj, void* setupData, ViInt32* nbrSetupObjReturned);



//! Returns the current trigger class control parameters of the digitizer.
/*! See remarks under '$prefixD1$_configTrigClass'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getTrigClass(ViSession instrumentID, ViInt32* trigClass, 
     ViInt32* sourcePattern, ViInt32* validatePattern, ViInt32* holdType, ViReal64* holdValue1,
     ViReal64* holdValue2);



//! Returns the current trigger source control parameters for a specified channel.
/*! See remarks under '$prefixD1$_configTrigSource'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getTrigSource(ViSession instrumentID, ViInt32 channel, 
     ViInt32* trigCoupling, ViInt32* trigSlope, ViReal64* trigLevel1, ViReal64* trigLevel2);



//! Returns the current TV trigger control parameters of the digitizer.
/*! See remarks under '$prefixD1$_configTrigTV'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getTrigTV(ViSession instrumentID, ViInt32 channel, 
     ViInt32* standard, ViInt32* field, ViInt32* line);      



//! Returns the current vertical control parameters for a specified channel.
/*! See remarks under '$prefixD1$_configVertical'

    Returns one of the following ViStatus values:
    VI_SUCCESS                         always.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_getVertical(ViSession instrumentID, ViInt32 channel, 
     ViReal64* fullScale, ViReal64* offset, ViInt32* coupling, ViInt32* bandwidth);



//! Automatically combines as many digitizers as possible to "MultiInstrument"s.
/*! Digitizers are only combined if they are physically connected via ASBus.
    This call must be followed by 'nbrInstruments' calls to '$prefixD1$_init' or 
    '$prefixD1$_InitWithOptions' to retrieve the 'instrumentID's of the (multi)digitizers.
 
    The following value must be supplied to the function:
 
    'optionsString'    = an ASCII string which specifies options. 
                         Currently, no options are supported
 
    Values returned by the function:
 
    'nbrInstruments'   = number of user-accessible instruments. This number includes 
                         also single instruments that don't participate on the ASBus.
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_EEPROM_DATA_INVALID  if one of the instruments being initialized has invalid 
                                       data in EEPROM.
    ACQIRIS_ERROR_UNSUPPORTED_DEVICE   if one of the instruments being initialized is not 
                                       supported by this driver version.
    VI_SUCCESS                         otherwise.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_multiInstrAutoDefine(ViConstString optionsString, 
    ViInt32* nbrInstruments);



//! 'Manually' combines a number of digitizers into a single "MultiInstrument"
/*! The following values must be supplied to the function:

    'instrumentList'    = array of 'instrumentID' of already initialized single digitizers
    'nbrInstruments'    = number of digitizers in the 'instrumentList'
    'masterID'          = 'instrumentID' of master digitizer
 
    Values returned by the function:
 
    'instrumentID'    = identifier of the new "MultiInstrument", for subsequent use 
                        in the other function calls.
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED        if this function is not supported by the instrument(s).
    ACQIRIS_ERROR_NOT_ENOUGH_DEVICES   if 'nbrInstruments' is < 2.
    ACQIRIS_ERROR_TOO_MANY_DEVICES     if 'nbrInstruments' exceeds the maximum number 
                                       of AS Bus instruments.
    ACQIRIS_ERROR_INSTRUMENT_NOT_FOUND if one of the 'instrumentList[]' entries is invalid.
    ACQIRIS_ERROR_NO_MASTER_DEVICE     if 'masterID' is invalid.
    ACQIRIS_ERROR_SETUP_NOT_AVAILABLE  if one of the 'instrumentList[]' entries is not AS Bus 
                                       compatible.
    ACQIRIS_ERROR_UNSUPPORTED_DEVICE   if one of the 'instrumentList[]' entries is not supported 
                                       by this driver version.
    ACQIRIS_ERROR_INTERNAL_DEVICENO_INVALID if one of the 'instrumentList[]' entries is duplicated. 
    VI_SUCCESS                         otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_multiInstrDefine(ViSession instrumentList[], 
    ViInt32 nbrInstruments, ViSession masterID, ViSession* instrumentID);



//! Undefines all "MultiInstruments".
/*! The following value must be supplied to the function:

    'optionsString'    = an ASCII string which specifies options. 
                         Currently, no options are supported
 
    Please refer to the User's manual for a detailed description of the steps required
    to reestablish the identifiers of the existing individual digitizers, in order
    to continue using them.
 
    Returns one of the following ViStatus values:
    VI_SUCCESS always.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_multiInstrUndefineAll(ViConstString optionsString);



//! Starts data processing on acquired data (only in instruments with on-board data processing)
/*! The following values must be supplied to the function:

    'processType'    =    0        for no processing
                          1        for extrema mode 
                          2        for hysteresis mode 
                          3        for interpolated extrema mode
                          4        for interpolated hysteresis mode
 
       defines how switching for the dual-bank memory is done 
 
    'flags'          =    0        switching is done by software
                          1        switching is automatic, auto switch turned on
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED      if this function is not supported by the instrument.
    VI_SUCCESS                       otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_processData(ViSession instrumentID, ViInt32 processType, 
            ViInt32 flags);



//! Checks if the on-board processing has terminated.
/*! Returns 'done' = VI_TRUE if the processing is terminated

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED      if this function is not supported by the instrument.
    ACQIRIS_ERROR_IO_READ            if a link error has been detected (e.g. PCI link lost).
    VI_SUCCESS                       otherwise.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_procDone(ViSession instrumentID, ViBoolean* done);



//! Returns a waveform and associated descriptors, in one of several possible formats
/*! This function is the preferred way of reading waveforms, since it includes the capabilities
    of all the other read functions, and more!
 
    The following values must be supplied to the function:
 
    'channel'        = 1...Nchan (as returned by '$prefixD1$_getNbrChannels' )
    'readParP'       = pointer to a user-supplied structure that specifies what and how to read
 
    Values returned by the function:
 
    'dataArrayP'     = user-allocated data destination array of type defined by 'readPar.dataType'
                       When reading a single segment of raw data, its size MUST be at least
                       (nbrSamples + 32), for reasons of data alignment. Please refer to the manual
                       for additional details.
    'dataDescP'      = user-allocated structure for returned data descriptor values
    'segDescArrayP'  = user allocated array of structures for returned segment descriptor values
                       This array must contain at least 'readPar.nbrSegments' elements of the
                       appropriate type (typically 'AqSegmentDescriptor' or 'AqSegmentDescriptorAvg')
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_INSTRUMENT_RUNNING       if the instrument is running.
    ACQIRIS_ERROR_READMODE                 if 'readParP->readMode' is not valid.
    ACQIRIS_ERROR_NOT_SUPPORTED            if 'readParP->readMode' is not supported by the instrument.
    ACQIRIS_ERROR_DATATYPE                 if 'readParP->dataType' is not valid for the 
                                           chosen 'readParP->readMode' or for this instrument.
    ACQIRIS_ERROR_FIRST_SEG                if 'readParP->firstSegment' is invalid.
    ACQIRIS_ERROR_NBR_SEG                  if 'readParP->nbrSegments' is invalid.
    ACQIRIS_ERROR_DATA_ARRAY or
    ACQIRIS_ERROR_NBR_SAMPLE               if 'readParP->dataArraySize' is invalid.
    ACQIRIS_ERROR_SEG_DESC_ARRAY           if 'readParP->segDescArraySize' is invalid.
    ACQIRIS_ERROR_SEG_OFF                  if 'readParP->segmentOffset' is invalid.
    ACQIRIS_ERROR_NBR_SEG                  if 'readParP->nbrSegments' is invalid.
    ACQIRIS_ERROR_BUFFER_OVERFLOW          if 'readParP->dataArraySize' is too small.
    ACQIRIS_ERROR_NO_DATA                  if nothing was acquired beforehand.
    ACQIRIS_ERROR_CANNOT_READ_THIS_CHANNEL if the requested channel is not available.
    ACQIRIS_ERROR_READ_TIMEOUT             if the reading encountered a problem.
    ACQIRIS_WARN_READPARA_NBRSEG_ADAPTED   if 'readParP->nbrSegments' has been adapted.
    ACQIRIS_WARN_ACTUAL_DATASIZE_ADAPTED   if 'readParP->dataArraySize' has been adapted.
    ACQIRIS_WARN_READPARA_FLAGS_ADAPTED    if 'readParP->flags' has been adapted.
    VI_SUCCESS                             otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_readData(ViSession instrumentID, ViInt32 channel,
    AqReadParameters* readParP, void* dataArrayP, AqDataDescriptor* dataDescP,
    void* segDescArrayP);


//! Reads data in streamed acquisitions. This function blocks until the buffer is full.
/*! 'channel'           = the digitizer channel to read.
    'stream'            = the data stream on the module to read:
                          0 = default stream (acquisition data)
    'flags'             = readout flags. Currently only 0 is valid.
    'bufferP'           = pointer to a user-allocated memory buffer to receive the data.
    'bufferSize'        = the number of bytes available in 'bufferP' for receiving data.
    'blockDescP'        = pointer to a user-allocated AqStreamBlockDescriptor structure which will be filled with
                          information about this block of streaming data. The 'descriptorVersion' of the structure
                          must be set to AqStreamBlockDescriptorVersionCurrent.
                          May be NULL, in which case no additional information will be returned.
 
    Common return values:
    ACQIRIS_ERROR_NOT_SUPPORTED     if this function is not supported by the instrument.
    ACQIRIS_ERROR_NO_DATA           if the instrument has not first acquired data in 'streaming' mode (modeFlag = 14).
    VI_ERROR_PARAMETERn             with n between 1 and 7, if the value given for parameter n is invalid.
    ACQIRIS_ERROR_STRUCT_VERSION    if the descriptorVersion field of the AqStreamBlockDescriptor structure is invalid.
    VI_SUCCESS                      otherwise.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_readStream(ViSession instrumentID, ViInt32 channel, ViInt32 stream, ViUInt32 flags,
                                           ViPByte bufferP, ViUInt64 bufferSize, AqStreamBlockDescriptor* blockDescP);


//! Reads the frequency counter
/*! 'result'       = result of measurement, whose units depend on the measurement 'type':
                     Hz     for typeMes = 0 (Frequency)
                     sec    for typeMes = 1 (Period)
                     counts for typeMes = 2 (Totalize)
 
    Common return values:
    ACQIRIS_ERROR_NOT_SUPPORTED if this function is not supported by the instrument.
    ACQIRIS_ERROR_NO_DATA       if the instrument has not first acquired data in the 'frequency 
                                counter' mode (mode = 6).
    VI_SUCCESS                  otherwise.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_readFCounter(ViSession instrumentID, ViReal64* result); 



//! Returns the number of segments already acquired.
/*! Can be called during the acquisition period, in order to follow the progress of a
    Sequence acquisition. 
    Can be called after an acquisition, in order to obtain the number of segments actually
    acquired (until '$prefixD1$_stopAcquisition' was called).
 
    Returns one of the following ViStatus values:
    VI_SUCCESS                           always.*/
ACQ_DLL ViStatus ACQ_CC AcqrsD1_reportNbrAcquiredSegments(ViSession instrumentID, 
    ViInt32* nbrSegments); 



//! Resets the digitizer memory to a known default state. 
/*! ONLY useful for a digitizer with the battery back-up option. 

    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_INSTRUMENT_RUNNING    if the instrument is already running. 
    VI_SUCCESS                          otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_resetDigitizerMemory(ViSession instrumentID);



//! Restores some internal registers of an instrument. 
/*! Needed ONLY after power-up of a digitizer with the battery back-up option.
    Please refer to the manual for a detailed description of the steps required
    to read battery backed-up waveforms.
 
    Returns one of the following ViStatus values:
    VI_SUCCESS                           always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_restoreInternalRegisters(ViSession instrumentID,
    ViReal64 delayOffset, ViReal64 delayScale);



//! Stops the acquisition immediately
/*! Returns one of the following ViStatus values:
    VI_SUCCESS                           always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_stopAcquisition(ViSession instrumentID);



//! Stops the on-board processing immediately.(only in instruments with on-board data processing)
/*! Returns one of the following ViStatus values:
    ACQIRIS_ERROR_NOT_SUPPORTED      if this function is not supported by the instrument.
    VI_SUCCESS                       otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_stopProcessing(ViSession instrumentID);



//! Returns after acquisition has terminated or after timeout, whichever comes first.
/*! 'timeout' is in milliseconds. For protection, 'timeout' is internally clipped to a
    range of [0, 10000] milliseconds.
 
    This function puts the calling thread into 'idle' until it returns, permitting optimal 
    use of the CPU by other threads.
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_OVERLOAD           if a channel/trigger overload was detected.
    ACQIRIS_ERROR_ACQ_TIMEOUT        if the acquisition timed out (and there was no overload). 
                                     In this case, you should use either 
                                     '$prefixD1$_stopAcquisition()' or '$prefixD1$_forceTrig()' to 
                                     stop the acquisition.
    ACQIRIS_ERROR_IO_READ            if a link error has been detected (e.g. PCI link lost).
    ACQIRIS_ERROR_INSTRUMENT_STOPPED if the acquisition was not started beforehand
    VI_SUCCESS                       always. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_waitForEndOfAcquisition(ViSession instrumentID, ViInt32 timeout);



//! Returns after on-board processing has terminated or after timeout, whichever comes first.
/*! 'timeout' is in milliseconds. For protection, 'timeout' is internally clipped to a
    range of [0, 10000] milliseconds. (only in instruments with on-board data processing)
 
    This function puts the calling thread into 'idle' until it returns, permitting optimal 
    use of the CPU by other threads.
 
    Returns one of the following ViStatus values:
    ACQIRIS_ERROR_PROC_TIMEOUT        if the processing timed out. In this case, you should use 
                                      '$prefixD1$_stopProcessing()' to stop the processing.
    ACQIRIS_ERROR_IO_READ             if a link error has been detected (e.g. PCI link lost).
    VI_SUCCESS                        otherwise. */
ACQ_DLL ViStatus ACQ_CC AcqrsD1_waitForEndOfProcessing(ViSession instrumentID, ViInt32 timeout);




#if defined( __cplusplus ) 
    }
#endif


#ifndef _ACQIRISIVI_H
#define _ACQIRISIVI_H

//////////////////////////////////////////////////////////////////////////////////////////
//
//  AcqirisExDataTypes.h        
//----------------------------------------------------------------------------------------
//  Copyright (C) 2011 Agilent Technologies, Inc.
//
//  Purpose: This file has been created to remove ivi.h dependencies from the fundamental
//           driver.
//
//////////////////////////////////////////////////////////////////////////////////////////


// Error codes used to remove ivi.h dependencies
#define ACQIRIS_IVI_ERROR_BASE                                      (0xBFFA0000L)
#define ACQIRIS_IVI_ERROR_CANNOT_RECOVER                            (ACQIRIS_IVI_ERROR_BASE + 0x00)
#define ACQIRIS_IVI_ERROR_INSTRUMENT_STATUS                         (ACQIRIS_IVI_ERROR_BASE + 0x01)
#define ACQIRIS_IVI_ERROR_CANNOT_OPEN_FILE                          (ACQIRIS_IVI_ERROR_BASE + 0x02)
#define ACQIRIS_IVI_ERROR_READING_FILE                              (ACQIRIS_IVI_ERROR_BASE + 0x03)            
#define ACQIRIS_IVI_ERROR_WRITING_FILE                              (ACQIRIS_IVI_ERROR_BASE + 0x04)            
#define ACQIRIS_IVI_ERROR_INVALID_PATHNAME                          (ACQIRIS_IVI_ERROR_BASE + 0x0B)
#define ACQIRIS_IVI_ERROR_INVALID_ATTRIBUTE                         (ACQIRIS_IVI_ERROR_BASE + 0x0C)
#define ACQIRIS_IVI_ERROR_IVI_ATTR_NOT_WRITABLE                     (ACQIRIS_IVI_ERROR_BASE + 0x0D)
#define ACQIRIS_IVI_ERROR_IVI_ATTR_NOT_READABLE                     (ACQIRIS_IVI_ERROR_BASE + 0x0E)
#define ACQIRIS_IVI_ERROR_INVALID_VALUE                             (ACQIRIS_IVI_ERROR_BASE + 0x10)
#define ACQIRIS_IVI_ERROR_FUNCTION_NOT_SUPPORTED                    (ACQIRIS_IVI_ERROR_BASE + 0x11) 
#define ACQIRIS_IVI_ERROR_ATTRIBUTE_NOT_SUPPORTED                   (ACQIRIS_IVI_ERROR_BASE + 0x12)
#define ACQIRIS_IVI_ERROR_VALUE_NOT_SUPPORTED                       (ACQIRIS_IVI_ERROR_BASE + 0x13)
#define ACQIRIS_IVI_ERROR_TYPES_DO_NOT_MATCH                        (ACQIRIS_IVI_ERROR_BASE + 0x15)
#define ACQIRIS_IVI_ERROR_NOT_INITIALIZED                           (ACQIRIS_IVI_ERROR_BASE + 0x1D)
#define ACQIRIS_IVI_ERROR_UNKNOWN_CHANNEL_NAME                      (ACQIRIS_IVI_ERROR_BASE + 0x20)
#define ACQIRIS_IVI_ERROR_TOO_MANY_OPEN_FILES                       (ACQIRIS_IVI_ERROR_BASE + 0x23)
#define ACQIRIS_IVI_ERROR_CHANNEL_NAME_REQUIRED                     (ACQIRIS_IVI_ERROR_BASE + 0x44)
#define ACQIRIS_IVI_ERROR_CHANNEL_NAME_NOT_ALLOWED                  (ACQIRIS_IVI_ERROR_BASE + 0x45)
#define ACQIRIS_IVI_ERROR_MISSING_OPTION_NAME                       (ACQIRIS_IVI_ERROR_BASE + 0x49)
#define ACQIRIS_IVI_ERROR_MISSING_OPTION_VALUE                      (ACQIRIS_IVI_ERROR_BASE + 0x4A)
#define ACQIRIS_IVI_ERROR_BAD_OPTION_NAME                           (ACQIRIS_IVI_ERROR_BASE + 0x4B)
#define ACQIRIS_IVI_ERROR_BAD_OPTION_VALUE                          (ACQIRIS_IVI_ERROR_BASE + 0x4C)
#define ACQIRIS_IVI_ERROR_OUT_OF_MEMORY                             (ACQIRIS_IVI_ERROR_BASE + 0x56)
#define ACQIRIS_IVI_ERROR_OPERATION_PENDING                         (ACQIRIS_IVI_ERROR_BASE + 0x57)
#define ACQIRIS_IVI_ERROR_NULL_POINTER                              (ACQIRIS_IVI_ERROR_BASE + 0x58)
#define ACQIRIS_IVI_ERROR_UNEXPECTED_RESPONSE                       (ACQIRIS_IVI_ERROR_BASE + 0x59)
#define ACQIRIS_IVI_ERROR_FILE_NOT_FOUND                            (ACQIRIS_IVI_ERROR_BASE + 0x5B)
#define ACQIRIS_IVI_ERROR_INVALID_FILE_FORMAT                       (ACQIRIS_IVI_ERROR_BASE + 0x5C)
#define ACQIRIS_IVI_ERROR_STATUS_NOT_AVAILABLE                      (ACQIRIS_IVI_ERROR_BASE + 0x5D)
#define ACQIRIS_IVI_ERROR_ID_QUERY_FAILED                           (ACQIRIS_IVI_ERROR_BASE + 0x5E)
#define ACQIRIS_IVI_ERROR_RESET_FAILED                              (ACQIRIS_IVI_ERROR_BASE + 0x5F)
#define ACQIRIS_IVI_ERROR_RESOURCE_UNKNOWN                          (ACQIRIS_IVI_ERROR_BASE + 0x60)
#define ACQIRIS_IVI_ERROR_ALREADY_INITIALIZED                       (ACQIRIS_IVI_ERROR_BASE + 0x61)
#define ACQIRIS_IVI_ERROR_CANNOT_CHANGE_SIMULATION_STATE            (ACQIRIS_IVI_ERROR_BASE + 0x62)
#define ACQIRIS_IVI_ERROR_INVALID_NUMBER_OF_LEVELS_IN_SELECTOR      (ACQIRIS_IVI_ERROR_BASE + 0x63)
#define ACQIRIS_IVI_ERROR_INVALID_RANGE_IN_SELECTOR                 (ACQIRIS_IVI_ERROR_BASE + 0x64)
#define ACQIRIS_IVI_ERROR_UNKOWN_NAME_IN_SELECTOR                   (ACQIRIS_IVI_ERROR_BASE + 0x65)
#define ACQIRIS_IVI_ERROR_BADLY_FORMED_SELECTOR                     (ACQIRIS_IVI_ERROR_BASE + 0x66)
#define ACQIRIS_IVI_ERROR_UNKNOWN_PHYSICAL_IDENTIFIER               (ACQIRIS_IVI_ERROR_BASE + 0x67)

#define ACQIRIS_IVI_MAX_SPECIFIC_ERROR_CODE                         (ACQIRIS_IVI_ERROR_BASE + 0x7FFF)


// Warning codes used to remove ivi.h dependencies
#define ACQIRIS_IVI_WARN_BASE                                       (0x3FFA0000L)
#define ACQIRIS_IVI_WARN_NSUP_ID_QUERY                              (ACQIRIS_IVI_WARN_BASE + 0x65)
#define ACQIRIS_IVI_WARN_NSUP_RESET                                 (ACQIRIS_IVI_WARN_BASE + 0x66)
#define ACQIRIS_IVI_WARN_NSUP_SELF_TEST                             (ACQIRIS_IVI_WARN_BASE + 0x67)
#define ACQIRIS_IVI_WARN_NSUP_ERROR_QUERY                           (ACQIRIS_IVI_WARN_BASE + 0x68)
#define ACQIRIS_IVI_WARN_NSUP_REV_QUERY                             (ACQIRIS_IVI_WARN_BASE + 0x69)

#define ACQIRIS_IVI_MAX_SPECIFIC_WARN_CODE                          (ACQIRIS_IVI_WARN_BASE + 0x7FFF)


// Attributes used to remove ivi.h dependencies
#define ACQIRIS_IVI_ATTR_BASE                                       1050000

#define ACQIRIS_IVI_ATTR_NONE                                       0xFFFFFFFF

#define ACQIRIS_IVI_ATTR_RANGE_CHECK                                (ACQIRIS_IVI_ATTR_BASE + 2)     
#define ACQIRIS_IVI_ATTR_QUERY_INSTRUMENT_STATUS                    (ACQIRIS_IVI_ATTR_BASE + 3)
#define ACQIRIS_IVI_ATTR_CACHE                                      (ACQIRIS_IVI_ATTR_BASE + 4)     
#define ACQIRIS_IVI_ATTR_SIMULATE                                   (ACQIRIS_IVI_ATTR_BASE + 5)    
#define ACQIRIS_IVI_ATTR_RECORD_COERCIONS                           (ACQIRIS_IVI_ATTR_BASE + 6)
#define ACQIRIS_IVI_ATTR_DRIVER_SETUP                               (ACQIRIS_IVI_ATTR_BASE + 7)

#define ACQIRIS_IVI_ATTR_INTERCHANGE_CHECK                          (ACQIRIS_IVI_ATTR_BASE + 21)
#define ACQIRIS_IVI_ATTR_SPY                                        (ACQIRIS_IVI_ATTR_BASE + 22)     
#define ACQIRIS_IVI_ATTR_USE_SPECIFIC_SIMULATION                    (ACQIRIS_IVI_ATTR_BASE + 23)

#define ACQIRIS_IVI_ATTR_DEFER_UPDATE                               (ACQIRIS_IVI_ATTR_BASE + 51)     
#define ACQIRIS_IVI_ATTR_RETURN_DEFERRED_VALUES                     (ACQIRIS_IVI_ATTR_BASE + 52)

#define ACQIRIS_IVI_ATTR_PRIMARY_ERROR                              (ACQIRIS_IVI_ATTR_BASE + 101)     
#define ACQIRIS_IVI_ATTR_SECONDARY_ERROR                            (ACQIRIS_IVI_ATTR_BASE + 102)     
#define ACQIRIS_IVI_ATTR_ERROR_ELABORATION                          (ACQIRIS_IVI_ATTR_BASE + 103)    

#define ACQIRIS_IVI_ATTR_CHANNEL_COUNT                              (ACQIRIS_IVI_ATTR_BASE + 203)

#define ACQIRIS_IVI_ATTR_CLASS_DRIVER_PREFIX                        (ACQIRIS_IVI_ATTR_BASE + 301)
#define ACQIRIS_IVI_ATTR_SPECIFIC_DRIVER_PREFIX                     (ACQIRIS_IVI_ATTR_BASE + 302)
#define ACQIRIS_IVI_ATTR_SPECIFIC_DRIVER_LOCATOR                    (ACQIRIS_IVI_ATTR_BASE + 303)
#define ACQIRIS_IVI_ATTR_IO_RESOURCE_DESCRIPTOR                     (ACQIRIS_IVI_ATTR_BASE + 304)
#define ACQIRIS_IVI_ATTR_LOGICAL_NAME                               (ACQIRIS_IVI_ATTR_BASE + 305)
#define ACQIRIS_IVI_ATTR_VISA_RM_SESSION                            (ACQIRIS_IVI_ATTR_BASE + 321)   
#define ACQIRIS_IVI_ATTR_IO_SESSION                                 (ACQIRIS_IVI_ATTR_BASE + 322)     
#define ACQIRIS_IVI_ATTR_IO_SESSION_TYPE                            (ACQIRIS_IVI_ATTR_BASE + 324)
#define ACQIRIS_IVI_ATTR_SUPPORTED_INSTRUMENT_MODELS                (ACQIRIS_IVI_ATTR_BASE + 327)

#define ACQIRIS_IVI_ATTR_GROUP_CAPABILITIES                         (ACQIRIS_IVI_ATTR_BASE + 401)    
#define ACQIRIS_IVI_ATTR_FUNCTION_CAPABILITIES                      (ACQIRIS_IVI_ATTR_BASE + 402)    
    
#define ACQIRIS_IVI_ATTR_ENGINE_MAJOR_VERSION                       (ACQIRIS_IVI_ATTR_BASE + 501)   
#define ACQIRIS_IVI_ATTR_ENGINE_MINOR_VERSION                       (ACQIRIS_IVI_ATTR_BASE + 502)   
#define ACQIRIS_IVI_ATTR_SPECIFIC_DRIVER_MAJOR_VERSION              (ACQIRIS_IVI_ATTR_BASE + 503)   
#define ACQIRIS_IVI_ATTR_SPECIFIC_DRIVER_MINOR_VERSION              (ACQIRIS_IVI_ATTR_BASE + 504)   
#define ACQIRIS_IVI_ATTR_CLASS_DRIVER_MAJOR_VERSION                 (ACQIRIS_IVI_ATTR_BASE + 505)   
#define ACQIRIS_IVI_ATTR_CLASS_DRIVER_MINOR_VERSION                 (ACQIRIS_IVI_ATTR_BASE + 506)   

#define ACQIRIS_IVI_ATTR_INSTRUMENT_FIRMWARE_REVISION               (ACQIRIS_IVI_ATTR_BASE + 510)
#define ACQIRIS_IVI_ATTR_INSTRUMENT_MANUFACTURER                    (ACQIRIS_IVI_ATTR_BASE + 511)
#define ACQIRIS_IVI_ATTR_INSTRUMENT_MODEL                           (ACQIRIS_IVI_ATTR_BASE + 512)
#define ACQIRIS_IVI_ATTR_SPECIFIC_DRIVER_VENDOR                     (ACQIRIS_IVI_ATTR_BASE + 513)
#define ACQIRIS_IVI_ATTR_SPECIFIC_DRIVER_DESCRIPTION                (ACQIRIS_IVI_ATTR_BASE + 514)
#define ACQIRIS_IVI_ATTR_SPECIFIC_DRIVER_CLASS_SPEC_MAJOR_VERSION   (ACQIRIS_IVI_ATTR_BASE + 515)
#define ACQIRIS_IVI_ATTR_SPECIFIC_DRIVER_CLASS_SPEC_MINOR_VERSION   (ACQIRIS_IVI_ATTR_BASE + 516)
#define ACQIRIS_IVI_ATTR_CLASS_DRIVER_VENDOR                        (ACQIRIS_IVI_ATTR_BASE + 517)
#define ACQIRIS_IVI_ATTR_CLASS_DRIVER_DESCRIPTION                   (ACQIRIS_IVI_ATTR_BASE + 518)
#define ACQIRIS_IVI_ATTR_CLASS_DRIVER_CLASS_SPEC_MAJOR_VERSION      (ACQIRIS_IVI_ATTR_BASE + 519)
#define ACQIRIS_IVI_ATTR_CLASS_DRIVER_CLASS_SPEC_MINOR_VERSION      (ACQIRIS_IVI_ATTR_BASE + 520)

#define ACQIRIS_IVI_ATTR_SPECIFIC_DRIVER_REVISION                   (ACQIRIS_IVI_ATTR_BASE + 551)    
#define ACQIRIS_IVI_ATTR_CLASS_DRIVER_REVISION                      (ACQIRIS_IVI_ATTR_BASE + 552)   
#define ACQIRIS_IVI_ATTR_ENGINE_REVISION                            (ACQIRIS_IVI_ATTR_BASE + 553)    

#define ACQIRIS_IVI_ATTR_BUFFERED_IO_CALLBACK                       (ACQIRIS_IVI_ATTR_BASE + 601)
#define ACQIRIS_IVI_ATTR_OPC_CALLBACK                               (ACQIRIS_IVI_ATTR_BASE + 602)
#define ACQIRIS_IVI_ATTR_CHECK_STATUS_CALLBACK                      (ACQIRIS_IVI_ATTR_BASE + 603)

#define ACQIRIS_IVI_ATTR_SUPPORTS_WR_BUF_OPER_MODE                  (ACQIRIS_IVI_ATTR_BASE + 704)
#define ACQIRIS_IVI_ATTR_UPDATING_VALUES                            (ACQIRIS_IVI_ATTR_BASE + 708)

#define ACQIRIS_IVI_ATTR_USER_INTERCHANGE_CHECK_CALLBACK            (ACQIRIS_IVI_ATTR_BASE + 801)


// Command string of last element
#define ACQIRIS_IVI_RANGE_TABLE_END_STRING      ((ViString)(-1))
// Last item in range table
#define ACQIRIS_IVI_RANGE_TABLE_LAST_ITEM    VI_NULL, VI_NULL, VI_NULL, ACQIRIS_IVI_RANGE_TABLE_END_STRING, VI_NULL


// Range Table values
#define ACQIRIS_IVI_VAL_DISCRETE            0 
#define ACQIRIS_IVI_VAL_RANGED              1
#define ACQIRIS_IVI_VAL_COERCED             2

// Elements of range table
typedef struct
{
    ViReal64    discreteOrMinValue;
    ViReal64    maxValue;
    ViReal64    coercedValue;
    ViString    cmdString;
    ViInt32     cmdValue;
} AqAttrRangeTableItem;
    
typedef struct
{
    ViInt32                     type;
    ViBoolean                   hasMin;
    ViBoolean                   hasMax;
    ViString                    customInfo;
    // ACQIRIS_IVI_RANGE_TABLE_LAST_ELEMENT marks the end of rangeValues
    AqAttrRangeTableItem        *rangeValues;  
} AqAttrRangeTable;
    
typedef AqAttrRangeTable*  AqAttrRangeTablePtr;

#endif// sentry

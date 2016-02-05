/******************************************************************************
 *
 * Copyright 2010-2013 Agilent Technologies. All rights reserved.
 *
 *****************************************************************************/

#ifndef AGMD1_HEADER
#define AGMD1_HEADER

#include <visatype.h>

#if defined(__cplusplus) || defined(__cplusplus__)
extern "C" {
#endif

/****************************************************************************
 *---------------------------- Attribute Defines ---------------------------*
 ****************************************************************************/
#ifndef IVI_ATTR_BASE
#define IVI_ATTR_BASE                 1000000
#endif

#ifndef IVI_INHERENT_ATTR_BASE
#define IVI_INHERENT_ATTR_BASE        (IVI_ATTR_BASE +  50000)   /* base for inherent capability attributes */
#endif

#ifndef IVI_CLASS_ATTR_BASE
#define IVI_CLASS_ATTR_BASE           (IVI_ATTR_BASE + 250000)   /* base for IVI-defined class attributes */
#endif

#ifndef IVI_LXISYNC_ATTR_BASE
#define IVI_LXISYNC_ATTR_BASE         (IVI_ATTR_BASE + 950000)   /* base for IviLxiSync attributes */
#endif

#ifndef IVI_SPECIFIC_ATTR_BASE
#define IVI_SPECIFIC_ATTR_BASE        (IVI_ATTR_BASE + 150000)   /* base for attributes of specific drivers */
#endif


/*===== IVI Inherent Instrument Attributes ==============================*/

/*- Driver Identification */

#define AGMD1_ATTR_SPECIFIC_DRIVER_DESCRIPTION              (IVI_INHERENT_ATTR_BASE + 514L)  /* ViString, read-only */
#define AGMD1_ATTR_SPECIFIC_DRIVER_PREFIX                   (IVI_INHERENT_ATTR_BASE + 302L)  /* ViString, read-only */
#define AGMD1_ATTR_SPECIFIC_DRIVER_VENDOR                   (IVI_INHERENT_ATTR_BASE + 513L)  /* ViString, read-only */
#define AGMD1_ATTR_SPECIFIC_DRIVER_REVISION                 (IVI_INHERENT_ATTR_BASE + 551L)  /* ViString, read-only */
#define AGMD1_ATTR_SPECIFIC_DRIVER_CLASS_SPEC_MAJOR_VERSION (IVI_INHERENT_ATTR_BASE + 515L)  /* ViInt32, read-only */
#define AGMD1_ATTR_SPECIFIC_DRIVER_CLASS_SPEC_MINOR_VERSION (IVI_INHERENT_ATTR_BASE + 516L)  /* ViInt32, read-only */

/*- User Options */

#define AGMD1_ATTR_RANGE_CHECK                              (IVI_INHERENT_ATTR_BASE + 2L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_QUERY_INSTRUMENT_STATUS                  (IVI_INHERENT_ATTR_BASE + 3L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_CACHE                                    (IVI_INHERENT_ATTR_BASE + 4L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_SIMULATE                                 (IVI_INHERENT_ATTR_BASE + 5L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_RECORD_COERCIONS                         (IVI_INHERENT_ATTR_BASE + 6L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_INTERCHANGE_CHECK                        (IVI_INHERENT_ATTR_BASE + 21L)  /* ViBoolean, read-write */

/*- Advanced Session Information */

#define AGMD1_ATTR_LOGICAL_NAME                             (IVI_INHERENT_ATTR_BASE + 305L)  /* ViString, read-only */
#define AGMD1_ATTR_IO_RESOURCE_DESCRIPTOR                   (IVI_INHERENT_ATTR_BASE + 304L)  /* ViString, read-only */
#define AGMD1_ATTR_DRIVER_SETUP                             (IVI_INHERENT_ATTR_BASE + 7L)  /* ViString, read-only */

/*- Driver Capabilities */

#define AGMD1_ATTR_GROUP_CAPABILITIES                       (IVI_INHERENT_ATTR_BASE + 401L)  /* ViString, read-only */
#define AGMD1_ATTR_SUPPORTED_INSTRUMENT_MODELS              (IVI_INHERENT_ATTR_BASE + 327L)  /* ViString, read-only */

/*- Instrument Identification */

#define AGMD1_ATTR_INSTRUMENT_FIRMWARE_REVISION             (IVI_INHERENT_ATTR_BASE + 510L)  /* ViString, read-only */
#define AGMD1_ATTR_INSTRUMENT_MANUFACTURER                  (IVI_INHERENT_ATTR_BASE + 511L)  /* ViString, read-only */
#define AGMD1_ATTR_INSTRUMENT_MODEL                         (IVI_INHERENT_ATTR_BASE + 512L)  /* ViString, read-only */


/*===== Instrument-Specific Attributes =====================================*/

/*- Temperature */

#define AGMD1_ATTR_CHANNEL_TEMPERATURE                      (IVI_CLASS_ATTR_BASE + 300L)  /* ViReal64, read-only */
#define AGMD1_ATTR_TEMPERATURE_UNITS                        (IVI_CLASS_ATTR_BASE + 101L)  /* ViInt32, read-write */
#define AGMD1_ATTR_BOARD_TEMPERATURE                        (IVI_CLASS_ATTR_BASE + 100L)  /* ViReal64, read-only */

/*- Waveform Acquisition */

#define AGMD1_ATTR_SAMPLE_MODE                              (IVI_CLASS_ATTR_BASE + 800L)  /* ViInt32, read-write */
#define AGMD1_ATTR_MAX_SAMPLES_PER_CHANNEL                  (IVI_CLASS_ATTR_BASE + 10L)  /* ViInt64, read-only */
#define AGMD1_ATTR_NUM_ACQUIRED_RECORDS                     (IVI_CLASS_ATTR_BASE + 12L)  /* ViInt64, read-only */
#define AGMD1_ATTR_SAMPLE_RATE                              (IVI_CLASS_ATTR_BASE + 15L)  /* ViReal64, read-write */
#define AGMD1_ATTR_MIN_RECORD_SIZE                          (IVI_CLASS_ATTR_BASE + 11L)  /* ViInt64, read-only */
#define AGMD1_ATTR_RECORD_SIZE                              (IVI_CLASS_ATTR_BASE + 14L)  /* ViInt64, read-write */
#define AGMD1_ATTR_NUM_RECORDS_TO_ACQUIRE                   (IVI_CLASS_ATTR_BASE + 13L)  /* ViInt64, read-write */
#define AGMD1_ATTR_IS_IDLE                                  (IVI_CLASS_ATTR_BASE + 5L)  /* ViInt32, read-only */
#define AGMD1_ATTR_IS_MEASURING                             (IVI_CLASS_ATTR_BASE + 6L)  /* ViInt32, read-only */
#define AGMD1_ATTR_IS_WAITING_FOR_ARM                       (IVI_CLASS_ATTR_BASE + 7L)  /* ViInt32, read-only */
#define AGMD1_ATTR_IS_WAITING_FOR_TRIGGER                   (IVI_CLASS_ATTR_BASE + 8L)  /* ViInt32, read-only */
#define AGMD1_ATTR_MAX_FIRST_VALID_POINT_VAL                (IVI_CLASS_ATTR_BASE + 9L)  /* ViInt64, read-write */

/*- Channel */

#define AGMD1_ATTR_CHANNEL_COUNT                            (IVI_INHERENT_ATTR_BASE + 203L)  /* ViInt32, read-only */
#define AGMD1_ATTR_VERTICAL_COUPLING                        (IVI_CLASS_ATTR_BASE + 24L)  /* ViInt32, read-write */
#define AGMD1_ATTR_INPUT_IMPEDANCE                          (IVI_CLASS_ATTR_BASE + 4L)  /* ViReal64, read-write */
#define AGMD1_ATTR_VERTICAL_RANGE                           (IVI_CLASS_ATTR_BASE + 26L)  /* ViReal64, read-write */
#define AGMD1_ATTR_VERTICAL_OFFSET                          (IVI_CLASS_ATTR_BASE + 25L)  /* ViReal64, read-write */
#define AGMD1_ATTR_CHANNEL_ENABLED                          (IVI_CLASS_ATTR_BASE + 2L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_INPUT_CONNECTOR_SELECTION                (IVI_CLASS_ATTR_BASE + 3L)  /* ViInt32, read-write */
#define AGMD1_ATTR_TIME_INTERLEAVED_CHANNEL_LIST            (IVI_CLASS_ATTR_BASE + 400L)  /* ViString, read-write */

/*- Filter */

#define AGMD1_ATTR_INPUT_FILTER_MAX_FREQUENCY               (IVI_CLASS_ATTR_BASE + 201L)  /* ViReal64, read-write */
#define AGMD1_ATTR_INPUT_FILTER_BYPASS                      (IVI_CLASS_ATTR_BASE + 200L)  /* ViBoolean, read-write */

/*- Downconversion */

#define AGMD1_ATTR_DOWNCONVERSION_ENABLED                   (IVI_CLASS_ATTR_BASE + 901L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_DOWNCONVERSION_CENTER_FREQUENCY          (IVI_CLASS_ATTR_BASE + 900L)  /* ViReal64, read-write */
#define AGMD1_ATTR_DOWNCONVERSION_IQ_INTERLEAVED            (IVI_CLASS_ATTR_BASE + 902L)  /* ViBoolean, read-write */

/*- Reference Oscillator */

#define AGMD1_ATTR_REFERENCE_OSCILLATOR_SOURCE              (IVI_CLASS_ATTR_BASE + 602L)  /* ViInt32, read-write */
#define AGMD1_ATTR_REFERENCE_OSCILLATOR_EXTERNAL_FREQUENCY  (IVI_CLASS_ATTR_BASE + 600L)  /* ViReal64, read-write */
#define AGMD1_ATTR_REFERENCE_OSCILLATOR_OUTPUT_ENABLED      (IVI_CLASS_ATTR_BASE + 601L)  /* ViBoolean, read-write */

/*- Sample Clock */

#define AGMD1_ATTR_SAMPLE_CLOCK_SOURCE                      (IVI_CLASS_ATTR_BASE + 703L)  /* ViInt32, read-write */
#define AGMD1_ATTR_SAMPLE_CLOCK_EXTERNAL_FREQUENCY          (IVI_CLASS_ATTR_BASE + 701L)  /* ViReal64, read-write */
#define AGMD1_ATTR_SAMPLE_CLOCK_EXTERNAL_DIVIDER            (IVI_CLASS_ATTR_BASE + 700L)  /* ViReal64, read-write */

/*- Trigger */

#define AGMD1_ATTR_TRIGGER_COUPLING                         (IVI_CLASS_ATTR_BASE + 16L)  /* ViInt32, read-write */
#define AGMD1_ATTR_TRIGGER_HYSTERESIS                       (IVI_CLASS_ATTR_BASE + 18L)  /* ViReal64, read-write */
#define AGMD1_ATTR_TRIGGER_LEVEL                            (IVI_CLASS_ATTR_BASE + 19L)  /* ViReal64, read-write */
#define AGMD1_ATTR_TRIGGER_TYPE                             (IVI_CLASS_ATTR_BASE + 23L)  /* ViInt32, read-write */
#define AGMD1_ATTR_ACTIVE_TRIGGER_SOURCE                    (IVI_CLASS_ATTR_BASE + 1L)  /* ViString, read-write */
#define AGMD1_ATTR_TRIGGER_MODIFIER                         (IVI_CLASS_ATTR_BASE + 1700L)  /* ViInt32, read-write */
#define AGMD1_ATTR_TRIGGER_DELAY                            (IVI_CLASS_ATTR_BASE + 17L)  /* ViReal64, read-write */
#define AGMD1_ATTR_TRIGGER_OUTPUT_ENABLED                   (IVI_CLASS_ATTR_BASE + 20L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_TRIGGER_SOURCE_COUNT                     (IVI_CLASS_ATTR_BASE + 22L)  /* ViInt32, read-only */

/*- Edge Triggering */

#define AGMD1_ATTR_TRIGGER_SLOPE                            (IVI_CLASS_ATTR_BASE + 21L)  /* ViInt32, read-write */

/*- TV Triggering */

#define AGMD1_ATTR_TV_TRIGGER_POLARITY                      (IVI_CLASS_ATTR_BASE + 2302L)  /* ViInt32, read-write */
#define AGMD1_ATTR_TV_TRIGGER_EVENT                         (IVI_CLASS_ATTR_BASE + 2300L)  /* ViInt32, read-write */
#define AGMD1_ATTR_TV_TRIGGER_LINE_NUMBER                   (IVI_CLASS_ATTR_BASE + 2301L)  /* ViInt32, read-write */
#define AGMD1_ATTR_TV_TRIGGER_SIGNAL_FORMAT                 (IVI_CLASS_ATTR_BASE + 2303L)  /* ViInt32, read-write */

/*- Multi Trigger */

#define AGMD1_ATTR_TRIGGER_SOURCE_LIST                      (IVI_CLASS_ATTR_BASE + 1800L)  /* ViString, read-write */
#define AGMD1_ATTR_TRIGGER_SOURCE_OPERATOR                  (IVI_CLASS_ATTR_BASE + 1801L)  /* ViInt32, read-write */

/*- Acquisition */

#define AGMD1_ATTR_ACQUISITION_START_TIME                   (IVI_SPECIFIC_ATTR_BASE + 1L)  /* ViReal64, read-write */
#define AGMD1_ATTR_ACQUISITION_TIME_PER_RECORD              (IVI_SPECIFIC_ATTR_BASE + 2L)  /* ViReal64, read-write */
#define AGMD1_ATTR_ACQUISITION_NUMBER_OF_AVERAGES           (IVI_SPECIFIC_ATTR_BASE + 3L)  /* ViInt32, read-write */
#define AGMD1_ATTR_ACQUISITION_REFERENCE_CHANNEL            (IVI_SPECIFIC_ATTR_BASE + 11L)  /* ViString, read-write */
#define AGMD1_ATTR_ACQUISITION_DITHER_ENABLED               (IVI_SPECIFIC_ATTR_BASE + 44L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_ACQUISITION_AUTO_ARM_ENABLED             (IVI_SPECIFIC_ATTR_BASE + 45L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_ACQUISITION_DITHER_RANGE                 (IVI_SPECIFIC_ATTR_BASE + 46L)  /* ViInt32, read-write */
#define AGMD1_ATTR_ACQUISITION_MODE                         (IVI_SPECIFIC_ATTR_BASE + 47L)  /* ViInt32, read-write */
#define AGMD1_ATTR_ACQUISITION_DOWNCONVERSION_DATA_SCALING_ENABLED (IVI_SPECIFIC_ATTR_BASE + 66L)  /* ViBoolean, read-write */

/*- UserControl */

#define AGMD1_ATTR_USER_CONTROL_COUNTERS_MODE               (IVI_SPECIFIC_ATTR_BASE + 69L)  /* ViInt32, read-write */
#define AGMD1_ATTR_USER_CONTROL_INTER_SEGMENT_DELAY_MODE    (IVI_SPECIFIC_ATTR_BASE + 70L)  /* ViInt32, read-write */
#define AGMD1_ATTR_USER_CONTROL_INTER_SEGMENT_DELAY_VALUE   (IVI_SPECIFIC_ATTR_BASE + 71L)  /* ViInt32, read-write */
#define AGMD1_ATTR_USER_CONTROL_POST_TRIGGER                (IVI_SPECIFIC_ATTR_BASE + 72L)  /* ViInt32, read-write */
#define AGMD1_ATTR_USER_CONTROL_PRE_TRIGGER                 (IVI_SPECIFIC_ATTR_BASE + 73L)  /* ViInt32, read-write */
#define AGMD1_ATTR_USER_CONTROL_TRIGGER_ENABLE_SOURCE       (IVI_SPECIFIC_ATTR_BASE + 74L)  /* ViInt32, read-write */
#define AGMD1_ATTR_USER_CONTROL_TRIGGER_FACTOR              (IVI_SPECIFIC_ATTR_BASE + 75L)  /* ViInt32, read-write */
#define AGMD1_ATTR_USER_CONTROL_TRIGGER_SELECTION           (IVI_SPECIFIC_ATTR_BASE + 89L)  /* ViInt32, read-write */
#define AGMD1_ATTR_USER_CONTROL_START_ON_TRIGGER_ENABLED    (IVI_SPECIFIC_ATTR_BASE + 90L)  /* ViBoolean, read-write */

/*- SAR */

#define AGMD1_ATTR_SAR_ENABLED                              (IVI_SPECIFIC_ATTR_BASE + 91L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_SAR_IS_ACQUISITION_COMPLETE              (IVI_SPECIFIC_ATTR_BASE + 92L)  /* ViBoolean, read-only */
#define AGMD1_ATTR_SAR_MEMORY_OVERFLOW_OCCURRED             (IVI_SPECIFIC_ATTR_BASE + 93L)  /* ViBoolean, read-only */

/*- Channel */

#define AGMD1_ATTR_CHANNEL_INPUT_FREQUENCY_MAX              (IVI_SPECIFIC_ATTR_BASE + 4L)  /* ViReal64, read-write */
#define AGMD1_ATTR_CHANNEL_PROBE_ATTENUATION                (IVI_SPECIFIC_ATTR_BASE + 5L)  /* ViReal64, read-write */
#define AGMD1_ATTR_CHANNEL_CONNECTOR_NAME                   (IVI_SPECIFIC_ATTR_BASE + 12L)  /* ViString, read-only */
#define AGMD1_ATTR_CHANNEL_ACTIVE_INPUTS                    (IVI_SPECIFIC_ATTR_BASE + 13L)  /* ViString, read-write */
#define AGMD1_ATTR_CHANNEL_CONVERTERS_PER_CHANNEL           (IVI_SPECIFIC_ATTR_BASE + 14L)  /* ViInt32, read-only */

/*- Measurement */

#define AGMD1_ATTR_MEASUREMENT_WAVEFORM_TYPE                (IVI_SPECIFIC_ATTR_BASE + 17L)  /* ViInt32, read-write */

/*- DigitalDownconversion */

#define AGMD1_ATTR_DIGITAL_DOWNCONVERSION_DATA_BANDWIDTH           (IVI_SPECIFIC_ATTR_BASE + 39L)  /* ViReal64, read-write */
#define AGMD1_ATTR_DIGITAL_DOWNCONVERSION_TRIGGER_BANDWIDTH        (IVI_SPECIFIC_ATTR_BASE + 40L)  /* ViReal64, read-write */
#define AGMD1_ATTR_DIGITAL_DOWNCONVERSION_FREQUENCY_SHIFT          (IVI_SPECIFIC_ATTR_BASE + 41L)  /* ViReal64, read-write */
#define AGMD1_ATTR_DIGITAL_DOWNCONVERSION_DOUBLE_DATA_RATE_ENABLED (IVI_SPECIFIC_ATTR_BASE + 42L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_DIGITAL_DOWNCONVERSION_FLIP_BAND_ENABLED        (IVI_SPECIFIC_ATTR_BASE + 43L)  /* ViBoolean, read-write */

/*- Counter */

#define AGMD1_ATTR_COUNTER_APERTURE_TIME                    (IVI_SPECIFIC_ATTR_BASE + 53L)  /* ViReal64, read-write */
#define AGMD1_ATTR_COUNTER_MODE                             (IVI_SPECIFIC_ATTR_BASE + 54L)  /* ViInt32, read-write */

/*- ReferenceLevel */

#define AGMD1_ATTR_REFERENCE_LEVEL_HIGH                     (IVI_SPECIFIC_ATTR_BASE + 6L)  /* ViReal64, read-write */
#define AGMD1_ATTR_REFERENCE_LEVEL_LOW                      (IVI_SPECIFIC_ATTR_BASE + 7L)  /* ViReal64, read-write */
#define AGMD1_ATTR_REFERENCE_LEVEL_MID                      (IVI_SPECIFIC_ATTR_BASE + 8L)  /* ViReal64, read-write */

/*- System */

#define AGMD1_ATTR_SYSTEM_TRACE_ENABLED                     (IVI_SPECIFIC_ATTR_BASE + 10L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_SYSTEM_NBR_INSTRUMENTS                   (IVI_SPECIFIC_ATTR_BASE + 37L)  /* ViInt32, read-only */

/*- DelayControl */

#define AGMD1_ATTR_DELAY_CONTROL_COUNT                      (IVI_SPECIFIC_ATTR_BASE + 59L)  /* ViInt32, read-only */
#define AGMD1_ATTR_DELAY_CONTROL_MAX                        (IVI_SPECIFIC_ATTR_BASE + 60L)  /* ViInt32, read-only */
#define AGMD1_ATTR_DELAY_CONTROL_MIN                        (IVI_SPECIFIC_ATTR_BASE + 61L)  /* ViInt32, read-only */
#define AGMD1_ATTR_DELAY_CONTROL_VALUE                      (IVI_SPECIFIC_ATTR_BASE + 62L)  /* ViInt32, read-write */
#define AGMD1_ATTR_DELAY_CONTROL_ENABLED                    (IVI_SPECIFIC_ATTR_BASE + 64L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_DELAY_CONTROL_RANGE                      (IVI_SPECIFIC_ATTR_BASE + 65L)  /* ViInt32, read-write */

/*- Trigger */

#define AGMD1_ATTR_PXI_TRIG_OUT_ENABLED                     (IVI_SPECIFIC_ATTR_BASE + 48L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_PXI_TRIG_SLOPE                           (IVI_SPECIFIC_ATTR_BASE + 50L)  /* ViInt32, read-write */

/*- Output */

#define AGMD1_ATTR_TRIGGER_OUTPUT_OFFSET                    (IVI_SPECIFIC_ATTR_BASE + 15L)  /* ViReal64, read-write */
#define AGMD1_ATTR_TRIGGER_OUTPUT_RESYNC_ENABLED            (IVI_SPECIFIC_ATTR_BASE + 16L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_TRIGGER_OUTPUT_MULTI_MODULE_SYNC_ENABLED (IVI_SPECIFIC_ATTR_BASE + 56L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_TRIGGER_OUTPUT_SOURCE                    (IVI_SPECIFIC_ATTR_BASE + 57L)  /* ViString, read-write */

/*- TriggerSource */

#define AGMD1_ATTR_EXT_TRIG_RANGE                           (IVI_SPECIFIC_ATTR_BASE + 49L)  /* ViReal64, read-write */

/*- Magnitude */

#define AGMD1_ATTR_TRIGGER_MAGNITUDE_SLOPE                  (IVI_SPECIFIC_ATTR_BASE + 55L)  /* ViInt32, read-write */
#define AGMD1_ATTR_TRIGGER_MAGNITUDE_DWELL_TIME_SAMPLES     (IVI_SPECIFIC_ATTR_BASE + 58L)  /* ViInt32, read-write */

/*- ControlIO */

#define AGMD1_ATTR_CONTROL_IO_SIGNAL_A                      (IVI_SPECIFIC_ATTR_BASE + 18L)  /* ViString, read-write */
#define AGMD1_ATTR_CONTROL_IO_SIGNAL_B                      (IVI_SPECIFIC_ATTR_BASE + 19L)  /* ViString, read-write */
#define AGMD1_ATTR_CONTROL_IO_SIGNAL_C                      (IVI_SPECIFIC_ATTR_BASE + 38L)  /* ViString, read-write */

/*- InstrumentInfo */

#define AGMD1_ATTR_INSTRUMENT_INFO_AQ_DRV_VERSION               (IVI_SPECIFIC_ATTR_BASE + 21L)  /* ViString, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_AQIO_VERSION                 (IVI_SPECIFIC_ATTR_BASE + 22L)  /* ViString, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_BUS_NUMBER                   (IVI_SPECIFIC_ATTR_BASE + 23L)  /* ViInt32, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_CPLD_FIRMWARE_REV            (IVI_SPECIFIC_ATTR_BASE + 24L)  /* ViString, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_CRATE_NUMBER                 (IVI_SPECIFIC_ATTR_BASE + 25L)  /* ViInt32, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_DEVICE_ID                    (IVI_SPECIFIC_ATTR_BASE + 26L)  /* ViInt32, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_EEPROM_COMMON_SECTION_REV    (IVI_SPECIFIC_ATTR_BASE + 27L)  /* ViString, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_EEPROM_DIGITIZER_SECTION_REV (IVI_SPECIFIC_ATTR_BASE + 28L)  /* ViString, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_HAS_HIGH_RES_SAMPLE_RATE     (IVI_SPECIFIC_ATTR_BASE + 29L)  /* ViBoolean, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_NBRADC_BITS                  (IVI_SPECIFIC_ATTR_BASE + 30L)  /* ViInt32, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_NBR_EXTERNAL_TRIGGERS        (IVI_SPECIFIC_ATTR_BASE + 31L)  /* ViInt32, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_NBR_INTERNAL_TRIGGERS        (IVI_SPECIFIC_ATTR_BASE + 32L)  /* ViInt32, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_NBR_MODULES_IN_INSTRUMENT    (IVI_SPECIFIC_ATTR_BASE + 33L)  /* ViInt32, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_OPTIONS                      (IVI_SPECIFIC_ATTR_BASE + 34L)  /* ViString, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_SERIAL_NUMBER                (IVI_SPECIFIC_ATTR_BASE + 35L)  /* ViInt32, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_SLOT_NUMBER                  (IVI_SPECIFIC_ATTR_BASE + 36L)  /* ViInt32, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_FUNDAMENTAL_HANDLE           (IVI_SPECIFIC_ATTR_BASE + 52L)  /* ViInt32, read-only */
#define AGMD1_ATTR_INSTRUMENT_INFO_SERIAL_NUMBER_STRING         (IVI_SPECIFIC_ATTR_BASE + 51L)  /* ViString, read-only */

/*- LogicDevice */

#define AGMD1_ATTR_LOGIC_DEVICE_COUNT                       (IVI_SPECIFIC_ATTR_BASE + 63L)  /* ViInt32, read-only */
#define AGMD1_ATTR_LOGIC_DEVICE_SAMPLES_UNSIGNED            (IVI_SPECIFIC_ATTR_BASE + 78L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_LOGIC_DEVICE_STREAM_MODE                 (IVI_SPECIFIC_ATTR_BASE + 79L)  /* ViInt32, read-write */
#define AGMD1_ATTR_LOGIC_DEVICE_STREAMS_COUNT               (IVI_SPECIFIC_ATTR_BASE + 80L)  /* ViInt32, read-only */
#define AGMD1_ATTR_LOGIC_DEVICE_FIRMWARE_STORE_COUNT        (IVI_SPECIFIC_ATTR_BASE + 88L)  /* ViInt32, read-only */

/*- LogicDeviceIFDL */

#define AGMD1_ATTR_LOGIC_DEVICE_IFDL_ENABLED                (IVI_SPECIFIC_ATTR_BASE + 81L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_LOGIC_DEVICE_IFDL_IS_ACTIVE              (IVI_SPECIFIC_ATTR_BASE + 82L)  /* ViBoolean, read-only */
#define AGMD1_ATTR_LOGIC_DEVICE_IFDL_COUNT                  (IVI_SPECIFIC_ATTR_BASE + 86L)  /* ViInt32, read-only */

/*- LogicDeviceMemoryBank */

#define AGMD1_ATTR_LOGIC_DEVICE_MEMORY_BANK_ACCESS_CONTROL    (IVI_SPECIFIC_ATTR_BASE + 83L)  /* ViInt32, read-write */
#define AGMD1_ATTR_LOGIC_DEVICE_MEMORY_BANK_ACCESS_MODE       (IVI_SPECIFIC_ATTR_BASE + 84L)  /* ViInt32, read-write */
#define AGMD1_ATTR_LOGIC_DEVICE_MEMORY_BANK_FIFO_MODE_ENABLED (IVI_SPECIFIC_ATTR_BASE + 85L)  /* ViBoolean, read-write */
#define AGMD1_ATTR_LOGIC_DEVICE_MEMORY_BANK_COUNT             (IVI_SPECIFIC_ATTR_BASE + 87L)  /* ViInt32, read-only */

/*- ModuleSynchronization */

#define AGMD1_ATTR_MODULE_SYNCHRONIZATION_MASTER_MODULE_HANDLE (IVI_SPECIFIC_ATTR_BASE + 67L)  /* ViInt32, read-only */
#define AGMD1_ATTR_MODULE_SYNCHRONIZATION_HANDLE               (IVI_SPECIFIC_ATTR_BASE + 68L)  /* ViInt32, read-only */


/****************************************************************************
 *------------------------ Attribute Value Defines -------------------------*
 ****************************************************************************/

/*- Defined values for
    attribute AGMD1_ATTR_VERTICAL_COUPLING
    parameter Coupling in function AgMD1_ConfigureChannel */

#define AGMD1_VAL_VERTICAL_COUPLING_AC                      0
#define AGMD1_VAL_VERTICAL_COUPLING_DC                      1
#define AGMD1_VAL_VERTICAL_COUPLING_GND                     2

/*- Defined values for
    attribute AGMD1_ATTR_TRIGGER_TYPE */

#define AGMD1_VAL_EDGE_TRIGGER                              1
#define AGMD1_VAL_TV_TRIGGER                                5
#define AGMD1_VAL_IMMEDIATE_TRIGGER                         1001
#define AGMD1_VAL_MAGNITUDE_TRIGGER                         1002

/*- Defined values for
    attribute AGMD1_ATTR_TEMPERATURE_UNITS
    parameter Units in function AgMD1_ConfigureTemperatureUnits */

#define AGMD1_VAL_CELSIUS                                   0
#define AGMD1_VAL_FAHRENHEIT                                1
#define AGMD1_VAL_KELVIN                                    2

/*- Defined values for
    attribute AGMD1_ATTR_REFERENCE_OSCILLATOR_SOURCE */

#define AGMD1_VAL_REFERENCE_OSCILLATOR_SOURCE_INTERNAL      0
#define AGMD1_VAL_REFERENCE_OSCILLATOR_SOURCE_EXTERNAL      1
#define AGMD1_VAL_REFERENCE_OSCILLATOR_SOURCE_AXIE_CLK100   4
#define AGMD1_VAL_REFERENCE_OSCILLATOR_SOURCE_PXI_CLK10     2
#define AGMD1_VAL_REFERENCE_OSCILLATOR_SOURCE_PXIE_CLK100   3

/*- Defined values for
    attribute AGMD1_ATTR_SAMPLE_CLOCK_SOURCE */

#define AGMD1_VAL_SAMPLE_CLOCK_SOURCE_INTERNAL              0
#define AGMD1_VAL_SAMPLE_CLOCK_SOURCE_EXTERNAL              1

/*- Defined values for
    attribute AGMD1_ATTR_TRIGGER_SLOPE
	attribute AGMD1_ATTR_PXI_TRIG_SLOPE
	attribute AGMD1_ATTR_TRIGGER_MAGNITUDE_SLOPE
	parameter Slope in function AgMD1_ConfigureEdgeTriggerSource
	parameter Slope in function AgMD1_TriggerMagnitudeConfigure */

#define AGMD1_VAL_NEGATIVE                                  0
#define AGMD1_VAL_POSITIVE                                  1
#define AGMD1_VAL_TRIGGER_SLOPE_NEGATIVE                    0
#define AGMD1_VAL_TRIGGER_SLOPE_POSITIVE                    1

/*- Defined values for
    attribute AGMD1_ATTR_TV_TRIGGER_POLARITY
    parameter Polarity in function AgMD1_ConfigureTVTriggerSource */

#define AGMD1_VAL_TV_POSITIVE                               1
#define AGMD1_VAL_TV_NEGATIVE                               2

/*- Defined values for
    attribute AGMD1_ATTR_TV_TRIGGER_EVENT
    parameter Event in function AgMD1_ConfigureTVTriggerSource */

#define AGMD1_VAL_TV_EVENT_FIELD1                           1
#define AGMD1_VAL_TV_EVENT_FIELD2                           2
#define AGMD1_VAL_TV_EVENT_LINE_NUMBER                      5

/*- Defined values for
    attribute AGMD1_ATTR_TV_TRIGGER_SIGNAL_FORMAT
    parameter SignalFormat in function AgMD1_ConfigureTVTriggerSource */

#define AGMD1_VAL_NTSC                                      1
#define AGMD1_VAL_PAL                                       2
#define AGMD1_VAL_SECAM                                     3

/*- Defined values for
    attribute AGMD1_ATTR_TRIGGER_COUPLING */

#define AGMD1_VAL_TRIGGER_COUPLING_AC                       0
#define AGMD1_VAL_TRIGGER_COUPLING_DC                       1
#define AGMD1_VAL_TRIGGER_COUPLING_HF_REJECT                2

/*- Defined values for */

#define AGMD1_VAL_MAX_TIME_IMMEDIATE                        0
#define AGMD1_VAL_MAX_TIME_INFINITE                         -1

/*- Defined values for
    attribute AGMD1_ATTR_TRIGGER_SOURCE_OPERATOR
    parameter Operator in function AgMD1_ConfigureMultiTrigger */

#define AGMD1_VAL_TRIGGER_SOURCE_OPERATOR_AND               0
#define AGMD1_VAL_TRIGGER_SOURCE_OPERATOR_OR                1
#define AGMD1_VAL_TRIGGER_SOURCE_OPERATOR_NONE              2
#define AGMD1_VAL_TRIGGER_SOURCE_OPERATOR_NAND              1001
#define AGMD1_VAL_TRIGGER_SOURCE_OPERATOR_NOR               1002

/*- Defined values for
    attribute AGMD1_ATTR_SAMPLE_MODE
    parameter SampleMode in function AgMD1_ConfigureSampleMode */

#define AGMD1_VAL_SAMPLE_MODE_REAL_TIME                     0
#define AGMD1_VAL_SAMPLE_MODE_EQUIVALENT_TIME               1

/*- Defined values for
    attribute AGMD1_ATTR_TRIGGER_MODIFIER
    parameter TriggerModifier in function AgMD1_ConfigureTriggerModifier */

#define AGMD1_VAL_TRIGGER_MODIFIER_NONE                     1
#define AGMD1_VAL_TRIGGER_MODIFIER_AUTO                     2

/*- Defined values for
    attribute AGMD1_ATTR_IS_IDLE
    attribute AGMD1_ATTR_IS_MEASURING
    attribute AGMD1_ATTR_IS_WAITING_FOR_ARM
    attribute AGMD1_ATTR_IS_WAITING_FOR_TRIGGER
    parameter Status in function AgMD1_IsIdle
    parameter Status in function AgMD1_IsMeasuring
    parameter Status in function AgMD1_IsWaitingForArm
    parameter Status in function AgMD1_IsWaitingForTrigger */

#define AGMD1_VAL_ACQUISITION_STATUS_RESULT_TRUE            1
#define AGMD1_VAL_ACQUISITION_STATUS_RESULT_FALSE           2
#define AGMD1_VAL_ACQUISITION_STATUS_RESULT_UNKNOWN         3

/*- Defined values for
    parameter MeasFunction in function AgMD1_MeasurementFetchWaveformMeasurement
    parameter MeasFunction in function AgMD1_MeasurementReadWaveformMeasurement */

#define AGMD1_VAL_MEASUREMENT_RISE_TIME                     0
#define AGMD1_VAL_MEASUREMENT_FALL_TIME                     1
#define AGMD1_VAL_MEASUREMENT_FREQUENCY                     2
#define AGMD1_VAL_MEASUREMENT_PERIOD                        3
#define AGMD1_VAL_MEASUREMENT_VOLTAGERMS                    4
#define AGMD1_VAL_MEASUREMENT_VOLTAGE_PEAK_TO_PEAK          5
#define AGMD1_VAL_MEASUREMENT_VOLTAGE_MAX                   6
#define AGMD1_VAL_MEASUREMENT_VOLTAGE_MIN                   7
#define AGMD1_VAL_MEASUREMENT_VOLTAGE_HIGH                  8
#define AGMD1_VAL_MEASUREMENT_VOLTAGE_LOW                   9
#define AGMD1_VAL_MEASUREMENT_VOLTAGE_AVERAGE               10
#define AGMD1_VAL_MEASUREMENT_WIDTH_NEG                     11
#define AGMD1_VAL_MEASUREMENT_WIDTH_POS                     12
#define AGMD1_VAL_MEASUREMENT_DUTY_CYCLE_NEG                13
#define AGMD1_VAL_MEASUREMENT_DUTY_CYCLE_POS                14
#define AGMD1_VAL_MEASUREMENT_AMPLITUDE                     15
#define AGMD1_VAL_MEASUREMENT_VOLTAGE_CYCLERMS              16
#define AGMD1_VAL_MEASUREMENT_VOLTAGE_CYCLE_AVERAGE         17
#define AGMD1_VAL_MEASUREMENT_OVER_SHOOT                    18
#define AGMD1_VAL_MEASUREMENT_PRESHOOT                      19
#define AGMD1_VAL_MEASUREMENT_DC_OFFSET                     20
#define AGMD1_VAL_MEASUREMENT_PHASE_ANGLE                   21
#define AGMD1_VAL_MEASUREMENT_PULSE_REPETITION_FREQUENCY    22

/*- Defined values for
    parameter Type in function AgMD1_CalibrationSelfCalibrate */

#define AGMD1_VAL_CALIBRATE_TYPE_FULL                       0
#define AGMD1_VAL_CALIBRATE_TYPE_CHANNEL_CONFIGURATION      1
#define AGMD1_VAL_CALIBRATE_TYPE_EXT_CLOCK_TIMING           2
#define AGMD1_VAL_CALIBRATE_TYPE_CURRENT_FREQUENCY          3
#define AGMD1_VAL_CALIBRATE_TYPE_FAST                       4

/*- Defined values for
    attribute AGMD1_ATTR_MEASUREMENT_WAVEFORM_TYPE */

#define AGMD1_VAL_MEASUREMENT_WAVEFORM_TYPE_UNKNOWN         0
#define AGMD1_VAL_MEASUREMENT_WAVEFORM_TYPE_SINE            1
#define AGMD1_VAL_MEASUREMENT_WAVEFORM_TYPE_SQUARE          2
#define AGMD1_VAL_MEASUREMENT_WAVEFORM_TYPE_RAMP            3
#define AGMD1_VAL_MEASUREMENT_WAVEFORM_TYPE_PULSEDDC        4
#define AGMD1_VAL_MEASUREMENT_WAVEFORM_TYPE_STEP            5

/*- Defined values for
	attribute AGMD1_ATTR_ACQUISITION_MODE */

#define AGMD1_VAL_ACQUISITION_MODE_NORMAL                   0
#define AGMD1_VAL_ACQUISITION_MODE_DOWNCONVERTER            12
#define AGMD1_VAL_ACQUISITION_MODE_COUNTER                  6
#define AGMD1_VAL_ACQUISITION_MODE_USER_FDK                 15

/*- Defined values for
	attribute AGMD1_ATTR_COUNTER_MODE */

#define AGMD1_VAL_COUNTER_MODE_FREQUENCY                    0
#define AGMD1_VAL_COUNTER_MODE_PERIOD                       1
#define AGMD1_VAL_COUNTER_MODE_TOTALIZE_TIME                2
#define AGMD1_VAL_COUNTER_MODE_TOTALIZE_GATE                3

/*- Defined values for
	attribute AGMD1_ATTR_DELAY_CONTROL_RANGE */

#define AGMD1_VAL_DELAY_CONTROL_RANGE_DEFAULT               0
#define AGMD1_VAL_DELAY_CONTROL_RANGE_EXTENDED              1

/*- Defined values for 
	attribute AGMD1_ATTR_USER_CONTROL_COUNTERS_MODE */

#define AGMD1_VAL_USER_CONTROL_COUNTERS_MODE_NORMAL         0
#define AGMD1_VAL_USER_CONTROL_COUNTERS_MODE_STREAMING      1

/*- Defined values for 
	attribute AGMD1_ATTR_USER_CONTROL_INTER_SEGMENT_DELAY_MODE */

#define AGMD1_VAL_USER_CONTROL_INTER_SEGMENT_DELAY_MODE_SOFTWARE      0
#define AGMD1_VAL_USER_CONTROL_INTER_SEGMENT_DELAY_MODE_USER_FIRMWARE 1

/*- Defined values for 
	attribute AGMD1_ATTR_USER_CONTROL_TRIGGER_ENABLE_SOURCE */

#define AGMD1_VAL_USER_CONTROL_TRIGGER_ENABLE_SOURCE_SEGMENTATION  0
#define AGMD1_VAL_USER_CONTROL_TRIGGER_ENABLE_SOURCE_USER_FIRMWARE 1

/*- Defined values for 
	parameter ProcessingType in function AgMD1_UserControlStartProcessing
	parameter ProcessingType in function AgMD1_UserControlStopProcessing */

#define AGMD1_VAL_USER_CONTROL_PROCESSING_TYPE1             1
#define AGMD1_VAL_USER_CONTROL_PROCESSING_TYPE2             2

/*- Defined values for 
	attribute AGMD1_ATTR_LOGIC_DEVICE_STREAM_MODE */

#define AGMD1_VAL_LOGIC_DEVICE_STREAM_MODE_ACQUISITION      0
#define AGMD1_VAL_LOGIC_DEVICE_STREAM_MODE_EMULATION        1
#define AGMD1_VAL_LOGIC_DEVICE_STREAM_MODE_SEQUENTIAL       2

/*- Defined values for 
	attribute AGMD1_ATTR_LOGIC_DEVICE_MEMORY_BANK_ACCESS_CONTROL */

#define AGMD1_VAL_LOGIC_DEVICE_MEMORY_BANK_ACCESS_CONTROL_SOFTWARE      0
#define AGMD1_VAL_LOGIC_DEVICE_MEMORY_BANK_ACCESS_CONTROL_USER_FIRMWARE 1

/*- Defined values for 
	attribute AGMD1_ATTR_LOGIC_DEVICE_MEMORY_BANK_ACCESS_MODE */

#define AGMD1_VAL_LOGIC_DEVICE_MEMORY_BANK_ACCESS_MODE_WRITE 0
#define AGMD1_VAL_LOGIC_DEVICE_MEMORY_BANK_ACCESS_MODE_READ  1

/*- Defined values for 
	parameter CoreId in function AgMD1_LogicDeviceGetCoreVersion */

#define AGMD1_VAL_LOGIC_DEVICE_CORE_PCIE                    0
#define AGMD1_VAL_LOGIC_DEVICE_CORE_DDR3A                   1
#define AGMD1_VAL_LOGIC_DEVICE_CORE_DDR3B                   2
#define AGMD1_VAL_LOGIC_DEVICE_CORE_ACQUISITION_PATH        3
#define AGMD1_VAL_LOGIC_DEVICE_CORE_IFDL_UP                 4
#define AGMD1_VAL_LOGIC_DEVICE_CORE_IFDL_DOWN               5
#define AGMD1_VAL_LOGIC_DEVICE_CORE_IFDL_CONTROL            6
#define AGMD1_VAL_LOGIC_DEVICE_CORE_QDR2                    7
#define AGMD1_VAL_LOGIC_DEVICE_CORE_ADC_RECEIVER            8
#define AGMD1_VAL_LOGIC_DEVICE_CORE_STREAM_PREPARE          9
#define AGMD1_VAL_LOGIC_DEVICE_CORE_TRIGGER_MANAGER         10

/*- Defined values for 
	attribute AGMD1_ATTR_USER_CONTROL_TRIGGER_SELECTION */

#define AGMD1_VAL_USER_CONTROL_TRIGGER_SELECTION_ANALOG     0
#define AGMD1_VAL_USER_CONTROL_TRIGGER_SELECTION_DIGITAL    1


/****************************************************************************
 *---------------- Instrument Driver Function Declarations -----------------*
 ****************************************************************************/

/*- AgMD1 */

ViStatus _VI_FUNC AgMD1_init ( ViRsrc ResourceName, ViBoolean IdQuery, ViBoolean Reset, ViSession* Vi );
ViStatus _VI_FUNC AgMD1_close ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_InitWithOptions ( ViRsrc ResourceName, ViBoolean IdQuery, ViBoolean Reset, ViConstString OptionsString, ViSession* Vi );

/*- Utility */

ViStatus _VI_FUNC AgMD1_revision_query ( ViSession Vi, ViChar DriverRev[], ViChar InstrRev[] );
ViStatus _VI_FUNC AgMD1_error_message ( ViSession Vi, ViStatus ErrorCode, ViChar ErrorMessage[] );
ViStatus _VI_FUNC AgMD1_GetError ( ViSession Vi, ViStatus* ErrorCode, ViInt32 ErrorDescriptionBufferSize, ViChar ErrorDescription[] );
ViStatus _VI_FUNC AgMD1_ClearError ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_ClearInterchangeWarnings ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_GetNextCoercionRecord ( ViSession Vi, ViInt32 CoercionRecordBufferSize, ViChar CoercionRecord[] );
ViStatus _VI_FUNC AgMD1_GetNextInterchangeWarning ( ViSession Vi, ViInt32 InterchangeWarningBufferSize, ViChar InterchangeWarning[] );
ViStatus _VI_FUNC AgMD1_InvalidateAllAttributes ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_ResetInterchangeCheck ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_Disable ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_error_query ( ViSession Vi, ViInt32* ErrorCode, ViChar ErrorMessage[] );
ViStatus _VI_FUNC AgMD1_LockSession ( ViSession Vi, ViBoolean* CallerHasLock );
ViStatus _VI_FUNC AgMD1_reset ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_ResetWithDefaults ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_self_test ( ViSession Vi, ViInt16* TestResult, ViChar TestMessage[] );
ViStatus _VI_FUNC AgMD1_UnlockSession ( ViSession Vi, ViBoolean* CallerHasLock );

/*- Temperature */

ViStatus _VI_FUNC AgMD1_ConfigureTemperatureUnits ( ViSession Vi, ViInt32 Units );
ViStatus _VI_FUNC AgMD1_QueryBoardTemperature ( ViSession Vi, ViReal64* Temperature );
ViStatus _VI_FUNC AgMD1_QueryChannelTemperature ( ViSession Vi, ViConstString Channel, ViReal64* Temperature );

/*- Attribute Accessors */

ViStatus _VI_FUNC AgMD1_GetAttributeViInt32 ( ViSession Vi, ViConstString RepCapIdentifier, ViAttr AttributeID, ViInt32* AttributeValue );
ViStatus _VI_FUNC AgMD1_GetAttributeViReal64 ( ViSession Vi, ViConstString RepCapIdentifier, ViAttr AttributeID, ViReal64* AttributeValue );
ViStatus _VI_FUNC AgMD1_GetAttributeViBoolean ( ViSession Vi, ViConstString RepCapIdentifier, ViAttr AttributeID, ViBoolean* AttributeValue );
ViStatus _VI_FUNC AgMD1_GetAttributeViSession ( ViSession Vi, ViConstString RepCapIdentifier, ViAttr AttributeID, ViSession* AttributeValue );
ViStatus _VI_FUNC AgMD1_GetAttributeViString ( ViSession Vi, ViConstString RepCapIdentifier, ViAttr AttributeID, ViInt32 AttributeValueBufferSize, ViChar AttributeValue[] );
ViStatus _VI_FUNC AgMD1_SetAttributeViInt32 ( ViSession Vi, ViConstString RepCapIdentifier, ViAttr AttributeID, ViInt32 AttributeValue );
ViStatus _VI_FUNC AgMD1_SetAttributeViReal64 ( ViSession Vi, ViConstString RepCapIdentifier, ViAttr AttributeID, ViReal64 AttributeValue );
ViStatus _VI_FUNC AgMD1_SetAttributeViBoolean ( ViSession Vi, ViConstString RepCapIdentifier, ViAttr AttributeID, ViBoolean AttributeValue );
ViStatus _VI_FUNC AgMD1_SetAttributeViSession ( ViSession Vi, ViConstString RepCapIdentifier, ViAttr AttributeID, ViSession AttributeValue );
ViStatus _VI_FUNC AgMD1_SetAttributeViString ( ViSession Vi, ViConstString RepCapIdentifier, ViAttr AttributeID, ViConstString AttributeValue );
ViStatus _VI_FUNC AgMD1_GetAttributeViInt64 ( ViSession Vi, ViConstString RepCapIdentifier, ViInt32 AttributeID, ViInt64* AttributeValue );
ViStatus _VI_FUNC AgMD1_SetAttributeViInt64 ( ViSession Vi, ViConstString RepCapIdentifier, ViInt32 AttributeID, ViInt64 AttributeValue );

/*- Calibration */

ViStatus _VI_FUNC AgMD1_SelfCalibrate ( ViSession Vi );

/*- Acquisition */

ViStatus _VI_FUNC AgMD1_ConfigureAcquisition ( ViSession Vi, ViInt64 NumRecordsToAcquire, ViInt64 RecordSize, ViReal64 SampleRate );
ViStatus _VI_FUNC AgMD1_ConfigureSampleMode ( ViSession Vi, ViInt32 SampleMode );

/*- Downconversion */

ViStatus _VI_FUNC AgMD1_ConfigureDownconversion ( ViSession Vi, ViConstString ChannelName, ViBoolean Enabled, ViReal64 CenterFrequency );

/*- Channel */

ViStatus _VI_FUNC AgMD1_ConfigureChannel ( ViSession Vi, ViConstString ChannelName, ViReal64 Range, ViReal64 Offset, ViInt32 Coupling, ViBoolean Enabled );
ViStatus _VI_FUNC AgMD1_GetChannelName ( ViSession Vi, ViInt32 Index, ViInt32 NameBufferSize, ViChar Name[] );

/*- Reference Oscillator */

ViStatus _VI_FUNC AgMD1_ConfigureReferenceOscillatorOutputEnabled ( ViSession Vi, ViBoolean Enabled );

/*- Trigger */

ViStatus _VI_FUNC AgMD1_ConfigureEdgeTriggerSource ( ViSession Vi, ViConstString Source, ViReal64 Level, ViInt32 Slope );
ViStatus _VI_FUNC AgMD1_ConfigureTVTriggerSource ( ViSession Vi, ViConstString Source, ViInt32 SignalFormat, ViInt32 Event, ViInt32 Polarity );
ViStatus _VI_FUNC AgMD1_ConfigureMultiTrigger ( ViSession Vi, ViConstString SourceList, ViInt32 Operator );
ViStatus _VI_FUNC AgMD1_GetTriggerSourceName ( ViSession Vi, ViInt32 Index, ViInt32 NameBufferSize, ViChar Name[] );
ViStatus _VI_FUNC AgMD1_ConfigureTriggerModifier ( ViSession Vi, ViInt32 TriggerModifier );

/*- Waveform Acquisition */

ViStatus _VI_FUNC AgMD1_ReadWaveformInt16 ( ViSession Vi, ViConstString ChannelName, ViInt32 MaxTimeMilliseconds, ViInt64 WaveformArraySize, ViInt16 WaveformArray[], ViInt64* ActualPoints, ViInt64* FirstValidPoint, ViReal64* InitialXOffset, ViReal64* InitialXTimeSeconds, ViReal64* InitialXTimeFraction, ViReal64* XIncrement, ViReal64* ScaleFactor, ViReal64* ScaleOffset );
ViStatus _VI_FUNC AgMD1_ReadWaveformInt32 ( ViSession Vi, ViConstString ChannelName, ViInt32 MaxTimeMilliseconds, ViInt64 WaveformArraySize, ViInt32 WaveformArray[], ViInt64* ActualPoints, ViInt64* FirstValidPoint, ViReal64* InitialXOffset, ViReal64* InitialXTimeSeconds, ViReal64* InitialXTimeFraction, ViReal64* XIncrement, ViReal64* ScaleFactor, ViReal64* ScaleOffset );
ViStatus _VI_FUNC AgMD1_ReadWaveformInt8 ( ViSession Vi, ViConstString ChannelName, ViInt32 MaxTimeMilliseconds, ViInt64 WaveformArraySize, ViChar WaveformArray[], ViInt64* ActualPoints, ViInt64* FirstValidPoint, ViReal64* InitialXOffset, ViReal64* InitialXTimeSeconds, ViReal64* InitialXTimeFraction, ViReal64* XIncrement, ViReal64* ScaleFactor, ViReal64* ScaleOffset );
ViStatus _VI_FUNC AgMD1_ReadWaveformReal64 ( ViSession Vi, ViConstString ChannelName, ViInt32 MaxTimeMilliseconds, ViInt64 WaveformArraySize, ViReal64 WaveformArray[], ViInt64* ActualPoints, ViInt64* FirstValidPoint, ViReal64* InitialXOffset, ViReal64* InitialXTimeSeconds, ViReal64* InitialXTimeFraction, ViReal64* XIncrement );

/*- Low-Level Acquisition */

ViStatus _VI_FUNC AgMD1_InitiateAcquisition ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_Abort ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_SendSoftwareArm ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_SendSoftwareTrigger ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_WaitForAcquisitionComplete ( ViSession Vi, ViInt32 MaxTimeMilliseconds );
ViStatus _VI_FUNC AgMD1_FetchWaveformReal64 ( ViSession Vi, ViConstString ChannelName, ViInt64 WaveformArraySize, ViReal64 WaveformArray[], ViInt64* ActualPoints, ViInt64* FirstValidPoint, ViReal64* InitialXOffset, ViReal64* InitialXTimeSeconds, ViReal64* InitialXTimeFraction, ViReal64* XIncrement );
ViStatus _VI_FUNC AgMD1_FetchWaveformInt8 ( ViSession Vi, ViConstString ChannelName, ViInt64 WaveformArraySize, ViChar WaveformArray[], ViInt64* ActualPoints, ViInt64* FirstValidPoint, ViReal64* InitialXOffset, ViReal64* InitialXTimeSeconds, ViReal64* InitialXTimeFraction, ViReal64* XIncrement, ViReal64* ScaleFactor, ViReal64* ScaleOffset );
ViStatus _VI_FUNC AgMD1_FetchWaveformInt16 ( ViSession Vi, ViConstString ChannelName, ViInt64 WaveformArraySize, ViInt16 WaveformArray[], ViInt64* ActualPoints, ViInt64* FirstValidPoint, ViReal64* InitialXOffset, ViReal64* InitialXTimeSeconds, ViReal64* InitialXTimeFraction, ViReal64* XIncrement, ViReal64* ScaleFactor, ViReal64* ScaleOffset );
ViStatus _VI_FUNC AgMD1_FetchWaveformInt32 ( ViSession Vi, ViConstString ChannelName, ViInt64 WaveformArraySize, ViInt32 WaveformArray[], ViInt64* ActualPoints, ViInt64* FirstValidPoint, ViReal64* InitialXOffset, ViReal64* InitialXTimeSeconds, ViReal64* InitialXTimeFraction, ViReal64* XIncrement, ViReal64* ScaleFactor, ViReal64* ScaleOffset );
ViStatus _VI_FUNC AgMD1_IsIdle ( ViSession Vi, ViInt32* Status );
ViStatus _VI_FUNC AgMD1_IsMeasuring ( ViSession Vi, ViInt32* Status );
ViStatus _VI_FUNC AgMD1_IsWaitingForArm ( ViSession Vi, ViInt32* Status );
ViStatus _VI_FUNC AgMD1_IsWaitingForTrigger ( ViSession Vi, ViInt32* Status );
ViStatus _VI_FUNC AgMD1_QueryMinWaveformMemory ( ViSession Vi, ViInt32 DataWidth, ViInt64 NumRecords, ViInt64 OffsetWithinRecord, ViInt64 NumPointsPerRecord, ViInt64* NumSamples );

/*- Multi-Record Acquisition */

ViStatus _VI_FUNC AgMD1_FetchMultiRecordWaveformInt16 ( ViSession Vi, ViConstString ChannelName, ViInt64 FirstRecord, ViInt64 NumRecords, ViInt64 OffsetWithinRecord, ViInt64 NumPointsPerRecord, ViInt64 WaveformArrayBufferSize, ViInt16 WaveformArray[], ViInt64* WaveformArrayActualSize, ViInt64* ActualRecords, ViInt64 ActualPoints[], ViInt64 FirstValidPoint[], ViReal64 InitialXOffset[], ViReal64 InitialXTimeSeconds[], ViReal64 InitialXTimeFraction[], ViReal64* XIncrement, ViReal64* ScaleFactor, ViReal64* ScaleOffset );
ViStatus _VI_FUNC AgMD1_FetchMultiRecordWaveformInt8 ( ViSession Vi, ViConstString ChannelName, ViInt64 FirstRecord, ViInt64 NumRecords, ViInt64 OffsetWithinRecord, ViInt64 NumPointsPerRecord, ViInt64 WaveformArrayBufferSize, ViChar WaveformArray[], ViInt64* WaveformArrayActualSize, ViInt64* ActualRecords, ViInt64 ActualPoints[], ViInt64 FirstValidPoint[], ViReal64 InitialXOffset[], ViReal64 InitialXTimeSeconds[], ViReal64 InitialXTimeFraction[], ViReal64* XIncrement, ViReal64* ScaleFactor, ViReal64* ScaleOffset );
ViStatus _VI_FUNC AgMD1_FetchMultiRecordWaveformInt32 ( ViSession Vi, ViConstString ChannelName, ViInt64 FirstRecord, ViInt64 NumRecords, ViInt64 OffsetWithinRecord, ViInt64 NumPointsPerRecord, ViInt64 WaveformArrayBufferSize, ViInt32 WaveformArray[], ViInt64* WaveformArrayActualSize, ViInt64* ActualRecords, ViInt64 ActualPoints[], ViInt64 FirstValidPoint[], ViReal64 InitialXOffset[], ViReal64 InitialXTimeSeconds[], ViReal64 InitialXTimeFraction[], ViReal64* XIncrement, ViReal64* ScaleFactor, ViReal64* ScaleOffset );
ViStatus _VI_FUNC AgMD1_FetchMultiRecordWaveformReal64 ( ViSession Vi, ViConstString ChannelName, ViInt64 FirstRecord, ViInt64 NumRecords, ViInt64 OffsetWithinRecord, ViInt64 NumPointsPerRecord, ViInt64 WaveformArrayBufferSize, ViReal64 WaveformArray[], ViInt64* WaveformArrayActualSize, ViInt64* ActualRecords, ViInt64 ActualPoints[], ViInt64 FirstValidPoint[], ViReal64 InitialXOffset[], ViReal64 InitialXTimeSeconds[], ViReal64 InitialXTimeFraction[], ViReal64* XIncrement );

/*- Acquisition */

ViStatus _VI_FUNC AgMD1_AcquisitionConfigureRecord ( ViSession Vi, ViReal64 TimePerRecord, ViInt32 MinNumPts, ViReal64 AcquisitionStartTime );

/*- UserControl */

ViStatus _VI_FUNC AgMD1_UserControlReadControlRegisterInt32 ( ViSession Vi, ViInt32 Offset, ViInt32* Value );
ViStatus _VI_FUNC AgMD1_UserControlStartProcessing ( ViSession Vi, ViInt32 ProcessingType );
ViStatus _VI_FUNC AgMD1_UserControlStartSegmentation ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_UserControlStopProcessing ( ViSession Vi, ViInt32 ProcessingType );
ViStatus _VI_FUNC AgMD1_UserControlStopSegmentation ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_UserControlWaitForProcessingComplete ( ViSession Vi, ViConstString LogicDevice, ViInt32 ProcessingType, ViInt32 MaxTimeMilliseconds );
ViStatus _VI_FUNC AgMD1_UserControlWriteControlRegisterInt32 ( ViSession Vi, ViInt32 Offset, ViInt32 Value );

/*- SAR */

ViStatus _VI_FUNC AgMD1_SARContinueAcquisition ( ViSession Vi );

/*- Measurement */

ViStatus _VI_FUNC AgMD1_MeasurementFetchWaveformMeasurement ( ViSession Vi, ViConstString Channel, ViInt32 MeasFunction, ViReal64* Measurement );
ViStatus _VI_FUNC AgMD1_MeasurementReadWaveformMeasurement ( ViSession Vi, ViConstString Channel, ViInt32 MeasFunction, ViInt32 MaxTimeMilliseconds, ViReal64* Measurement );

/*- DigitalDownconversion */

ViStatus _VI_FUNC AgMD1_DigitalDownconversionConfigure ( ViSession Vi, ViConstString Channel, ViReal64 DataBandwidth, ViReal64 TriggerBandwidth );

/*- Counter */

ViStatus _VI_FUNC AgMD1_CounterRead ( ViSession Vi, ViConstString Channel, ViReal64* Val );

/*- ReferenceLevel */

ViStatus _VI_FUNC AgMD1_ReferenceLevelConfigure ( ViSession Vi, ViReal64 Low, ViReal64 Mid, ViReal64 High );

/*- System */

ViStatus _VI_FUNC AgMD1_SystemPowerOnAll ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_SystemPowerOffAll ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_SystemSuspendControl ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_SystemResumeControl ( ViSession Vi );

/*- Calibration */

ViStatus _VI_FUNC AgMD1_CalibrationCancel ( ViSession Vi );
ViStatus _VI_FUNC AgMD1_CalibrationLoadCal ( ViSession Vi, ViConstString FilePathName );
ViStatus _VI_FUNC AgMD1_CalibrationSaveCal ( ViSession Vi, ViConstString FilePathName );
ViStatus _VI_FUNC AgMD1_CalibrationCalRequired ( ViSession Vi, ViInt32 Channel, ViBoolean* Val );
ViStatus _VI_FUNC AgMD1_CalibrationSelfCalibrate ( ViSession Vi, ViInt32 Type, ViInt32 Channel );

/*- DelayControl */

ViStatus _VI_FUNC AgMD1_GetDelayControlName ( ViSession Vi, ViInt32 Index, ViInt32 NameBufferSize, ViChar Name[] );

/*- Magnitude */

ViStatus _VI_FUNC AgMD1_TriggerMagnitudeConfigure ( ViSession Vi, ViConstString TriggerSource, ViReal64 Level, ViReal64 Hysteresis, ViInt32 Slope );

/*- Temperature */

ViStatus _VI_FUNC AgMD1_TemperatureGetModuleTemperature ( ViSession Vi, ViInt32 ModuleNumber, ViInt32* Val );

/*- LogicDevice */

ViStatus _VI_FUNC AgMD1_GetLogicDeviceName ( ViSession Vi, ViInt32 Index, ViInt32 NameBufferSize, ViChar Name[] );
ViStatus _VI_FUNC AgMD1_LoadLogicDeviceFromFile ( ViSession Vi, ViConstString LogicDevice, ViConstString Path );
ViStatus _VI_FUNC AgMD1_ReadRegisterInt32 ( ViSession Vi, ViConstString LogicDevice, ViInt32 Offset, ViInt32* Value );
ViStatus _VI_FUNC AgMD1_WriteRegisterInt32 ( ViSession Vi, ViConstString LogicDevice, ViInt32 Offset, ViInt32 Value );
ViStatus _VI_FUNC AgMD1_ReadIndirectInt32 ( ViSession Vi, ViConstString LogicDevice, ViInt32 Id, ViInt32 StartAddress, ViInt32 NumElements, ViInt32 DataBufferSize, ViInt32 Data[], ViInt32* DataActualSize );
ViStatus _VI_FUNC AgMD1_WriteIndirectInt32 ( ViSession Vi, ViConstString LogicDevice, ViInt32 Id, ViInt32 StartAddress, ViInt32 NumElements, ViInt32 DataSize, ViInt32 Data[] );
ViStatus _VI_FUNC AgMD1_LogicDeviceConfigureBufferAddresses ( ViSession Vi, ViConstString LogicDevice, ViInt32 BufferId, ViInt64 Base, ViInt64 Limit );
ViStatus _VI_FUNC AgMD1_LogicDeviceGetBufferAddresses ( ViSession Vi, ViConstString LogicDevice, ViInt32 BufferId, ViInt64* Base, ViInt64* Limit );
ViStatus _VI_FUNC AgMD1_LogicDeviceGetCoreVersion ( ViSession Vi, ViConstString LogicDevice, ViInt32 CoreId, ViInt32* Version, ViInt32 VersionStringBufferSize, ViChar VersionString[] );
ViStatus _VI_FUNC AgMD1_LogicDeviceWriteStreamWaveformInt16 ( ViSession Vi, ViConstString LogicDevice, ViInt32 Stream, ViInt32 SamplesBufferSize, ViInt16 Samples[] );
ViStatus _VI_FUNC AgMD1_LogicDeviceWriteFirmwareStoreFromFile ( ViSession Vi, ViConstString LogicDevice, ViInt32 FirmwareStore, ViConstString Path );
ViStatus _VI_FUNC AgMD1_LogicDevicePlaybackStreamWaveforms ( ViSession Vi, ViConstString LogicDevice );

/*- LogicDeviceIFDL */

ViStatus _VI_FUNC AgMD1_GetLogicDeviceIFDLName ( ViSession Vi, ViConstString LogicDevice, ViInt32 Index, ViInt32 NameBufferSize, ViChar Name[] );
ViStatus _VI_FUNC AgMD1_LogicDeviceIFDLStartSelfTestReceiver ( ViSession Vi, ViConstString LogicDeviceIFDL );
ViStatus _VI_FUNC AgMD1_LogicDeviceIFDLStartSelfTestTransmitter ( ViSession Vi, ViConstString LogicDeviceIFDL );
ViStatus _VI_FUNC AgMD1_LogicDeviceIFDLCheckSelfTestReceiver ( ViSession Vi, ViConstString LogicDeviceIFDL, ViBoolean* Val );

/*- LogicDeviceMemoryBank */

ViStatus _VI_FUNC AgMD1_GetLogicDeviceMemoryBankName ( ViSession Vi, ViConstString LogicDevice, ViInt32 Index, ViInt32 NameBufferSize, ViChar Name[] );

/*- ModuleSynchronization */

ViStatus _VI_FUNC AgMD1_ModuleSynchronizationConfigureSlaves ( ViSession Vi, ViInt32 SlavesCount, ViInt32 Slaves[] );
ViStatus _VI_FUNC AgMD1_ModuleSynchronizationGetSlaves ( ViSession Vi, ViInt32 SlavesBufferSize, ViInt32 Slaves[], ViInt32* ActualSlaves );


/****************************************************************************
 *----------------- Instrument Error And Completion Codes ------------------*
 ****************************************************************************/
#ifndef _IVIC_ERROR_BASE_DEFINES_
#define _IVIC_ERROR_BASE_DEFINES_

#define IVIC_WARN_BASE                           (0x3FFA0000)
#define IVIC_CROSS_CLASS_WARN_BASE               (IVIC_WARN_BASE + 0x1000)
#define IVIC_CLASS_WARN_BASE                     (IVIC_WARN_BASE + 0x2000)
#define IVIC_SPECIFIC_WARN_BASE                  (IVIC_WARN_BASE + 0x4000)

#define IVIC_ERROR_BASE                          (0x3FFA0000 - 0x40000000 - 0x40000000)
#define IVIC_CROSS_CLASS_ERROR_BASE              (IVIC_ERROR_BASE + 0x1000)
#define IVIC_CLASS_ERROR_BASE                    (IVIC_ERROR_BASE + 0x2000)
#define IVIC_SPECIFIC_ERROR_BASE                 (IVIC_ERROR_BASE + 0x4000)
#define IVIC_LXISYNC_ERROR_BASE                  (IVIC_ERROR_BASE + 0x2000)

#define IVIC_ERROR_INVALID_ATTRIBUTE             (IVIC_ERROR_BASE + 0x000C)
#define IVIC_ERROR_TYPES_DO_NOT_MATCH            (IVIC_ERROR_BASE + 0x0015)
#define IVIC_ERROR_IVI_ATTR_NOT_WRITABLE         (IVIC_ERROR_BASE + 0x000D)
#define IVIC_ERROR_IVI_ATTR_NOT_READABLE         (IVIC_ERROR_BASE + 0x000E)
#define IVIC_ERROR_INVALID_SESSION_HANDLE        (IVIC_ERROR_BASE + 0x1190)

#endif


#define AGMD1_ERROR_CANNOT_RECOVER                          (IVIC_ERROR_BASE + 0x0000)
#define AGMD1_ERROR_INSTRUMENT_STATUS                       (IVIC_ERROR_BASE + 0x0001)
#define AGMD1_ERROR_CANNOT_OPEN_FILE                        (IVIC_ERROR_BASE + 0x0002)
#define AGMD1_ERROR_READING_FILE                            (IVIC_ERROR_BASE + 0x0003)
#define AGMD1_ERROR_WRITING_FILE                            (IVIC_ERROR_BASE + 0x0004)
#define AGMD1_ERROR_INVALID_PATHNAME                        (IVIC_ERROR_BASE + 0x000B)
#define AGMD1_ERROR_INVALID_VALUE                           (IVIC_ERROR_BASE + 0x0010)
#define AGMD1_ERROR_FUNCTION_NOT_SUPPORTED                  (IVIC_ERROR_BASE + 0x0011)
#define AGMD1_ERROR_ATTRIBUTE_NOT_SUPPORTED                 (IVIC_ERROR_BASE + 0x0012)
#define AGMD1_ERROR_VALUE_NOT_SUPPORTED                     (IVIC_ERROR_BASE + 0x0013)
#define AGMD1_ERROR_NOT_INITIALIZED                         (IVIC_ERROR_BASE + 0x001D)
#define AGMD1_ERROR_UNKNOWN_CHANNEL_NAME                    (IVIC_ERROR_BASE + 0x0020)
#define AGMD1_ERROR_TOO_MANY_OPEN_FILES                     (IVIC_ERROR_BASE + 0x0023)
#define AGMD1_ERROR_CHANNEL_NAME_REQUIRED                   (IVIC_ERROR_BASE + 0x0044)
#define AGMD1_ERROR_MISSING_OPTION_NAME                     (IVIC_ERROR_BASE + 0x0049)
#define AGMD1_ERROR_MISSING_OPTION_VALUE                    (IVIC_ERROR_BASE + 0x004A)
#define AGMD1_ERROR_BAD_OPTION_NAME                         (IVIC_ERROR_BASE + 0x004B)
#define AGMD1_ERROR_BAD_OPTION_VALUE                        (IVIC_ERROR_BASE + 0x004C)
#define AGMD1_ERROR_OUT_OF_MEMORY                           (IVIC_ERROR_BASE + 0x0056)
#define AGMD1_ERROR_OPERATION_PENDING                       (IVIC_ERROR_BASE + 0x0057)
#define AGMD1_ERROR_NULL_POINTER                            (IVIC_ERROR_BASE + 0x0058)
#define AGMD1_ERROR_UNEXPECTED_RESPONSE                     (IVIC_ERROR_BASE + 0x0059)
#define AGMD1_ERROR_FILE_NOT_FOUND                          (IVIC_ERROR_BASE + 0x005B)
#define AGMD1_ERROR_INVALID_FILE_FORMAT                     (IVIC_ERROR_BASE + 0x005C)
#define AGMD1_ERROR_STATUS_NOT_AVAILABLE                    (IVIC_ERROR_BASE + 0x005D)
#define AGMD1_ERROR_ID_QUERY_FAILED                         (IVIC_ERROR_BASE + 0x005E)
#define AGMD1_ERROR_RESET_FAILED                            (IVIC_ERROR_BASE + 0x005F)
#define AGMD1_ERROR_RESOURCE_UNKNOWN                        (IVIC_ERROR_BASE + 0x0060)
#define AGMD1_ERROR_ALREADY_INITIALIZED                     (IVIC_ERROR_BASE + 0x0061)
#define AGMD1_ERROR_CANNOT_CHANGE_SIMULATION_STATE          (IVIC_ERROR_BASE + 0x0062)
#define AGMD1_ERROR_INVALID_NUMBER_OF_LEVELS_IN_SELECTOR    (IVIC_ERROR_BASE + 0x0063)
#define AGMD1_ERROR_INVALID_RANGE_IN_SELECTOR               (IVIC_ERROR_BASE + 0x0064)
#define AGMD1_ERROR_UNKOWN_NAME_IN_SELECTOR                 (IVIC_ERROR_BASE + 0x0065)
#define AGMD1_ERROR_BADLY_FORMED_SELECTOR                   (IVIC_ERROR_BASE + 0x0066)
#define AGMD1_ERROR_UNKNOWN_PHYSICAL_IDENTIFIER             (IVIC_ERROR_BASE + 0x0067)



#define AGMD1_SUCCESS                                       0
#define AGMD1_WARN_NSUP_ID_QUERY                            (IVIC_WARN_BASE + 0x0065)
#define AGMD1_WARN_NSUP_RESET                               (IVIC_WARN_BASE + 0x0066)
#define AGMD1_WARN_NSUP_SELF_TEST                           (IVIC_WARN_BASE + 0x0067)
#define AGMD1_WARN_NSUP_ERROR_QUERY                         (IVIC_WARN_BASE + 0x0068)
#define AGMD1_WARN_NSUP_REV_QUERY                           (IVIC_WARN_BASE + 0x0069)



#define AGMD1_ERROR_IO_GENERAL                              (IVIC_SPECIFIC_ERROR_BASE + 0x0214)
#define AGMD1_ERROR_IO_TIMEOUT                              (IVIC_SPECIFIC_ERROR_BASE + 0x0215)
#define AGMD1_ERROR_MODEL_NOT_SUPPORTED                     (IVIC_SPECIFIC_ERROR_BASE + 0x0216)
#define AGMD1_ERROR_PERSONALITY_NOT_ACTIVE                  (IVIC_SPECIFIC_ERROR_BASE + 0x0211)
#define AGMD1_ERROR_PERSONALITY_NOT_LICENSED                (IVIC_SPECIFIC_ERROR_BASE + 0x0213)
#define AGMD1_ERROR_PERSONALITY_NOT_INSTALLED               (IVIC_SPECIFIC_ERROR_BASE + 0x0212)
#define AGMD1_ERROR_CHANNEL_NOT_ENABLED                     (IVIC_CLASS_ERROR_BASE + 0x0001)
#define AGMD1_ERROR_MAX_TIME_EXCEEDED                       (IVIC_CLASS_ERROR_BASE + 0x0002)
#define AGMD1_ERROR_TRIGGER_NOT_SOFTWARE                    (IVIC_CROSS_CLASS_ERROR_BASE + 0x0001)
#define AGMD1_ERROR_ARM_NOT_SOFTWARE                        (IVIC_CLASS_ERROR_BASE + 0x0003)
#define AGMD1_ERROR_INCOMPATIBLE_FETCH                      (IVIC_CLASS_ERROR_BASE + 0x0004)
#define AGMD1_ERROR_MODULE_API                              (IVIC_SPECIFIC_ERROR_BASE + 0x0217)
#define AGMD1_ERROR_NOT_SUPPORTED_IN_CURRENT_STATE          (IVIC_SPECIFIC_ERROR_BASE + 0x0218)



#define AGMD1_WARN_CALIBRATED_WITH_INTERNAL_CLOCK           (IVIC_SPECIFIC_WARN_BASE + 0x0001)



#if defined(__cplusplus) || defined(__cplusplus__)
}
#endif


#endif // sentry


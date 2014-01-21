#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <string>

/*! 
 *  \file 
 *  \brief This File holds all configurations and constants. 
 */

/*! \var SAMPLE_COUNT 
 *  \brief Number of samples per event */
const unsigned int SAMPLE_COUNT = 1000;
/*! \var CHUNK_COUNT 
 *  \brief Number of events copied to the GPU in one step */
const unsigned int CHUNK_COUNT = 100;
/*! \var CHUNK_BUFFER_COUNT
 *  \brief  Number of chunks in the input buffer */
const unsigned int CHUNK_BUFFER_COUNT = 1024;
/*! \var FILTER_MODE
 *  \brief  Interpolation mode */
const cudaTextureFilterMode FILTER_MODE = cudaFilterModeLinear;

const std::string FILENAME_TESTFILE = "../data/Al_25keV-259.cdb";
const unsigned int SAMPLE_COUNT_TESTFILE = 1000;
const unsigned int SEGMENT_COUNT_TESTFILE = 1;
const unsigned int WAVEFORM_COUNT_TESTFILE = 100000;



/*!
 * \var MAXCOUNTDATA
 * \brief max. number of samples per event for compute capability 2.0 or higher - currently ca. 2450 is max. because (COUNTPARAM + 2) * MAXCOUNTDATA * sizeof(float) = 48 kB (= max. shared memory); for compute capability 1.x - currently ca. 800 is max. because (COUNTPARAM + 2) * MAXCOUNTDATA * sizeof(float) = 16 kB (= max. shared memory)
*/
const unsigned int MAXCOUNTDATA = 800; //1000 //2450

/*!
 * \var MAXCALL
 * \brief max. calls for Levenberg Marquardt until stops
*/
const unsigned int MAXCALL = 100;

/*!
 * \var FITVALUETHRESHOLD
 * \brief threshold between min (0.0) and max (1.0) value to define the data using interval to calculate the fit function
*/
const float FITVALUETHRESHOLD = 0.0; //0.5

/*!
 * \var STARTENDPROPORTION
 * \brief proportion of countData for calculating the average of start/end value (e. g. 0.1 means average of the first 10% of data for start value and the last 10% for end value)
*/
const float STARTENDPROPORTION = 0.01;

/*!
 * \var COUNTPARAM
 * \brief number of parameters for the fit function
*/
const unsigned int COUNTPARAM = 3;

#endif

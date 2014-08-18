#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <string>
#include "FitFunction.hpp"

/*! 
 *  \file 
 *  \brief This File holds all configurations and constants. 
 */

/*! \var SAMPLE_COUNT 
 *  \brief Number of samples per event */
const unsigned int SAMPLE_COUNT = 1000;
/*! \var CHUNK_COUNT 
 *  \brief Number of events copied to the GPU in one step */
const unsigned int CHUNK_COUNT = 1024;
/*! \var CHUNK_BUFFER_COUNT
 *  \brief  Number of chunks in the input buffer */
const unsigned int CHUNK_BUFFER_COUNT = 2048;
/*! \var INTERPOLATION_COUNT
 *  \brief  Number of chunks in the input buffer */
const unsigned int INTERPOLATION_COUNT = 1;

const std::string OUTPUT_FILENAME = "results.txt";
const std::string FILENAME_TESTFILE = "../data/Al_25keV-1.cdb";

const unsigned int window_size = 100;//SAMPLE_COUNT/INTERPOLATION_COUNT;
//typedef Gauss FitFunction;
typedef WindowPolynom<2> FitFunction;


const unsigned int SPACE = ((window_size+FitFunction::numberOfParams)*2+(window_size+FitFunction::numberOfParams)*FitFunction::numberOfParams);
#endif

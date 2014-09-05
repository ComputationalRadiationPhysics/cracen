#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#define RANGE_MINIMUM  -40000
#define GAUSS_PARAM0  20000
#define GAUSS_PARAM2  -30000 //Baseline
#define GAUSS_PARAM3  175	
#define SAMPLE_COUNT 1000
#define window_size SAMPLE_COUNT //100;//SAMPLE_COUNT/INTERPOLATION_COUNT;

#include <string>
#include "FitFunction.hpp"

/*! 
 *  \file 
 *  \brief This File holds all configurations and constants. 
 */

/*! \var SAMPLE_COUNT 
 *  \brief Number of samples per event */
/*! \var CHUNK_COUNT 
 *  \brief Number of events copied to the GPU in one step */
const unsigned int CHUNK_COUNT = 1024;
/*! \var CHUNK_BUFFER_COUNT
 *  \brief  Number of chunks in the input buffer */
const unsigned int CHUNK_BUFFER_COUNT = 200;
/*! \var INTERPOLATION_COUNT
 *  \brief  Number of chunks in the input buffer */
const unsigned int INTERPOLATION_COUNT = 1;
/*! \var MAX_ITERATIONS
	\brief Maximum amount of iterations levenbergMarquardt will try to fit
*/
const unsigned int MAX_ITERATIONS = 100;

const unsigned int maxNumberOfDevices = 1;
const unsigned int pipelineDepth = 1;
	
const std::string OUTPUT_FILENAME = "results.txt";
const std::string FILENAME_TESTFILE = "../data/Al_25keV-1.cdb";

const unsigned int polynom_order = 2;
typedef Gauss FitFunction; // y = p0*e^(-1* ((x-p1)/p3)^2) + p2
//typedef WindowPolynom<polynom_order> FitFunction; // y = p0 + p1*x + p2*x^2 + p3*x^3 + ...
//typedef Polynom<polynom_order> FitFunction; // y = p0 + p1*x + p2*x^2 + p3*x^3 + ...

/*  Do not touch */
const unsigned int MIN_COMPUTE_CAPABILITY_MAJOR = 3;
const unsigned int MIN_COMPUTE_CAPABILITY_MINOR = 5;
const unsigned int SPACE = ((window_size+FitFunction::numberOfParams)*2+(window_size+FitFunction::numberOfParams)*FitFunction::numberOfParams);

#endif

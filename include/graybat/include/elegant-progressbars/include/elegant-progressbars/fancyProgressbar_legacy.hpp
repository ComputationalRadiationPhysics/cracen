#pragma once

#include <sstream>
#include <string>
#include <sys/time.h>

#include "printPattern_legacy.hpp"
#include "printPercentage_legacy.hpp"
#include "printTime_legacy.hpp"

namespace ElegantProgressbars{
    
/**
 * Gives the difference of 1 timevals in terms of seconds with fraction.
 * 
 * Usage is similar to the traditional way to calculate passed time
 * (endTime-startTime), but this one uses timeval-structs internally to reach
 * microsecond-precision.
 *
 * @param end the end-time
 * @param start the start-time
 */
inline float timevalDiff(timeval const end, timeval const start){
  return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
}

/**
 * Writes a fancy progressbar with minimal input
 *
 * Takes the total number of elements to process and creates a nice progressbar
 * from that. This is intended to be called in a loop / recursion / MPI thread
 * / pthread / etc. It should be called each time one of the nTotal elements is
 * done processing.
 * Can be fine-tuned for different length of the progressbar through a template
 *
 * @param nTotal the total number of elements to process. If multiple values
 *               are supplied in different calls, the function will try to use
 *               the highest of those values.
 * @param current (optional) the element you are currently at. This can be used
 *               to overwrite the default behaviour (which is to assume that an
 *               element was successfully processed each time this function is
 *               called)
 * @param length (template, optional) used to change the length of the printed
 *               progressbar
 *
 */
template<unsigned length = 50>
inline std::string fancyProgressBarLegacy(
    unsigned const nTotal, 
    unsigned const current = 0
    ){

  static unsigned maxNTotal = 0;
  static unsigned part = 0;
  static unsigned tic  = 0;
  static timeval startTime;
  if(part==0){ gettimeofday(&startTime,NULL); } // get the starting time on the very first call

  std::stringstream ss;
  timeval now;
  gettimeofday(&now,NULL);

  maxNTotal = std::max(maxNTotal, nTotal);
  part = current ? current : part+1;

  //limit the update intervall (not faster than every 35ms. This would be madness.)
  float const timeSpent = timevalDiff(now, startTime);  
  if(timeSpent > 0.035f*tic || part == maxNTotal){
    ++tic;
    float const percentage  = static_cast<float>(part) / static_cast<float>(maxNTotal);

    ss << "\rProgress: ";
    ss << printPattern<length>(tic, percentage);
    ss << printPercentage(part, maxNTotal, percentage);
    ss << printTime(timeSpent, percentage);
    if(part == maxNTotal) ss << std::endl;
    ss << std::flush;
  }

  return ss.str();
}

}

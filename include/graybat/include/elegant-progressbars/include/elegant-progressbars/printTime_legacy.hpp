#pragma once

#include <sstream>
#include <string>
#include <cassert> /*assert*/

namespace ElegantProgressbars{
    
/**
 * Writes progress expressed as time passed
 *
 * Takes the time that was already spent on a task and the percentage of
 * completion of said task. These parameters are used to calculate remaining
 * time and overall time to completion and writes it in a nice, human-friendly
 * fashion.
 *
 * @param tSpent the time that was spent at the current task
 * @param percentage the percentage the current task is at (as a fraction of 1)
 */
    
inline std::string printTime(float const timeSpent, float const percentage){
  std::stringstream stream;

  assert(percentage <= 1.f);

  float const timeTotal     = timeSpent/percentage;
  float const timeRemaining = timeTotal-timeSpent;

  stream << "after "  << static_cast<int>(timeSpent)      << "s";
  stream << " ("      << static_cast<int>(timeTotal)      << "s total";
  stream << ", "      << static_cast<int>(timeRemaining)  << "s remaining)";
  return stream.str();
}

}

#pragma once

#include <sstream>
#include <string>
#include <chrono>
#include <tuple>

//#include "printPattern.hpp"
//#include "printPercentage.hpp"
//#include "printTime.hpp"
#include "Pattern.hpp"
#include "Time.hpp"
#include "Percentage.hpp"
#include "Label.hpp"

/**
 * A set of fancy progress bars.
 *
 */
namespace ElegantProgressbars{
    
/**
 * Writes a fancy progressbar with minimal input
 *
 * Takes the total number of elements to process and creates a nice progressbar
 * from that. This is intended to be called in a loop / recursion / MPI thread
 * / pthread / etc. It should be called each time one of the nTotal elements is
 * done processing.
 * Operation can be fine-tuned to write time with high precision and different
 * length of the progressbar through template arguments.
 *
 * @param nTotal the total number of elements to process. If multiple values
 *               are supplied in different calls, the function will try to use
 *               the highest of those values.
 * @param current (optional) the element you are currently at. This can be used
 *               to overwrite the default behaviour (which is to assume that an
 *               element was successfully processed each time this function is
 *               called)
 * @param highPrecision (template, optional) if set to true, time will be 
 *               displayed with additional milliseconds
 * @param length (template, optional) used to change the length of the printed
 *               progressbar
 *
 */
template<bool highPrecision = false, unsigned length = 50>
inline std::string fancyProgressBar(
    unsigned const nTotal, 
    unsigned const current = 0
    ){

  using namespace std::chrono;

  static unsigned maxNTotal = 0;
  static unsigned part = 0;
  static unsigned tic  = 0;
  static time_point<steady_clock> startTime;
  if(part==0){ startTime = steady_clock::now(); } // get the starting time on the very first call

  std::stringstream ss;
  auto const now = steady_clock::now();

  maxNTotal = std::max(maxNTotal, nTotal);
  part = current ? current : part+1;

  //limit the update intervall (not faster than every 35ms. This would be madness.)
  duration<float> const timeSpent = now - startTime;
  if(timeSpent.count() > 0.035f*tic || part == maxNTotal){
    ++tic;
    auto const percentage  = static_cast<float>(part) / static_cast<float>(maxNTotal);
    std::string s;
    unsigned h;

    ss << "\r";
    std::tie(s,h) = Label::print();
    ss << s;
    std::tie(s,h) = Pattern<length>::print(part, maxNTotal, percentage);
    ss << s;
    std::tie(s,h) = Percentage::print(part, maxNTotal, percentage);
    ss << s;
    std::tie(s,h) = Time<highPrecision>::print(part, maxNTotal, percentage);
    ss << s;
    if(part==maxNTotal) ss << std::endl;
    ss << std::flush;
  }

  return ss.str();
}

}

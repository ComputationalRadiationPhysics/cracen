#pragma once

#include <sstream>
#include <string>
#include <chrono>
#include <tuple>

namespace ElegantProgressbars{
    


/** The base case of the recursion, see function below for more details
 */
template<typename Head>
std::tuple<std::string, unsigned> iteratePolicies(unsigned part, unsigned const maxPart){
  return Head::print(part, maxPart);
}

/** Recurses over a variable list of templates
 *
 * Can take an arbitrary list of template arguments and will call the static
 * function print() on each of these arguments. Will combine the outputs of all
 * those calls into a final output.  The output format is a tuple consisting of
 * a string and the number of newline-symbols that this string contains
 * (basically the number of lines)
 *
 */
template<typename Head, typename Head2, typename... Tail>
std::tuple<std::string, unsigned> iteratePolicies(unsigned part, unsigned const maxPart){
  std::stringstream ss;
  std::string s;
  unsigned pheight;
  unsigned height = 0;

  std::tie(s, pheight) = iteratePolicies<Head>(part, maxPart);
  height += pheight;
  ss << s;

  std::tie(s, pheight) = iteratePolicies<Head2, Tail...>(part, maxPart);
  height += pheight;
  ss << s;

  return std::make_tuple(ss.str(),height);
}


/**
 * Writes a fancy progressbar that can be flexibly configured
 *
 * Takes the total number of elements to process and creates a nice progressbar
 * from that. This is intended to be called in a loop / recursion / MPI thread
 * / pthread / etc. It should be called each time one of the nTotal elements is
 * done processing.
 * Operation can be fine-tuned by giving multiple policies as template
 * arguments which each can have internal behaviour to influence different
 * parts of the output. The output of the policies will be concatenated to a
 * single frame for each update of the progressbar
 *
 * @param nTotal the total number of elements to process. If multiple values
 *               are supplied in different calls, the function will try to use
 *               the highest of those values.
 *
 */
template<typename... PolicyList>
inline std::string policyProgressbar(unsigned const nTotal, unsigned const current = 0){

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
    std::string frame;
    unsigned height;

    // use all policies to compose a single frame and save its height
    std::tie(frame, height) = iteratePolicies<PolicyList...>(part, maxNTotal);
    ss << frame;
    ss << std::flush;
    
    //move the cursor back to the beginning
    if(part!=maxNTotal) ss << "\033[" << height << "A\r";
    else ss << "\n";
  }

  return ss.str();
}

} //namespace ElegantProgressbars

#pragma once

#include <iomanip>  /*setw, setfill*/
#include <sstream>  /*stringstream*/
#include <cmath>    /*log10*/
#include <string>   /*return value string*/
#include <cassert>  /*assert*/

namespace ElegantProgressbars{
    
/**
 * Writes progress expressed as percentage
 *
 * Takes the elements that were already processed, the maximum number of
 * elements to process and the percentage that is described by that, which is
 * somewhat redundant. These parameters are printed in a nice and
 * human-friendly fashion.
 *
 * @param part at which element between 1 and maxPart the process is
 * @param maxPart the maximum number of elements to process
 * @param percentage (optional) the percentage the current task is at (as a
 *                   fraction of 1)
 */
inline std::string printPercentage(unsigned part, unsigned const maxPart, float percentage = -1.f){
    std::stringstream stream;

    if(percentage < 0)
      percentage = static_cast<float>(part) / static_cast<float>(maxPart);

    assert(percentage <= 1.f);
    assert(maxPart > 0);

    static const unsigned fillwidthPart = static_cast<unsigned>(1+std::log10(maxPart));
    stream << std::setfill(' ') << std::setw(3) << static_cast<int>(percentage*100) << "%";
    stream << " (" << std::setfill(' ') << std::setw(fillwidthPart) << part;
    stream << "/" << maxPart << ")";
    return stream.str();
}

}

#pragma once

#include <string>
#include <sstream>
#include <cassert> /*assert*/
#include <tuple>

namespace ElegantProgressbars{
    
/**
 * Writes progress as an ascii-art progressbar
 *
 * Takes some external time reference and a percentage to express how far the
 * progressbar should be filled. There may be supplied an additional pattern,
 * which will be moving as time passes. The pattern defaults to a sine-wave
 * like pattern. If no such movement and fancyness is desired, a single element
 * may be used as a pattern (e.g. "#").
 *
 * @param percentage the progress as a fraction of 1
 * @param pattern (optional) the pattern of the wave (defaults to a wave)
 * @param length (template, optional) parameter to change the length of the
 *               finished pattern
 */
template<unsigned length = 50>
class Pattern{
    
  public:
  inline static std::tuple<std::string, unsigned> print(
      unsigned part,
      unsigned const maxPart,
      float percentage = -1.f,
      std::wstring pattern = L"ø¤º°`°º¤ø,¸,"
      ){

    static unsigned tic = 0;
    std::stringstream stream;
    if(percentage < 0)
      percentage = static_cast<float>(part) / static_cast<float>(maxPart);
    assert(maxPart > 0);
    assert(percentage <= 1.f);

    unsigned const progress = static_cast<unsigned>(percentage*length);
    unsigned const plength = pattern.length();
    std::wstring::iterator const bg = pattern.begin();
    ++tic;

    stream << "[";
    for(unsigned i=0 ; i<progress ; ++i){
      unsigned const pos = (tic+i) % plength;
      stream << std::string(bg + pos, bg + pos+1);
    }
    for(unsigned i=0; i < length-progress ; ++i){
      stream << " ";
    }
    stream << "] ";
    return std::make_tuple(stream.str(), 0);
  }
};

}

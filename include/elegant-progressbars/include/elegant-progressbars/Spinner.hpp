#pragma once

#include <string>
#include <sstream>
#include <tuple>
#include <vector>


namespace ElegantProgressbars{
    

/** Produces an ascii-art spinning line
 *
 * The line will keep turning clockwise or counterclockwise at a constant speed
 * (in case of a constant duration between calls...). The direction as well as
 * the base speed can be adjusted using the template parameters.
 */
template<unsigned slowness = 2, bool counterclockwise = false>
class Spinner{
    
  public:
  inline static std::tuple<std::string, unsigned> print(...){
    const std::vector<std::string> frames =  {"-", "\\", "|", "/"};

    static unsigned tic = 0;
    std::stringstream stream;

    unsigned framecount = frames.size();
    std::string frame;
    if(counterclockwise) frame = frames[framecount - ((tic/slowness) % framecount) -1 ];
    else frame = frames[(tic/slowness) % framecount];
    stream << frame << " ";
    ++tic;
    return std::tuple<std::string, unsigned>(stream.str(), 0);
  }
};

}

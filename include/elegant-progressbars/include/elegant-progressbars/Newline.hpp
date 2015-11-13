#pragma once

#include <string>
#include <tuple>

namespace ElegantProgressbars{
    
/** Produces a singe newline. Ignores all input
 */
class Newline{
  public:
  inline static std::tuple<std::string,unsigned> print(...){
    return std::make_tuple("\n",1);
  }
};

}


#pragma once

#include <string>
#include <tuple>

namespace ElegantProgressbars{
    
/** Writes a simple label, ignores all input
 */
class Label{
  public:
  inline static std::tuple<std::string,unsigned> print(...){
    return std::make_tuple("Progress: ",0);
  }
};

}


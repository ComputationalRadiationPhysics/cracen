#include <iostream>

#include "elegant-progressbars/policyProgressbar.hpp"
#include "elegant-progressbars/all_policies.hpp"

#include "elegant-progressbars/fancyProgressbar.hpp"
#include "elegant-progressbars/fancyProgressbar_legacy.hpp"


//just some workload -> don't use optimizations, if you want that to work
void workload(){
  for(int j=0; j<2000000; ++j){
    int g = j;
    if (g){
      ;
    }
  }
}


using namespace ElegantProgressbars;

int main(){
  static int const nElements = 1000;

  //This progressbar is composed by using different policies that contribute to the output
  for(int i=0; i<nElements; ++i){
    workload();
    std::cerr << policyProgressbar<Label, Spinner<>, Percentage, Time<> >(nElements);
  }

  //This progressbar is composed by using different policies that contribute to the output
  for(int i=0; i<nElements; ++i){
    workload();
    std::cerr << policyProgressbar<Hourglass<>, Label, Pattern<>, Percentage, Time<> >(nElements);
  }

  //the template argument is for displaying milliseconds and can be omitted (defaults to false)
  for(int i=0; i<nElements; ++i){
    workload();
    std::cerr << fancyProgressBar<false,30>(nElements);
  }

  //this one is the progressbar without C++11 features (std::chrono and std::tuple, most notably)
  for(int i=0; i<nElements; ++i){
    workload();
    std::cerr << fancyProgressBarLegacy(nElements);
  }

  return 0;
}

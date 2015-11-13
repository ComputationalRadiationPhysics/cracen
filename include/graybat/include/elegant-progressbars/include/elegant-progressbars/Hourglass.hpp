#pragma once

#include <string>
#include <sstream>
#include <cassert> /*assert*/
#include <tuple>
#include <vector>


namespace ElegantProgressbars{
    

/** Produces an ascii-art hourglass to visualize time
 *
 * The hourglass will turn after all the sand ran through. The progress is not
 * related to the number of turns or the filling state of the hourglass, since
 * nobody can be sure how much sand there is inside. Hourglasses are not the
 * most accurate way to measure time. Do not be surprised if there is still
 * sand left at the end of the process.
 */
template<unsigned slowness = 8>
class Hourglass{
    
  public:
  inline static std::tuple<std::string, unsigned> print(...){
    static const std::string HTOP = "          #############      ";
    static const std::string HE1  = "          #           #      ";
    static const std::string HE2  = "           #         #       ";
    static const std::string HE3  = "             #     #         ";
    static const std::string HE4  = "               # #           ";
    static const std::string H5   = "                #            ";
    static const std::string HE6  = "               #.#           ";
    static const std::string HE7  = "             #  .  #         ";
    static const std::string HE8  = "           #    .    #       ";
    static const std::string HE9  = "          #     .     #      ";
    static const std::string HBOT = "          #############      ";

    static const std::string HF1  = "          #############      ";
    static const std::string HF2  = "           ###########       ";
    static const std::string HF3  = "             #######         ";
    static const std::string HF4  = "               ###           ";

    static const std::string HF6  = "               ###           ";
    static const std::string HF7  = "             #######         ";
    static const std::string HF8  = "           ###########       ";
    static const std::string HF9  = "          #############      ";

    static const std::string HR1  = "          ###       ###      ";
    static const std::string HR2  = "           ###     ###       ";
    static const std::string HR3  = "             ##   ##         ";
    static const std::string HR4  = "               #.#           ";

    static const std::string HR6  = "               #.#           ";
    static const std::string HR7  = "             #  #  #         ";
    static const std::string HR8  = "           #   ###   #       ";
    static const std::string HR9  = "          #   #####   #      ";
    static const std::string CLR  = "                             ";

    static const std::string HH0  = "       ####             #### ";
    static const std::string HH1  = "       ######         ##   # ";
    static const std::string HH2  = "       #########   ###     # ";
    static const std::string HH3  = "       ############        # ";
    static const std::string HH4  = "       #########   ###     # ";
    static const std::string HH5  = "       ######         ##   # ";
    static const std::string HH6  = "       ####             #### ";

    static const std::vector<std::string> hg1 =  {CLR, CLR, HTOP, HF1, HF2, HF3, HF4, H5, HE6, HE7, HE8, HE9, HBOT, CLR, CLR};
    static const std::vector<std::string> hg2 =  {CLR, CLR, HTOP, HF1, HF2, HF3, HF4, H5, HE6, HE7, HE8, HR9, HBOT, CLR, CLR};
    static const std::vector<std::string> hg3 =  {CLR, CLR, HTOP, HE1, HF2, HF3, HF4, H5, HE6, HE7, HE8, HF9, HBOT, CLR, CLR};
    static const std::vector<std::string> hg4 =  {CLR, CLR, HTOP, HE1, HF2, HF3, HF4, H5, HE6, HE7, HR8, HF9, HBOT, CLR, CLR};
    static const std::vector<std::string> hg5 =  {CLR, CLR, HTOP, HE1, HE2, HF3, HF4, H5, HE6, HE7, HF8, HF9, HBOT, CLR, CLR};
    static const std::vector<std::string> hg6 =  {CLR, CLR, HTOP, HE1, HE2, HF3, HF4, H5, HE6, HR7, HF8, HF9, HBOT, CLR, CLR};
    static const std::vector<std::string> hg7 =  {CLR, CLR, HTOP, HE1, HE2, HE3, HF4, H5, HE6, HF7, HF8, HF9, HBOT, CLR, CLR};
    static const std::vector<std::string> hg8 =  {CLR, CLR, HTOP, HE1, HE2, HE3, HE4, H5, HF6, HF7, HF8, HF9, HBOT, CLR, CLR};
    static const std::vector<std::string> hg9 =  {CLR, CLR, CLR,  CLR, HH0, HH1, HH2, HH3,HH4, HH5, HH6, CLR, CLR,  CLR, CLR};
    static const std::vector<std::vector<std::string> > hglass = {  hg1, hg2, hg3, hg4, hg5, hg6, hg7, hg8, hg9};

    static unsigned tic = 0;
    std::stringstream stream;


    unsigned framecount = hglass.size();
    auto frame = hglass[(tic/slowness) % framecount];
    for(unsigned i=0 ; i< frame.size() ; ++i){
      stream << frame[i] << std::endl;
    }
    ++tic;
    return std::tuple<std::string, unsigned>(stream.str(), frame.size());
  }
};

}

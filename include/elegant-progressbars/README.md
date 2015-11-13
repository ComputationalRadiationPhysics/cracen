elegant progressbars for a more civilized age
=============================================
Because yay progress!

Offers progressbars to be used in command-line applications, to give a little
bit of eye-candy to the waiting user.

This is a header-only library that will use C++11 features (some progressbars
offer a legacy-variant, which is very nice of them).

Installation
------------

###Manual###
Just put the folder `lib/elegant-progressbars` in a place where your compiler
will find it when searching for libraries to include.

You can use ```doxygen``` to get a nice documentation:  
```bash
doxygen Doxyfile
firefox doc/html/index.html
```

###CMake Supported###

Clone the project to some local path and add the following line to your `CMakeLists.txt`:

    find_package(elegant-progressbars PATHS "<ABSOLUTE-PATH-TO-ELEGANT-PROGRESSBAR>/cmake")


Example
-------
There is an example which can be compiled and tested:

###Dependencies
 - `cmake` >= 3.0.2
 - `clang` >= 3.4 **or** `gcc >= 4.8.2`

###Build:
```bash
git clone https://github.com/slizzered/elegant-progressbars-for-a-more-civilized-age.git
cd elegant-progressbars-for-a-more-civilized-age
cmake .
make ProgressBarExample
./ProgressBarExample
```

Usage
-----
There are currently 2 different kinds of progressbars.  First, the traditional
preconfigured progressbars (including the legacy-progressbars that work with
older C++ compilers.  Generally, usage is straightforward but might be enhanced
by different (template) parameters:
```c++
using namespace ElegantProgressbars;

// call one of those in a loop that iterates until maxElement

std::cerr << fancyProgressBarLegacy(maxElement);
std::cerr << fancyProgressBar(maxElement);
std::cerr << fancyProgressBar<true>(maxElement);
std::cerr << fancyProgressBar<false, 80>(maxElement, currentElement);
```

If you desire more flexibility, the new style of progressbars is implemented as
a policy based design and currently consists of a single function that can be
configured with different output policies. Each of those policies contributes
to the output in its own way, so it becomes possible customize a progressbar
and even extend it with own policies. Each of those policies might be
configured through additional template arguments:
```c++
using namespace ElegantProgressbars;

// call this one in a loop that iterates until maxElement
std::cerr << policyProgressbar<Hourglass, Label, Pattern<>, Percentage, Time<> >(maxElement);
```
result:
```
          #############      
          #           #      
           #         #       
             #     #         
               ###           
                #            
               #.#           
             #######         
           ###########       
          #############      
          #############      
                             
                             
Progress: [¤º°`°º¤ø,¸,ø¤º°`°º¤ø,¸,ø¤º°`°º¤ø,¸,ø¤º°`°º¤ø,¸,ø¤º] 100% (1000/1000) after 6s (6s total, 0s remaining)
```

License
-------
Copyright (c) 2014 Carlchristian Eckert  
Licensed under the MIT license.  
Free as in beer.

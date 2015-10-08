Graybat
=======

<b>Gr</b>aph <b>A</b>pproach  for Highl<b>y</b>  Generic Communication
Schemes <b>B</b>ased on <b>A</b>daptive <b>T</b>opologies


##Description##

**Graybat** is a C++ library that presents a graph-based communication
approach, which enables a mapping of algorithms to communication
patterns and further a mapping of these communication patterns to
varying hardware architectures. Therefore, a flexible and configurable
communication approach for parallel and distributed
applications. These mappings are established as an intermediate layer
between an application and communication libraries and are dynamically
adptable during run-time.


##Documentation##

Have a look at the documentation that is available [here](https://ComputationalRadiationPhysics.github.io/graybat)


##Referencing##

Graybat is a scientific project. If you **present and/or publish** scientific
results that used graybat, you should set this as a **reference**.

##Software License##

Graybat is licensed under the <b>LGPLv3+</b>. Please refer to our [LICENSE.md](LICENSE.md)


##Dependencies##

###Mandatory###
 * cmake 3.0.2
 * Boost 1.57.0
 * g++ 5.2.0 or clang 3.5
 * c++14

###Optional###
 * OpenMPI 1.8.0 (mpi communication policy)
 * zeromq 4.1.3  (zeromq communication policy) 
 * metis 5.1 (graph partitioning mapping)


##Usage##

Graybat is a header only library so nothing has to be build.
The most easy way to include graybat into your application
is to use the CMAKE `find_package()` interface:

    set(graybat_DIR <ABSOLUT-PATH-TO-GRAYBAT-LIB>)
    find_package(graybat REQUIRED CONFIG)
    include_directories(SYSTEM ${graybat_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${graybat_LIBRARIES})

Finally, the application can use graybat like `#include <graybat/Cage.hpp>`.


##Compiling Tests/Examples##

 * Clone the repository: `git clone https://github.com/computationalradiationphysics/graybat.git`
 * Change directory: `cd graybat`
 * Create the build directory: `mkdir -p build`
 * Change to build directory: `cd build`
 * Set compiler: `export CXX=[g++,clang++]`
 * Create Makefile `cmake ..`
 * Build project : `make [target]`


##Tested Compilers##

 * clang 3.5
 * g++ 5.2.0


### Current Compilation Status:

| *branch* | *state* | *description* |
| -------- | --------| ------------- |
| **master** | [![Build Status](http://haseongpu.mooo.com/api/badge/github.com/erikzenker/GrayBat/status.svg?branch=master)](http://haseongpu.mooo.com/github.com/erikzenker/GrayBat) |  stable releases |
| **dev**  | [![Build Status](http://haseongpu.mooo.com/api/badge/github.com/erikzenker/GrayBat/status.svg?branch=dev)](http://haseongpu.mooo.com/github.com/erikzenker/GrayBat) |development branch |


##Predefined Targets##

 * **example**: All example applications.

 * **test** : Build, unit and integration test.

 * **doc**: Build documentation in `doc/`.

 * **clean**: Cleanup build directory.


##Project Organization##

The project is organized in a couple of subdirectories.

 * The [example](example) directory contains examples produced during development of graybat.
 * The [include](include) directory contains the library itself, which is header only.
 * The [test](test) directory contains unit and integration tests (might me used as examples)
 * The [utils](utils) directory contains cmake modules and doxygen files.


##Related Material##
 * Talk by Erik Zenker of his diploma defence [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.16306.svg)](http://dx.doi.org/10.5281/zenodo.16306)


##Authors##

 * Erik Zenker (erikzenker@posteo.de)


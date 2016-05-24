project(DSP)
cmake_minimum_required(VERSION 3.0.1)

# helper for libs and packages
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")

###############################################################################
# Compiler Flags
###############################################################################
string(FIND ${CMAKE_CXX_COMPILER} "clang++" CXX_COMPILER_IS_CLANG++)
string(FIND ${CMAKE_CXX_COMPILER} "g++" CXX_COMPILER_IS_G++)

if(${CXX_COMPILER_IS_G++})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  # Supress boost hana warning
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cpp")
  
endif(${CXX_COMPILER_IS_G++})

if(${CXX_COMPILER_IS_CLANG++})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif(${CXX_COMPILER_IS_CLANG++})

###############################################################################
# PThreads
###############################################################################

find_package(Threads REQUIRED)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
set(OnlineDSP_LIBS ${OnlineDSP_LIBS} ${CMAKE_THREAD_LIBS_INIT})

###############################################################################
# Boost
###############################################################################

find_package(Boost 1.55.0 REQUIRED regex system program_options)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

#nvcc + boost 1.55 work around
if(Boost_VERSION EQUAL 105500)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif(Boost_VERSION EQUAL 105500)

################################################################################
# GRAYBAT
################################################################################
find_package(graybat REQUIRED CONFIG)
#include_directories(SYSTEM ${graybat_INCLUDE_DIRS})
include_directories(${graybat_INCLUDE_DIRS})
set(LIBS ${LIBS} ${graybat_LIBRARIES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

###############################################################################
# Executables
###############################################################################

file(GLOB_RECURSE SRCFILES 
	"${CMAKE_CURRENT_SOURCE_DIR}/src/"
)
LIST(REMOVE_DUPLICATES SRCFILES)


add_executable(CracenSource "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/example/CracenSource.cpp")
target_link_libraries(CracenSource ${LIBS})

add_executable(CracenSink "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/example/CracenSink.cpp")
target_link_libraries(CracenSink ${LIBS})

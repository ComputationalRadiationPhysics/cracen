project(DSP)
cmake_minimum_required(VERSION 3.0.1)

# helper for libs and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")

###############################################################################
# CUDA
###############################################################################
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-use_fast_math;-arch=compute_35;-code=sm_35;-Xptxas=-v;-O3;-DDEBUG_ENABLED")
set(CUDA_ARCH sm_35 CACHE STRING "Set GPU architecture")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

###############################################################################
# Compiler Flags
###############################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")

###############################################################################
# PThreads
###############################################################################

find_package(Threads)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

###############################################################################
# Boost
###############################################################################

find_package(Boost 1.55.0 REQUIRED COMPONENTS system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

#nvcc + boost 1.55 work around
if(Boost_VERSION EQUAL 105500)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif(Boost_VERSION EQUAL 105500)


###############################################################################
# Executables
###############################################################################


# Main processing programm
file(GLOB SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
cuda_add_executable(DSP "${SRCFILES}")
target_link_libraries(DSP ${LIBS})


# Sender programm

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
file(GLOB TRANSSRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/DSPTransmitter/*")
set(${TRANSSRCFILES} "${TRANSSRCFILES} ${SRCFILES}")
add_executable(DSPTransmitter ${TRANSSRCFILES})
target_link_libraries(DSPTransmitter ${LIBS})

ADD_CUSTOM_TARGET (
	run 
	./${CMAKE_CURRENT_BUILD_DIR}/DSP
	DEPENDS DSP
	COMMENT "Executing programm." VERBATIM
)

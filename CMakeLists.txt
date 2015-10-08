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
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif(${CXX_COMPILER_IS_G++})

if(${CXX_COMPILER_IS_CLANG++})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif(${CXX_COMPILER_IS_CLANG++})

###############################################################################
# CUDA
###############################################################################

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-use_fast_math;-arch=compute_35;-code=sm_35;-Xptxas=-v;-O3;-DDEBUG_ENABLED")
set(CUDA_ARCH sm_35 CACHE STRING "Set GPU architecture")

# Separate C++ from CUDA file compilation (So we can use our own compiler)
set(CUDA_SEPARABLE_COMPILATION ON)
# Do not propagate CXX flags to the nvcc host compiler (especially -std=c++11)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

###############################################################################
# PThreads
###############################################################################

find_package(Threads)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
set(OnlineDSP_LIBS ${OnlineDSP_LIBS} ${CMAKE_THREAD_LIBS_INIT})

###############################################################################
# Boost
###############################################################################

find_package(Boost 1.55.0 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})
set(OnlineDSP_LIBS ${OnlineDSP_LIBS} ${Boost_LIBRARIES})


#nvcc + boost 1.55 work around
if(Boost_VERSION EQUAL 105500)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif(Boost_VERSION EQUAL 105500)


###############################################################################
# Acquiris Library for the ScopeReader
###############################################################################

find_package(Acqiris REQUIRED)
include_directories(${Acqiris_INCLUDE_DIRS})
add_definitions(${Acqiris_CFLAGS})
add_definitions(${Acqiris_LDFLAGS})
add_definitions(${Acqiris_CFLAGS_OTHER})
set(LIBS ${LIBS} ${Acqiris_LIBRARIES})

################################################################################
# GRAYBAT
################################################################################
set(graybat_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/graybat)
find_package(graybat REQUIRED CONFIG)
#include_directories(SYSTEM ${graybat_INCLUDE_DIRS})
include_directories(${graybat_INCLUDE_DIRS})
set(LIBS ${LIBS} ${graybat_LIBRARIES})

###############################################################################
# Gtk3mm for the viewer
###############################################################################

find_package(PkgConfig REQUIRED)

PKG_CHECK_MODULES(GTK3 REQUIRED gtk+-3.0)
include_directories(${GTK3_INCLUDE_DIRS})
link_directories(${GTK3_LIBRARY_DIRS})
add_definitions(${GTK3_CFLAGS})
add_definitions(${GTK3_LDFLAGS})
add_definitions(${GTK3_CFLAGS_OTHER})
set(LIBS ${LIBS} ${GTK3_LIBRARIES})

PKG_CHECK_MODULES(GTK3MM gtkmm-3.0)
include_directories(${GTK3MM_INCLUDE_DIRS})
link_directories(${GTK3MM_LIBRARY_DIRS})
add_definitions(${GTK3MM_CFLAGS})
add_definitions(${GTK3MM_LDFLAGS})
add_definitions(${GTK3MM_CFLAGS_OTHER})
set(LIBS ${LIBS} ${GTK3MM_LIBRARIES})

###############################################################################
# Qt5 for the viewer
###############################################################################

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_BUILD_TYPE Debug)

find_package(Qt5Widgets)
include_directories(${Qt5Widgets_INCLUDE_DIRS})
find_package(Qt5Gui)
include_directories(${Qt5Gui_INCLUDE_DIRS})
set(OnlineDSP_LIBS ${OnlineDSP_LIBS} Qt5::Gui Qt5::Widgets)

###############################################################################
# Include directories
###############################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Config)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Device)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Input)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Output)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Utility)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Device)

###############################################################################
# Executables
###############################################################################

file(GLOB_RECURSE SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/Config/*"
						   "${CMAKE_CURRENT_SOURCE_DIR}/src/Input/*"
						   "${CMAKE_CURRENT_SOURCE_DIR}/src/Output/*"
						   "${CMAKE_CURRENT_SOURCE_DIR}/src/Utility/*")
						  
file(GLOB_RECURSE CUDASRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/Device/*")

file(GLOB_RECURSE OnlineDSP_SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/OnlineDSP/*")

#FOREACH(item ${SRCFILES})
#  IF(${item} MATCHES DSP.cu)
#    LIST(REMOVE_ITEM ${SRCFILES) ${item})
#  ENDIF(${item} MATCHES DSP.cu)
#  IF(${item} MATCHES Viewer.cpp)
#    LIST(REMOVE_ITEM ${SRCFILES} ${item})
#  ENDIF(${item} MATCHES Viewer.cpp)
#ENDFOREACH(item)
LIST(REMOVE_DUPLICATES SRCFILES)

#LIST(REMOVE_ITEM SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/viewer/Viewer.cpp)
#LIST(REMOVE_ITEM SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/Dsp.cu)
#LIST(REMOVE_ITEM SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/OnlineDSP/*)

cuda_add_executable(DSP "${SRCFILES};${CUDASRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Dsp.cpp")
target_link_libraries(DSP ${LIBS})

cuda_add_executable(Fitter "${SRCFILES};${CUDASRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Fitter.cpp")
target_link_libraries(Fitter ${LIBS})

add_executable(ScopeReader "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/ScopeReader.cpp")
target_link_libraries(ScopeReader ${LIBS})

add_executable(FileReader "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/FileReader.cpp")
target_link_libraries(FileReader ${LIBS})

add_executable(Viewer "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/viewer/Viewer.cpp")
target_link_libraries(Viewer ${LIBS})

add_executable(OnlineDSP "${OnlineDSP_SRCFILES}")
target_link_libraries(OnlineDSP ${OnlineDSP_LIBS})

ADD_CUSTOM_TARGET (
	run 
	./${CMAKE_CURRENT_BUILD_DIR}/DSP
	DEPENDS DSP
	COMMENT "Executing programm." VERBATIM
)
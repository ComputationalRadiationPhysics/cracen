project(DSP)
cmake_minimum_required(VERSION 3.0.1)

# helper for libs and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")

###############################################################################
# Compiler Flags
###############################################################################

set(CMAKE_CXX_FLAGGS "${CMAKE_CXX_FLAGGS} -Wall")
set(CMAKE_CXX_FLAGGS "${CMAKE_CXX_FLAGGS} -Wextra")

###############################################################################
# CUDA
###############################################################################
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-use_fast_math;-arch=compute_35;-code=sm_35;-Xptxas=-v;-O3;-DDEBUG_ENABLED")
set(CUDA_ARCH sm_35 CACHE STRING "Set GPU architecture")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Config)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Device)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Input)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Output)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Utility)

cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/Device)


###############################################################################
# PThreads
###############################################################################

find_package(Threads)
set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})

###############################################################################
# Boost
###############################################################################

find_package(Boost 1.55.0 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

#nvcc + boost 1.55 work around
if(Boost_VERSION EQUAL 105500)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif(Boost_VERSION EQUAL 105500)

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
# Executables
###############################################################################

file(GLOB_RECURSE SRCFILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*")
#FOREACH(item ${SRCFILES})
#  IF(${item} MATCHES DSP.cu)
#    LIST(REMOVE_ITEM ${SRCFILES) ${item})
#  ENDIF(${item} MATCHES DSP.cu)
#  IF(${item} MATCHES Viewer.cpp)
#    LIST(REMOVE_ITEM ${SRCFILES} ${item})
#  ENDIF(${item} MATCHES Viewer.cpp)
#ENDFOREACH(item)
LIST(REMOVE_DUPLICATES SRCFILES)
LIST(REMOVE_ITEM SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/viewer/Viewer.cpp)
LIST(REMOVE_ITEM SRCFILES ${CMAKE_CURRENT_SOURCE_DIR}/src/Dsp.cu)

MESSAGE(STATUS "${SRCFILES}")

cuda_add_executable(DSP "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Dsp.cu")
target_link_libraries(DSP ${LIBS})

add_executable(Viewer "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/viewer/Viewer.cpp")
target_link_libraries(Viewer ${LIBS})


ADD_CUSTOM_TARGET (
	run 
	./${CMAKE_CURRENT_BUILD_DIR}/DSP
	DEPENDS DSP
	COMMENT "Executing programm." VERBATIM
)

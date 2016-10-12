project(cracenExample)
cmake_minimum_required(VERSION 3.0.1)

# helper for libs and packages
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")

###############################################################################
# Compiler Flags
###############################################################################

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
# Supress boost hana warning
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-cpp")

###############################################################################
# Cracen
###############################################################################

find_package(cracen REQUIRED)
set(LIBS ${LIBS} ${cracen_LIBARIES})
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${cracen_INCLUDE_DIRS})

###############################################################################
# Executables
###############################################################################

include_directories(${INCLUDE_DIRS})

file(GLOB_RECURSE SRCFILES 
	"${CMAKE_CURRENT_SOURCE_DIR}/src/Cracen/"
)
LIST(REMOVE_DUPLICATES SRCFILES)


add_executable(CracenSource "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Cracen/example/CracenSource.cpp")
target_link_libraries(CracenSource ${LIBS})

add_executable(CracenSink "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Cracen/example/CracenSink.cpp")
target_link_libraries(CracenSink ${LIBS})

project(cracenExample)
cmake_minimum_required(VERSION 3.0.1)

# helper for libs and packages
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu/" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")

###############################################################################
# Compiler Flags
###############################################################################

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
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
	"src/*"
)
file(GLOB_RECURSE REMOVE_SOURCES "src/Example/*")

LIST(REMOVE_ITEM SRCFILES ${REMOVE_SOURCES})
LIST(REMOVE_DUPLICATES SRCFILES)

message(STATUS "REMOVE_SOURCES:" ${REMOVE_SOURCES})
message(STATUS "SRCFILES:" ${SRCFILES})

add_library(Cracen SHARED "${SRCFILES}")

#add_executable(CracenSource "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/CracenSource.cpp")
#target_link_libraries(CracenSource ${LIBS})
#
#add_executable(CracenSink "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/CracenSink.cpp")
#target_link_libraries(CracenSink ${LIBS})
#
#add_executable(CApi "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/CApi.cpp")
#target_link_libraries(CApi ${LIBS})

add_executable(BufferAdapterTest "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/BufferAdapterTest.cpp")
target_link_libraries(BufferAdapterTest ${LIBS})

#add_executable(Chain "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/example/Chain.cpp")
#target_link_libraries(Chain ${LIBS})

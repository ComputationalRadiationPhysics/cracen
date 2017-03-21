project(cracenExample)
cmake_minimum_required(VERSION 3.0.1)

# helper for libs and packages
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/lib/x86_64-linux-gnu/" "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")

message(STATUS "CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")

###############################################################################
# Compiler Flags
###############################################################################

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
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
# Find Boost
###############################################################################
find_package(Boost 1.60.0 COMPONENTS unit_test_framework REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})
ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)
enable_testing()

###############################################################################
# Executables
###############################################################################

include_directories(${INCLUDE_DIRS})

file(GLOB_RECURSE SRCFILES
	"src/*"
)
file(GLOB_RECURSE REMOVE_SOURCES "src/Example/*")
file(GLOB_RECURSE REMOVE_TESTS "src/Test/*")

LIST(REMOVE_ITEM SRCFILES ${REMOVE_SOURCES})
LIST(REMOVE_ITEM SRCFILES ${REMOVE_TESTS})
LIST(REMOVE_DUPLICATES SRCFILES)

message(STATUS "CXX_FLAGS:" ${CMAKE_CXX_FLAGS})
message(STATUS "SRCFILES:" ${SRCFILES})

#add_library(Cracen STATIC "${SRCFILES}")
#target_link_libraries(Cracen ${LIBS})
#set(LIBS "Cracen" ${LIBS})

add_executable(CracenSource "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/CracenSource.cpp")
target_link_libraries(CracenSource ${LIBS})

add_executable(CracenSink "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/CracenSink.cpp")
target_link_libraries(CracenSink ${LIBS})

add_executable(Benchmark "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/Benchmark.cpp")
target_link_libraries(Benchmark ${LIBS})

# Test cases
file(GLOB TESTS Tests/*.cpp)
add_executable(unittest "${SRCFILES};${REMOVE_TESTS}")
target_link_libraries(unittest ${LIBS})
add_test(ALL unittest)

#
#add_executable(CApi "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/CApi.cpp")
#target_link_libraries(CApi ${LIBS})

#add_executable(BufferAdapterTest EXCLUDE_FROM_ALL "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/BufferAdapterTest.cpp")
#target_link_libraries(BufferAdapterTest ${LIBS})

#add_executable(Chain "${SRCFILES};${CMAKE_CURRENT_SOURCE_DIR}/src/Example/example/Chain.cpp")
#target_link_libraries(Chain ${LIBS})

###############################################################################
# Documentation
###############################################################################


# add a target to generate API documentation with Doxygen
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_FOUND})

if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxygen.conf)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

#    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/html DESTINATION share/doc)
endif()


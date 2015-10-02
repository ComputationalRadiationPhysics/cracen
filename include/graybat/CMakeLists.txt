###############################################################################
# Graybat examples and tests
###############################################################################
cmake_minimum_required(VERSION 3.3.1)
project("graybat examples and tests")

###############################################################################
# Boost
###############################################################################
find_package(Boost 1.58.0 COMPONENTS unit_test_framework  REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

###############################################################################
# Graybat
###############################################################################
set(graybat_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
find_package(graybat REQUIRED CONFIG)
include_directories(SYSTEM ${graybat_INCLUDE_DIRS})
set(LIBS ${LIBS} ${graybat_LIBRARIES})

###############################################################################
# Compiler Flags
###############################################################################
string(FIND ${CMAKE_CXX_COMPILER} "clang++" CXX_COMPILER_IS_CLANG++)
string(FIND ${CMAKE_CXX_COMPILER} "g++" CXX_COMPILER_IS_G++)

if(${CXX_COMPILER_IS_G++})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif(${CXX_COMPILER_IS_G++})

if(${CXX_COMPILER_IS_CLANG++})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif(${CXX_COMPILER_IS_CLANG++})

###############################################################################
# Targets
###############################################################################

# Test cases
file(GLOB TESTS test/*.cpp)
add_executable(check EXCLUDE_FROM_ALL ${TESTS})
target_link_libraries(check ${LIBS})

# Examples
add_custom_target(example)
file(GLOB EXAMPLES  example/*.cpp)
foreach(EXAMPLE ${EXAMPLES})
	get_filename_component(DEP ${EXAMPLE} NAME_WE)
	add_executable(${DEP} EXCLUDE_FROM_ALL ${EXAMPLE})
	target_link_libraries(${DEP} LINK_INTERFACE_LIBRARIES graybat)	
	target_link_libraries(${DEP} ${LIBS})	
	add_dependencies(example ${DEP})
endforeach(EXAMPLE)

# ZMQ signaling server
add_executable(zmq_signaling EXCLUDE_FROM_ALL utils/zmq_signaling_server.cpp )
target_link_libraries(zmq_signaling ${LIBS})

# CTest
enable_testing()
add_test(graybat_check_build "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target check)
add_test(graybat_test_run  mpiexec -n 2 check )
set_tests_properties(graybat_test_run PROPERTIES DEPENDS graybat_check_build)

# Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.conf ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc 
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen." VERBATIM
    )
endif(DOXYGEN_FOUND)

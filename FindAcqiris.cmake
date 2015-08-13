# - Try to find Acqiris Driver API
# Once done this will define
#  Acqiris_FOUND - System has Acqiris Driver
#  Acqiris_INCLUDE_DIRS - The Acqiris Driver include directories
#  Acqiris_LIBRARIES - The libraries needed to use Acqiris Driver
#  Acqiris_DEFINITIONS - Compiler switches required for using Acqiris Driver

find_path(Acqiris_INCLUDE_DIR NAMES AgMD1.h
          HINTS ${CMAKE_CURRENT_SOURCE_DIR}/include/* /usr/include /usr/include/*})

find_library(Acqiris_LIBRARY_1 NAMES AgMD1Fundamental
             HINTS ${CMAKE_CURRENT_SOURCE_DIR}/lib /usr/lib /usr/lib32})

find_library(Acqiris_LIBRARY2 NAMES AgMD1 libAgMD1
             HINTS ${CMAKE_CURRENT_SOURCE_DIR}/lib /usr/lib /usr/lib32})


set(Acqiris_INCLUDE_DIRS ${Acqiris_INCLUDE_DIR})
set(Acqiris_LIBRARIES "${Acqiris_LIBRARY_1}${Acqiris_LIBRARY_2}")

# TODO: Add Switch for Windows
set(Acqiris_CFLAGS "-D_LINUX")

set(Acqiris_LDFLAGS)
set(Acqiris_CFLAGS_OTHER)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Acqiris_LIBRARY_1 Acqiris_LIBRARY_2 Acqiris_INCLUDE_DIR)
# handle the QUIETLY and REQUIRED arguments and set Acqiris_FOUND to TRUE
# if all listed variables are TRUE
mark_as_advanced(Acqiris_INCLUDE_DIR Acqiris_LIBRARY_1 Acqiris_LIBRARY_2)


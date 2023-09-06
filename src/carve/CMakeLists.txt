cmake_minimum_required(VERSION 3.5)
project(VL)

set(CMAKE_CXX_STANDARD 17)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif ()

MESSAGE(STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

MESSAGE(STATUS "Compiling for processor: " ${CMAKE_HOST_SYSTEM_PROCESSOR})

if (UNIX AND (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64"))
  MESSAGE(STATUS "Compiling with flags: -march=native")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

if (UNIX)

  MESSAGE(STATUS "Compiling with flags: -std=c++17 -O3 -ggdb -Wall -Wextra -Wno-missing-braces -Wno-unknown-attributes -Wno-unused-function")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-missing-braces -Wno-unknown-attributes -Wno-unused-function")

  if (VL_USE_SANITIZERS)
    MESSAGE(STATUS "Using sanitizers. Compiling with flags: -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  endif()

endif()

include_directories(external/cmd_line_parser/include)
include_directories(include)

add_executable(verify tools/verify.cpp)

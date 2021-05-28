cmake_minimum_required(VERSION 3.6)
project(THESIS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

find_package(PkgConfig REQUIRED)
pkg_search_module(SNDFILE REQUIRED sndfile)

include_directories("include")
include_directories(${SNDFILE_INCLUDE_DIRS})

add_executable(server src/server.cpp)
add_executable(client src/client.cpp)

add_executable(f2ch-enc src/f2ch/encoder.cpp src/wav.cpp)
target_link_libraries(f2ch-enc m ${SNDFILE_LIBRARIES})

add_executable(f2ch-dec src/f2ch/decoder.cpp src/wav.cpp)
target_link_libraries(f2ch-dec m ${SNDFILE_LIBRARIES})

add_executable(alaw-enc src/alaw/encoder.cpp src/wav.cpp)
target_link_libraries(alaw-enc m ${SNDFILE_LIBRARIES})

add_executable(alaw-dec src/alaw/decoder.cpp src/wav.cpp)
target_link_libraries(alaw-dec m ${SNDFILE_LIBRARIES})

add_executable(mulaw-enc src/mulaw/encoder.cpp src/wav.cpp)
target_link_libraries(mulaw-enc m ${SNDFILE_LIBRARIES})

add_executable(mulaw-dec src/mulaw/decoder.cpp src/wav.cpp)
target_link_libraries(mulaw-dec m ${SNDFILE_LIBRARIES})

add_executable(2bit-enc src/2bit/encoder.cpp src/wav.cpp)
target_link_libraries(2bit-enc m ${SNDFILE_LIBRARIES})

add_executable(2bit-dec src/2bit/decoder.cpp src/wav.cpp)
target_link_libraries(2bit-dec m ${SNDFILE_LIBRARIES})

add_executable(1bit-enc src/1bit/encoder.cpp src/wav.cpp)
target_link_libraries(1bit-enc m ${SNDFILE_LIBRARIES})

add_executable(1bit-dec src/1bit/decoder.cpp src/wav.cpp)
target_link_libraries(1bit-dec m ${SNDFILE_LIBRARIES})


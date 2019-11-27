cmake_minimum_required (VERSION 2.8.11)

add_executable(DeviceInfo DeviceInfo.cpp)

target_link_libraries(DeviceInfo PUBLIC ${OpenCL_LIBRARY})
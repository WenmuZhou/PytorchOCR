cmake_minimum_required(VERSION 2.8)

project( dbnet )

# 设置可执行文件输出目录
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/.)

#============
#debug mode
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(OpenCV_DIR /home/jun/ProgramFiles/opencv3.4/share/OpenCV/)
find_package(OpenCV 3.4.0 REQUIRED)

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

find_package(CUDA REQUIRED)
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# 根据 c 或者 c++ 进行条件编译
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
    message(STATUS "optional:-std=c++11")
endif()


set(dbnet_SRC_LISTS calibrator.cpp dbnet_utils.cpp dbnet_infer.cpp)

# clipper
include_directories(./ ./clipper)
add_subdirectory(clipper)

add_executable( dbnet ${dbnet_SRC_LISTS}  )
target_link_libraries(dbnet clipper)
target_link_libraries(dbnet ${OpenCV_LIBS})
target_link_libraries(dbnet nvinfer)
target_link_libraries(dbnet cudart)
target_link_libraries(dbnet nvonnxparser)

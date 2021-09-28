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


find_package(CUDA REQUIRED)
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#opencv
set(OpenCV_DIR usr/local/opencv340)
find_package(OpenCV 3.4.0 REQUIRED)

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# 根据 c 或者 c++ 进行条件编译
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
    message(STATUS "optional:-std=c++11")
endif()


set(dbnet_SRC_LISTS calibrator.cpp dbnet_utils.cpp dbnet_infer.cpp)

# clipper
include_directories(./ ./clipper)
add_subdirectory(clipper)

configure_file (
  "${PROJECT_SOURCE_DIR}/Config.h.in"
  "${PROJECT_BINARY_DIR}/Config.h"
  )
include_directories("$PROJECT_SOURCE_DIR")

add_executable( dbnet ${dbnet_SRC_LISTS}  )
target_link_libraries(dbnet clipper)
target_link_libraries(dbnet ${OpenCV_LIBS})
target_link_libraries(dbnet nvinfer)
target_link_libraries(dbnet cudart)
target_link_libraries(dbnet nvonnxparser)

if(DEFINED BATCH_SIZE)
    target_compile_definitions( dbnet PRIVATE DBATCH_SIZE )
endif()
if(DEFINED EXPANDRATIO)
    target_compile_definitions( dbnet PRIVATE DEXPANDRATIO )
endif()
if(DEFINED BOX_MINI_SIZE)
    target_compile_definitions( dbnet PRIVATE DBOX_MINI_SIZE )
endif()
if(DEFINED SCORE_THRESHOLD)
    target_compile_definitions( dbnet PRIVATE DSCORE_THRESHOLD )
endif()
if(DEFINED BOX_THRESHOLD)
    target_compile_definitions( dbnet PRIVATE DBOX_THRESHOLD )
endif()
if(DEFINED SHORT_INPUT)
    target_compile_definitions( dbnet PRIVATE DSHORT_INPUT )
endif()
if(DEFINED SCORE_THRESHOLD)
    target_compile_definitions( dbnet PRIVATE DSCORE_THRESHOLD )
endif()
if(DEFINED DEVICE)
    target_compile_definitions( dbnet PRIVATE DDEVICE )
endif()

add_executable( cv_forearch cv_forearch_test.cpp  )
target_link_libraries(cv_forearch ${OpenCV_LIBS})
target_link_libraries(cv_forearch ${openmp})
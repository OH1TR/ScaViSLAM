CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})SET(PROJECT_NAME ScaViSLAM)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMakeModules/")

#SET (OpenCV_DIR ${CMAKE_PREFIX_PATH}/share/OpenCV)
SET (OpenCV_DIR /usr/local/share/OpenCV)

SET (CMAKE_VERBOSE_MAKEFILE ON)

SET(SSE4_1_SUPPORT ON)

IF(NOT CUDA_SUPPORT)
  SET(CUDA_SUPPORT ON)
ENDIF()

IF( NOT PCL_SUPPORT)
  SET(PCL_SUPPORT OFF)
ENDIF()


IF(CUDA_SUPPORT)
  MESSAGE ("CUDA SUPPORT ON")
  ADD_DEFINITIONS(-DSCAVISLAM_CUDA_SUPPORT)
ELSE()
  MESSAGE ("CUDA SUPPORT OFF")
ENDIF()

IF(PCL_SUPPORT)
  MESSAGE ("PCL SUPPORT ON")
  ADD_DEFINITIONS(-DSCAVISLAM_PCL_SUPPORT)
  FIND_PACKAGE(PCL 1.2 REQUIRED)
  ADD_DEFINITIONS(${PCL_DEFINITIONS})
ELSE()
  MESSAGE ("PCL SUPPORT OFF")
ENDIF()

IF( NOT CMAKE_BUILD_TYPE )
  SET( CMAKE_BUILD_TYPE Release )
ENDIF()

IF (CMAKE_COMPILER_IS_GNUCXX )
  SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g")

  IF( NOT SSE4_1_SUPPORT )
    SET(SSE4_1_SUPPORT OFF)
  ENDIF()
  IF(SSE4_1_SUPPORT)
    SET(SSE4_1_SUPPORT ON)
    MESSAGE ("SSE 4.1 SUPPORT ON")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse4.1 -g")
  ELSE()
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
    MESSAGE ("SSE 4.1  SUPPORT OFF")
  ENDIF()
ENDIF()

FIND_PACKAGE(OpenCV 2.4.9 EXACT REQUIRED)
FIND_PACKAGE(SuiteSparse REQUIRED)
FIND_PACKAGE(Eigen3 REQUIRED)
LIST(APPEND INCLUDE_DIRS
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/EXTERNAL/VisionTools/
    ${PROJECT_SOURCE_DIR}/EXTERNAL/include
    )

LIST(APPEND CMAKE_LIBRARY_PATH
    ${OpenCV_LIB_DIR}
    ${PROJECT_SOURCE_DIR}/EXTERNAL/VisionTools/build/
    ${PROJECT_SOURCE_DIR}/EXTERNAL/lib
    )

SET (LIB_NAMES GL pangolin glut g2o_stuff g2o_core g2o_solver_csparse csparse
               g2o_csparse_extension  VisionTools Sophus
               boost_system boost_thread boost_filesystem boost_regex
               opencv_core opencv_features2d opencv_flann opencv_gpu
               opencv_nonfree
               opencv_highgui opencv_imgproc opencv_calib3d)

IF(CUDA_SUPPORT)
  FIND_PACKAGE(CUDA REQUIRED)
  SET(CUDA_NVCC_FLAGS  "-arch=sm_20" "--use_fast_math" "-O3"
                       "--ptxas-options=--verbose" "-keep"  )
  ADD_DEFINITIONS(-DCUDA_BUILD -DBT_USE_DOUBLE_PRECISION)
  LIST(APPEND INCLUDE_DIRS "${CUDA_SDK_ROOT_DIR}/common/inc")

  macro(ROS_CUDA_ADD_LIBRARY cuda_target)

      CUDA_ADD_CUDA_INCLUDE_ONCE()

      # Separate the sources from the options
      CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
      CUDA_BUILD_SHARED_LIBRARY(_cuda_shared_flag ${ARGN})
      # Create custom commands and targets for each file.
      CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources}
          ${_cmake_options} ${_cuda_shared_flag}
          OPTIONS ${_options} )

      # Add the library.
      rosbuild_add_library(${cuda_target} ${_cmake_options}
          ${_generated_files}
          ${_sources}
          )

      target_link_libraries(${cuda_target}
          ${CUDA_LIBRARIES}
          )

      # We need to set the linker language based on what the expected generated file
      # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
      set_target_properties(${cuda_target}
          PROPERTIES
          LINKER_LANGUAGE ${CUDA_C_OR_CXX}
          )

  endmacro(ROS_CUDA_ADD_LIBRARY cuda_target)

  macro(ROS_CUDA_ADD_EXECUTABLE cuda_target)

      CUDA_ADD_CUDA_INCLUDE_ONCE()

      # Separate the sources from the options
      CUDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
      # Create custom commands and targets for each file.
      CUDA_WRAP_SRCS( ${cuda_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )

      # Add the library.
      rosbuild_add_executable(${cuda_target} ${_cmake_options}
          ${_generated_files}
          ${_sources}
          )

      rosbuild_add_compile_flags(${cuda_target} "-Wall -Werror -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
      target_link_libraries(${cuda_target}
          ${CUDA_LIBRARIES}
          )

      # We need to set the linker language based on what the expected generated file
      # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
      set_target_properties(${cuda_target}
          PROPERTIES
          LINKER_LANGUAGE ${CUDA_C_OR_CXX}
          )

  endmacro(ROS_CUDA_ADD_EXECUTABLE cuda_target)

ENDIF()


FOREACH(lib ${LIB_NAMES})
  FIND_LIBRARY(LIB_${lib} ${lib})
  IF (LIB_${lib})
    MESSAGE(STATUS "found library '${lib}': ${LIB_${lib}}")
  ELSE (LIB_${lib})
    MESSAGE(FATAL_ERROR "cannot find library '${lib}' - this will not work ...")
  ENDIF (LIB_${lib})
  LIST(APPEND LIBS ${LIB_${lib}})
ENDFOREACH(lib)

#Add class XYZ here which consits of two files: XYZ.cpp and XYZ.h
SET (CLASSES  placerecognizer
              maths_utils
              ransac_models
              homography
              g2o_types/anchored_points
              stereo_camera
              backend
              fast_grid
              stereo_frontend
              dense_tracking
              filegrabber)


SET (TEMPLATE ransac
              frame_grabber
              matcher
              slam_graph )

SET (SOURCE_DIR "scavislam")

IF(PCL_SUPPORT)
  LIST(APPEND CLASSES rgbd_grabber)
ENDIF()

IF(CUDA_SUPPORT)
  LIST(APPEND SOURCES ${SOURCE_DIR}/gpu/dense_tracking.cuh
                      ${SOURCE_DIR}/gpu/dense_tracking.cu)
ENDIF()


FOREACH(class ${CLASSES})
  LIST(APPEND SOURCES ${SOURCE_DIR}/${class}.cpp)
ENDFOREACH(class)

FOREACH(template ${TEMPLATE})
  LIST(APPEND SOURCES ${SOURCE_DIR}/${template}.cpp
                      ${SOURCE_DIR}/${template}-impl.cpp )
ENDFOREACH(template)

INCLUDE_DIRECTORIES(${INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
LINK_DIRECTORIES(${LINK_DIRS} ${PCL_LIBRARY_DIRS} ${SOURCE_DIR})
LINK_LIBRARIES (${PROJECT_NAME} ${LIBS} ${PCL_LIBRARIES})

IF(CUDA_SUPPORT)
  LIST(APPEND INCLUDE_DIRS ${CUDA_SDK_ROOT_DIR}/common/inc ${CUDA_INCLUDE_DIRS})
  ROS_CUDA_ADD_EXECUTABLE(stereo_slam ${SOURCE_DIR}/stereo_slam.cpp)
  ROS_CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCES})
ELSE()
  rosbuild_add_executable(stereo_slam ${SOURCE_DIR}/stereo_slam.cpp)
  rosbuild_add_library(${PROJECT_NAME} SHARED ${SOURCES})
ENDIF()
ROSBUILD_ADD_EXECUTABLE(create_dictionary ${SOURCE_DIR}/create_dictionary.cpp)
IF(PCL_SUPPORT)
    ROSBUILD_ADD_EXECUTABLE(dump_rgbd ${SOURCE_DIR}/dump_rgbd.cpp)
ENDIF()


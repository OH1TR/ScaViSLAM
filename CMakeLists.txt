cmake_minimum_required(VERSION 2.8.3)
project(ScaViSLAM)

SET( CMAKE_BUILD_TYPE RelWidthDebInfo )

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMakeModules/")

SET (OpenCV_DIR /usr/local/share/OpenCV)

SET (CMAKE_VERBOSE_MAKEFILE ON)

SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse4.1 -g")

SET(CUDA_NVCC_FLAGS  "-arch=sm_20" "--use_fast_math" "-O3"
                   "--ptxas-options=--verbose" "-keep"  )

find_package(catkin REQUIRED
    COMPONENTS
        dynamic_reconfigure
        pcl
        pcl_ros
        Pangolin
        Sophus
        VisionTools
    )
find_package(OpenCV 2.4.3 REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED
    COMPONENTS
        system
        thread
        filesystem
        regex
    )
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
set(G2O_ROOT /opt/ros/groovy)
find_package(G2O REQUIRED)

catkin_package(
)

ADD_DEFINITIONS(-DCUDA_BUILD -DBT_USE_DOUBLE_PRECISION -DSCAVISLAM_CUDA_SUPPORT)


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
              framedata
              matcher
              slam_graph )

SET (SOURCE_DIR "scavislam")

LIST(APPEND SOURCES ${SOURCE_DIR}/gpu/dense_tracking.cuh
                  ${SOURCE_DIR}/gpu/dense_tracking.cu)


FOREACH(class ${CLASSES})
  LIST(APPEND SOURCES ${SOURCE_DIR}/${class}.cpp)
ENDFOREACH(class)

FOREACH(template ${TEMPLATE})
  LIST(APPEND SOURCES ${SOURCE_DIR}/${template}.cpp
                      ${SOURCE_DIR}/${template}-impl.cpp )
ENDFOREACH(template)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${OpenGL_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    ${CUDA_SDK_ROOT_DIR}/common/inc
    ${CUDA_INCLUDE_DIRS}
    ${G2O_INCLUDE_DIR}
    )

link_directories(
    ${catkin_LIBRARY_DIRS}
    ${EIGEN3_LIBRARY_DIR}
    ${OpenCV_LIBRARY_DIRS}
    ${CSPARSE_LIBRARY_DIR}
    ${Boost_LIBRARY_DIRS}
    ${OpenGL_LIBRARY_DIRS}
    ${GLUT_LIBRARY_DIRS}
    )

LINK_LIBRARIES (
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${OpenGL_LIBRARIES}
    ${GLUT_LIBRARIES}
    ${G2O_LIBRARIES}
)

CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCES})

CUDA_ADD_EXECUTABLE(stereo_slam ${SOURCE_DIR}/stereo_slam.cpp)

CUDA_ADD_EXECUTABLE(create_dictionary ${SOURCE_DIR}/create_dictionary.cpp)

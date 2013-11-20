cmake_minimum_required(VERSION 2.8.3)
project(ScaViSLAM)

SET( CMAKE_BUILD_TYPE RelWidthDebInfo )

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMakeModules/")

SET (OpenCV_DIR /usr/local/share/OpenCV)

# SET (CMAKE_VERBOSE_MAKEFILE ON)

SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -msse4.1 -g")

SET(CUDA_NVCC_FLAGS  "-arch=sm_20" "--use_fast_math" "-O3"
                   "--ptxas-options=--verbose" "-keep"  )

find_package(catkin REQUIRED
    COMPONENTS
        pcl
        pcl_ros
        Pangolin
        Sophus
        VisionTools
        image_transport
        image_geometry
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
    CATKIN_DEPENDS
        VisionTools
        Sophus
        Pangolin
    DEPENDS
        G2O
        CUDA
        OpenCV
    INCLUDE_DIRS
        scavislam
        ${CUDA_SDK_ROOT_DIR}/common/inc
    LIBRARIES ${PROJECT_NAME}
    CFG_EXTRAS definitions.cmake
)

INCLUDE(cmake/definitions.cmake)

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
              )

SET (TEMPLATE ransac
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

install(TARGETS ${PROJECT_NAME}
       LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
       )

install(FILES
      scavislam/backend.h
      scavislam/data_structures.h
      scavislam/dense_tracking.h
      scavislam/fast_grid.h
      scavislam/framedata.hpp
      scavislam/global.h
      scavislam/homography.h
      scavislam/keyframes.h
      scavislam/matcher.hpp
      scavislam/math_utils.h
      scavislam/placerecognizer.h
      scavislam/pose_optimizer.h
      scavislam/quadtree.h
      scavislam/ransac.hpp
      scavislam/ransac_models.h
      scavislam/slam_graph.hpp
      scavislam/stereo_camera.h
      scavislam/stereo_frontend.h
      scavislam/transformations.h
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

#include "stereo_camera.h"
#include "visiontools/accessor_macros.h"

#include "framedata.cpp"

namespace ScaViSLAM
{
template<>
FrameData<LinearCamera>::
FrameData(const LinearCamera &cam)
    :
      cam(cam),
    #ifdef SCAVISLAM_CUDA_SUPPORT
      gpu_pyr_float32_dx(NUM_PYR_LEVELS),
      gpu_pyr_float32_dy(NUM_PYR_LEVELS),
    #endif
      have_disp_img(false),
      offset(0)
{
}

template<>
FrameData<StereoCamera>::
FrameData(const StereoCamera &cam)
    :
      cam(cam),
    #ifdef SCAVISLAM_CUDA_SUPPORT
      gpu_pyr_float32_dx(NUM_PYR_LEVELS),
      gpu_pyr_float32_dy(NUM_PYR_LEVELS),
    #endif
      have_disp_img(false),
      offset(0)
{
    cur_left()
        = ImageSet(cv::Mat(cam.image_size(),CV_8UC1));
    prev_left()
        = ImageSet(cv::Mat(cam.image_size(),CV_8UC1));
    right =
        ImageSet(cv::Mat(cam.image_size(),CV_8UC1));
    cam_vec
        = ALIGNED<StereoCamera>::vector(NUM_PYR_LEVELS);
    for (int level = 0; level<NUM_PYR_LEVELS; ++level)
    {

        cam_vec.at(level)
            = StereoCamera(pyrFromZero_d(cam.focal_length(),level),
                    pyrFromZero_2d(cam.principal_point(),level),
                    cv::Size(pyrFromZero_d(cam.image_size().width,level),
                        pyrFromZero_d(cam.image_size().height,level)),
                    cam.baseline()*(1<<level));
    }
}

template class FrameData<StereoCamera>;
template class FrameData<LinearCamera>;
}

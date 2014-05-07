#include <opencv2/opencv.hpp>
#include <ros/time.h>

#ifdef SCAVISLAM_CUDA_SUPPORT
#include <opencv2/gpu/gpu.hpp>
#endif

#include "global.h"

namespace ScaViSLAM
{

struct ImageSet
{
public:
  ImageSet()
  {
  };

  ImageSet(const cv::Mat & img)
      :
          uint8(img),
          pyr_uint8(NUM_PYR_LEVELS)
#ifdef SCAVISLAM_CUDA_SUPPORT
          , gpu_pyr_float32(NUM_PYR_LEVELS)
#endif
          {
          };

  void clone(ImageSet & new_set)
  {
    new_set.uint8 = uint8.clone();
    for (int l = 0; l<NUM_PYR_LEVELS; ++l)
    {
      new_set.pyr_uint8[l] = pyr_uint8[l].clone();
    }

#ifdef SCAVISLAM_CUDA_SUPPORT
    new_set.gpu_uint8 = gpu_uint8.clone();
    for (int l = 0; l<NUM_PYR_LEVELS; ++l)
    {
      new_set.gpu_pyr_float32[l] = gpu_pyr_float32[l].clone();
    }
#endif
  };


  cv::Mat color_uint8;
  cv::Mat uint8;
  vector<cv::Mat> pyr_uint8;
  ros::Time time;
#ifdef SCAVISLAM_CUDA_SUPPORT
  cv::gpu::GpuMat gpu_uint8;
  vector<cv::gpu::GpuMat> gpu_pyr_float32;
#endif
};

template <class Camera>
class FrameData
{
public:
  FrameData(const Camera &cam);

  ImageSet& cur_left();

  ImageSet& prev_left();

  const ImageSet& cur_left() const;

  const ImageSet& prev_left() const;

  void nextFrame();

  Camera cam;
  typename ALIGNED<Camera>::vector cam_vec;
  ImageSet left[2];
  ImageSet right;

  cv::Mat disp;
  cv::Mat color_disp;
#ifdef SCAVISLAM_CUDA_SUPPORT
  cv::gpu::GpuMat gpu_disp_32f;
  cv::gpu::GpuMat gpu_xyzw;
  cv::gpu::GpuMat gpu_disp_16s;
  cv::gpu::GpuMat gpu_color_disp;
  vector<cv::gpu::GpuMat> gpu_pyr_float32_dx;
  vector<cv::gpu::GpuMat> gpu_pyr_float32_dy;
#else
  vector<cv::Mat> pyr_float32;
  vector<cv::Mat> pyr_float32_dx;
  vector<cv::Mat> pyr_float32_dy;
#endif
  int frame_id;
  bool have_disp_img;
private:
  int offset;
};

}

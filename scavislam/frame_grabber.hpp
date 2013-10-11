// This file is part of ScaViSLAM.
//
// Copyright 2011 Hauke Strasdat (Imperial College London)
//
// ScaViSLAM is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// any later version.
//
// ScaViSLAM is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with ScaViSLAM.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SCAVISLAM_FRAME_PIPELINE_H
#define SCAVISLAM_FRAME_PIPELINE_H

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <opencv2/opencv.hpp>
#include <pangolin/video.h>

#ifdef SCAVISLAM_CUDA_SUPPORT
#include <opencv2/gpu/gpu.hpp>
#endif

#include "framedata.hpp"

#include "filegrabber.h"

#ifdef SCAVISLAM_PCL_SUPPORT
#include "rgbd_grabber.h"
#endif


#include "global.h"

namespace VisionTools
{
class PerformanceMonitor;
}

namespace ScaViSLAM
{

using namespace std;
using namespace Eigen;
using namespace VisionTools;

template<class Camera>
class FrameGrabber
{
public:
  struct Params
  {
    Vector3d rot_left;
    Vector3d rot_right;
    Vector5d dist_coeff_left;
    Vector5d dist_coeff_right;
    bool  livestream;
    std::string  base_str;
    std::string  path_str;
    std::string  format_str;
    int  skip_imgs;
    bool color_img;
    bool right_img;
    bool disp_img;
    bool depth_img;
    bool rectify_frame;
  };

#ifdef SCAVISLAM_PCL_SUPPORT
  RgbdGrabber grabber;
#endif
  FileGrabber file_grabber_;
  FileGrabberMonitor file_grabber_mon_;

  FrameGrabber               (const Camera & cam,
                              const Vector4d & cam_distortion_,
                              PerformanceMonitor * per_mon_);
  void
  initialise                 ();
  void
  processNextFrame           ();

  const Params& params() const
  {
    return params_;
  }

  FrameData<Camera> frame_data;

private:
  void
  loadParams                 ();

  void
  rectifyFrame               ();

  void
  intializeRectifier         ();

  void
  frameFromLiveCamera        ();

  void
  preprocessing              ();

  void
  depthToDisp                (const cv::Mat & depth_img,
                              cv::Mat * disp_img) const;

  PerformanceMonitor * per_mon_;
  cv::Mat rect_map_left_[2];
  cv::Mat rect_map_right_[2];
  Vector4d cam_distortion_;

  Params params_;
  double size_factor_;
  //vector<std::string> file_base_vec_;
  std::string file_extension_;
#ifdef SCAVISLAM_CUDA_SUPPORT
  cv::Ptr<cv::gpu::FilterEngine_GPU> dx_filter_;
  cv::Ptr<cv::gpu::FilterEngine_GPU> dy_filter_;
#endif

private:
  DISALLOW_COPY_AND_ASSIGN(FrameGrabber)
};





}

#endif

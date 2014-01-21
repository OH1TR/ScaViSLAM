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

#ifndef SCAVISLAM_STEREO_FRONTEND_H
#define SCAVISLAM_STEREO_FRONTEND_H

#include <visiontools/performance_monitor.h>

#include "global.h"
#include "draw_items.h"
#include "frame_grabber.hpp"
#include "data_structures.h"
#include "quadtree.h"
#include "fast_grid.h"
#include "stereo_camera.h"
#include "transformations.h"
#include "dense_tracking.h"


namespace ScaViSLAM
{
using namespace VisionTools;

template<int obs_dim>
struct TrackData;

struct StereoFrontendDrawData
{

  StereoFrontendDrawData() :
    tracked_points2d(NUM_PYR_LEVELS),
    tracked_points3d(NUM_PYR_LEVELS),
    newtracked_points2d(NUM_PYR_LEVELS),
    newtracked_points3d(NUM_PYR_LEVELS),
    tracked_anchorpoints2d(NUM_PYR_LEVELS),
    fast_points2d(NUM_PYR_LEVELS),
    new_points2d(NUM_PYR_LEVELS),
    new_points3d(NUM_PYR_LEVELS)
  {
  }

  void clear()
  {
    for (int l = 0; l<NUM_PYR_LEVELS; ++l)
    {
      new_points2d.at(l).clear();
      new_points3d.at(l).clear();
      fast_points2d.at(l).clear();
      tracked_points2d.at(l).clear();
      tracked_points3d.at(l).clear();
      newtracked_points2d.at(l).clear();
      newtracked_points3d.at(l).clear();
      tracked_anchorpoints2d.at(l).clear();
    }
    blobs2d.clear();
  }

  ALIGNED<DrawItems::Line2dList>::vector tracked_points2d;
  ALIGNED<DrawItems::Point3dVec>::vector tracked_points3d;
  ALIGNED<DrawItems::Line2dList>::vector newtracked_points2d;
  ALIGNED<DrawItems::Point3dVec>::vector newtracked_points3d;
  ALIGNED<ALIGNED<DrawItems::Point2dVec>::int_hash_map>::vector
  tracked_anchorpoints2d;
  ALIGNED<DrawItems::Point2dVec>::vector fast_points2d;
  ALIGNED<DrawItems::Point2dVec>::vector new_points2d;
  ALIGNED<DrawItems::Point3dVec>::vector new_points3d;
  DrawItems::CircleList blobs2d;
};


class StereoFrontend
{
public:
  struct Parameters {
          double parallax_threshold;
          int newpoint_clearance;
          int covis_threshold;
          int new_keyframe_featureless_corners_thr;
          int use_n_levels_in_frontend;
          int ui_min_num_points;
          int num_disp16;
          int stereo_method;
          int stereo_iters;
          int stereo_levels;
          int stereo_nr_plane;
          int var_num_max_points;
          double max_reproj_error;
          int switch_keyframe_conection_threshold;
       Parameters() {
          parallax_threshold=0.75;
          newpoint_clearance=2;
          covis_threshold=15;
          new_keyframe_featureless_corners_thr=2;
          use_n_levels_in_frontend=2;
          ui_min_num_points=20;
          num_disp16=2;
          stereo_method=2;
          stereo_iters=4;
          stereo_levels=4;
          stereo_nr_plane=1;
          var_num_max_points=300;
          max_reproj_error=2;
          switch_keyframe_conection_threshold=100;
      }
  };

  StereoFrontend             (FrameData<StereoCamera> * frame_data_,
                              PerformanceMonitor * per_mon_,
                              Parameters params);

  void
  processFirstFrame          ();
  bool
  processFrame               (bool * is_frame_dropped);
  void
  initialize                 ();

  // getter and setter
  NeighborhoodPtr & neighborhood()
  {
    return neighborhood_;
  }

  const NeighborhoodPtr & neighborhood() const
  {
    return neighborhood_;
  }

  const StereoFrontendDrawData & draw_data() const
  {
    return draw_data_;
  }

  const SE3d& T_cur_from_actkey() const
  {
    return T_cur_from_actkey_;
  }

  void currentPose(SE3d& T_cur) const;

  const DenseTracker & tracker() const
  {
    return tracker_;
  }

  void SetParameters(Parameters par)
  {
    params_ = par;
  }
#ifdef SCAVISLAM_CUDA_SUPPORT
  void
  getDisparityParameters(int *windowSize, int *minDisparity, int *num_disparities);
#endif

  stack<AddToOptimzerPtr> to_optimizer_stack;
  tr1::unordered_map<int,Frame>  keyframe_map;
  tr1::unordered_map<int,list<CandidatePoint3Ptr > >  newpoint_map;
  vector<int>  keyframe_num2id;
  IntTable keyframe_id2num;
  int actkey_id;

private:

  struct RemoveCondition
  {
    RemoveCondition(const tr1::unordered_set<CandidatePoint3Ptr > &
                    matched_new_feat)
      :matched_new_feat(matched_new_feat)
    {
    }

    const tr1::unordered_set<CandidatePoint3Ptr > & matched_new_feat;

    bool operator()(const CandidatePoint3Ptr& ptr)
    {
      return matched_new_feat.find(ptr)!=matched_new_feat.end();
    }
  };

  struct PointStatistics
  {
    PointStatistics(int USE_N_LEVELS_FOR_MATCHING)
      : num_matched_points(USE_N_LEVELS_FOR_MATCHING)
    {
      num_points_grid2x2.setZero();
      num_points_grid3x3.setZero();

      for (int l=0; l<USE_N_LEVELS_FOR_MATCHING; ++l)
      {
        num_matched_points[l]=0;
      }
    }

    vector<int> num_matched_points;
    Matrix2i num_points_grid2x2;
    Matrix3i num_points_grid3x3;
    tr1::unordered_map<int,int> strength_to_neighbor;
    tr1::unordered_map<int,double> distance_to_neighbor;
    tr1::unordered_map<int,int> mutually_tracked_points;
  };

  void
  calculateDistanceToNeighbors( PointStatistics *point_stats,
                                double *min_dist,
                                int *min_id,
                                SE3d *T_cur_from_other);
  void
  calculateTrackCounts( const list<TrackPoint3Ptr> &track_point_list,
                        PointStatistics *point_stats);
  void
  switchOrDrop( const list<TrackPoint3Ptr> &track_point_list,
                PointStatistics *point_stats,
                bool *switchKeyframe,
                bool *dropKeyframe,
                int *other_id,
                SE3d *T_cur_from_other);

  bool
  shallWeDropNewKeyframe     (const PointStatistics & point_stats);

  bool
  shallWeSwitchKeyframe      (const list<TrackPoint3Ptr> & trackpoint_list,
                              int * other_id,
                              SE3d * T_cur_from_other,
                              ALIGNED<QuadTree<int> >::vector
                              * other_point_tree,
                              PointStatistics * other_stat);

  void
  addNewKeyframe             (const ALIGNED<QuadTree<int> >::vector &
                              feature_tree,
                              const AddToOptimzerPtr & to_optimizer,
                              tr1::unordered_set<CandidatePoint3Ptr > *
                              matched_new_feat,
                              ALIGNED<QuadTree<int> >::vector * point_tree,
                              PointStatistics * point_stats);
  AddToOptimzerPtr
  processMatchedPoints       (const TrackData<3> & track_data,
                              int num_new_feat_matched,
                              ALIGNED<QuadTree<int> >::vector * point_tree,
                              tr1::unordered_set<CandidatePoint3Ptr > *
                              matched_new_feat,
                              PointStatistics * stats);
  bool
  matchAndTrack              (const ALIGNED<QuadTree<int> >::vector &
                              feature_tree,
                              TrackData<3> * track_data,
                              int * num_new_feat_matched);

#ifdef SCAVISLAM_CUDA_SUPPORT
  void
  calcDisparityGpu           ();

#else
  void
  calcDisparityCpu           ();

#endif
  void
  computeFastCorners         (int trials,
                              ALIGNED<QuadTree<int> >::vector * feature_tree,
                              vector<CellGrid2d> * cell_grid_2d);
  void
  recomputeFastCorners       (const Frame & frame,
                              ALIGNED<QuadTree<int> >::vector * feature_tree);
  void
  addNewPoints               (int new_keyframe_id,
                              const ALIGNED<QuadTree<int> >::vector &
                              feature_tree);
  void
  addMorePoints              (int new_keyframe_id,
                              const ALIGNED<QuadTree<int> >::vector &
                              feature_tree,
                              const Matrix3i & add_flags,
                              ALIGNED<QuadTree<int> >::vector * new_qt,
                              vector<int> * num_points);

  void
  addMorePointsToOtherFrame  (int new_keyframe_id,
                              const SE3d & T_newkey_from_cur,
                              const ALIGNED<QuadTree<int> >::vector &
                              feature_tree,
                              const Matrix3i & add_flags,
                              const cv::Mat & disp,
                              ALIGNED<QuadTree<int> >::vector * new_qt,
                              vector<int> * num_points);
  int
  getNewUniqueId           ();

  Frame cur_frame_;

  FrameData<StereoCamera> * frame_data_;
  PerformanceMonitor * per_mon_;

  NeighborhoodPtr neighborhood_;
  SE3d T_cur_from_actkey_;

  vector<FastGrid> fast_grid_;
  SE3XYZ_STEREO se3xyz_stereo_;

  int USE_N_LEVELS_FOR_MATCHING;
  int unique_id_counter_;

  Parameters params_;
  StereoFrontendDrawData draw_data_;
  DenseTracker tracker_;

  double av_track_length_;

private:
  DISALLOW_COPY_AND_ASSIGN(StereoFrontend)
};

}

#endif

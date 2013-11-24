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

#include "placerecognizer.h"

#include <tr1/memory>

#include <visiontools/accessor_macros.h>
#include <visiontools/stopwatch.h>


#include <opencv2/nonfree/nonfree.hpp>
#include "ransac_models.h"
#include "ransac.hpp"

// Thanks a lot to Adrien Angeli for all help and discussion concerning
// place recognition using "bag of words".

namespace ScaViSLAM
{

bool PlaceRecognizerMonitor
::getKeyframeDate(PlaceRecognizerData * data)
{
  boost::mutex::scoped_lock lock(my_mutex_);

  if (new_keyframe_queue_.size()>0
      // there is a new keyframe waiting in input stack
      && detected_loop_stack_.size()==0)
    // no detected loop is in output stack
  {
    *data = new_keyframe_queue_.front();
    new_keyframe_queue_.pop();
    return true;
  }
  return false;
}

void PlaceRecognizerMonitor
::addKeyframeData(const PlaceRecognizerData & data)
{
  boost::mutex::scoped_lock lock(my_mutex_);

  while(new_keyframe_queue_.size()>0)
    new_keyframe_queue_.pop();


  new_keyframe_queue_.push(data);
}

bool PlaceRecognizerMonitor
::getQuery(PlaceRecognizerData * data)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  if(!query_stack_.empty()){
      *data=query_stack_.top();
      query_stack_.pop();
      return true;
  }
  return false;
}

void PlaceRecognizerMonitor
::query(const PlaceRecognizerData & data)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  while(!query_stack_.empty())
      query_stack_.pop();
  query_stack_.push(data);
}

bool PlaceRecognizerMonitor
::getLoop(DetectedLoop * loop)
{
  boost::mutex::scoped_lock lock(my_mutex_);

  if (detected_loop_stack_.size()>0)
  {
    *loop = detected_loop_stack_.top();
    detected_loop_stack_.pop();
    return true;
  }
  return false;
}

void PlaceRecognizerMonitor
::addLoop(const DetectedLoop & loop)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  detected_loop_stack_.push(loop);
}

bool PlaceRecognizerMonitor
::getQueryResponse(DetectedLoop * loop)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  if(!query_response_stack_.empty()){
    *loop=query_response_stack_.top();
    query_response_stack_.pop();
    return true;
  }
  return false;
}

void PlaceRecognizerMonitor
::queryResponse(const DetectedLoop & loop)
{
  boost::mutex::scoped_lock lock(my_mutex_);
  query_response_stack_.push(loop);
}

PlaceRecognizer
::PlaceRecognizer(const StereoCamera & stereo_cam, string wordspath)
  : stereo_cam_(stereo_cam)
{
  cv::Mat words_float_as_four_uint8
      = cv::imread(wordspath,-1);
  assert(words_float_as_four_uint8.size().area()>0);
  assert(words_float_as_four_uint8.type()==CV_8U);
  assert(sizeof(float)==4);
  assert(words_float_as_four_uint8.cols%4==0);
  words_ = cv::Mat(words_float_as_four_uint8.rows,
                   words_float_as_four_uint8.cols/4,
                   CV_32F,
                   words_float_as_four_uint8.data).clone();
  flann_index_
      = tr1::shared_ptr<generic_index_type>
        (new cv::flann::GenericIndex<distance_type>(
           words_,
           cvflann::KMeansIndexParams
           (32,11,cvflann::FLANN_CENTERS_KMEANSPP)));

  inverted_index_
      = vector<IntTable >(flann_index_->size(),
                                             IntTable());
  stop = false;
}

void PlaceRecognizer
::operator()()
{
  while(stop==false)
  {
    PlaceRecognizerData data;
    bool got_data = monitor.getKeyframeDate(&data);
    if(got_data)
    {
      addLocation(data);
    }

    got_data = monitor.getQuery(&data);
    if(got_data)
    {
        queryLocation(data);
    }

    boost::this_thread::sleep(boost::posix_time::milliseconds(1));
  }
}

void PlaceRecognizer
::calcLoopStatistics(int cur_keyframe_id,
                     const tr1::unordered_set<int> & exclude_set,
                     const tr1::unordered_map< int,int>
                     & keyframe_to_wordcount_map,
                     tr1::unordered_map<int,float> & location_stats)
{
  float number_of_locations
      = location_map_.size();
  float number_of_locations_containing_word
      = keyframe_to_wordcount_map.size();
  if (number_of_locations_containing_word>0)
  {
    float idf = number_of_locations/number_of_locations_containing_word;

    for(tr1::unordered_map<int,int >::const_iterator
        it = keyframe_to_wordcount_map.begin();
        it!=keyframe_to_wordcount_map.end(); ++it)
    {
      int other_keyframe_id = it->first;

      if (other_keyframe_id==cur_keyframe_id
          || exclude_set.find(other_keyframe_id)
          !=exclude_set.end())
        continue;

      float number_of_word_occurance = it->second;

      int number_of_words_in_loc
          = GET_MAP_ELEM(other_keyframe_id,location_map_).number_of_words;

      float tf = number_of_word_occurance/number_of_words_in_loc;

      if (other_keyframe_id!=cur_keyframe_id)
      {
        float val = tf*idf;

        ADD_TO_MAP_ELEM(other_keyframe_id, val, &location_stats);

      }
    }
  }
}

int PlaceRecognizer
::geometricCheck(const Place & query,
                 const Place & train,
                 DetectedLoop & loop)
{
  cv::BFMatcher matcher(cv::NORM_L2);
  vector<cv::DMatch > matches;

  matcher.match(query.descriptors,
                train.descriptors,
                matches);
  vector<cv::DMatch> inliers;

  loop.query_keyframe_id = query.keyframe_id;
  loop.loop_keyframe_id = train.keyframe_id;

  RanSaC<SE3Model>::compute(100,
                            stereo_cam_,
                            matches,
                            train.xyz_vec,
                            query.uvu_0_vec,
                            inliers,
                            loop.T_query_from_loop);

  return inliers.size();
}


void PlaceRecognizer
::computeSURFFeatures(const PlaceRecognizerData& pr_data, Place& new_loc)
{
  //todo: adpative SURF thr
  double surf_thr  = 600;
  vector<cv::KeyPoint> keypoints;
  vector<cv::KeyPoint> keypoints_with_depth;
  cv::SurfFeatureDetector surf(surf_thr, 2);
  surf.detect(pr_data.keyframe.pyr.at(0),keypoints);


  for (unsigned i=0; i<keypoints.size(); ++i)
  {
    const cv::KeyPoint & kp = keypoints[i];

    double disp = interpolateDisparity(pr_data.keyframe.disp,
                                       Vector2i(round(kp.pt.x),
                                                round(kp.pt.y)),
                                       0);
    if (disp>0)
    {
      Vector3d uvu(kp.pt.x,kp.pt.y,kp.pt.x-disp);

      new_loc.uvu_0_vec.push_back(uvu);
      new_loc.xyz_vec.push_back(stereo_cam_.unmap_uvu(uvu));
      keypoints_with_depth.push_back(kp);
    }
  }

  cv::SurfDescriptorExtractor surf_ext(2, 4, 2, false);
  surf_ext.compute(pr_data.keyframe.pyr.at(0),
                   keypoints_with_depth,
                   new_loc.descriptors);

  std::cerr << "Found " << new_loc.descriptors.rows << " SURF points" << std::endl;

  assert(new_loc.uvu_0_vec.size()==keypoints_with_depth.size());
  // Make sure SURF extractor did not remove keypoint from list!
}

void PlaceRecognizer
::assignWordsToFeatures(Place& new_loc,
                        const int keyframe_id,
                        const tr1::unordered_set<int>&  exclude_set,
                        tr1::unordered_map<int,float>& location_stats)
{
  int max_number_of_words = 1;
  cv::Mat idx(1,max_number_of_words,CV_32S);
  cv::Mat dists(1,max_number_of_words,CV_32F);

  // for all descriptors
  for (int r=0; r<new_loc.descriptors.rows; ++r)
  {
    const cv::Mat & query = new_loc.descriptors.row(r);


    //find corresponding words (0 to max_number_of_words)
    int num_found
        = flann_index_->radiusSearch(query,
                                     idx,
                                     dists,
                                     0.1,
                                     cvflann::SearchParams());

    int num_found_words = min(num_found,max_number_of_words);
    new_loc.number_of_words += num_found_words;
    cv::Mat found_idx =  idx(cv::Rect(0,0,num_found_words,1));

    // for all found words (0 to max_number_of_words)
    for (int i=0; i<num_found_words; ++i)
    {
      int word_idx = found_idx.at<int>(0,i);

      tr1::unordered_map< int,int> & keyframe_to_wordcount_map
          = inverted_index_.at(word_idx);

      calcLoopStatistics(keyframe_id,
                         exclude_set,
                         keyframe_to_wordcount_map,
                         location_stats);

      IntTable::iterator it
          = keyframe_to_wordcount_map.find(keyframe_id);
      if (it!=keyframe_to_wordcount_map.end())
      {
        ++it->second;
      }
      else
      {
        keyframe_to_wordcount_map.insert(make_pair(keyframe_id,1));
      }
    }
  }
}

//TODO: method too long
void PlaceRecognizer
::addLocation
( const PlaceRecognizerData & pr_data  )
{
    int best_match = -1;

    Place new_loc;
    new_loc.keyframe_id = pr_data.keyframe_id;

    computeSURFFeatures(pr_data, new_loc);

    tr1::unordered_map<int,float> location_stats;
    assignWordsToFeatures(new_loc,
            pr_data.keyframe_id,
            pr_data.exclude_set,
            location_stats);

    location_map_.insert(make_pair(pr_data.keyframe_id,new_loc));

    if (pr_data.do_loop_detection)
    {
        float max_score = 0;
        int max_score_idx = -1;

        for (tr1::unordered_map<int,float>::iterator it=location_stats.begin();
                it!=location_stats.end(); ++it)
        {
            float v = it->second;
            if (v>max_score)
            {
                max_score = v;
                max_score_idx = it->first;
            }
        }
        if (max_score>2.)
        {
            std::cerr << "Word match detected" << std::endl;
            best_match = max_score_idx;
            const Place & matched_loc = GET_MAP_ELEM(best_match, location_map_);
            DetectedLoop loop;
            int inliers = geometricCheck(new_loc,
                                         matched_loc,
                                         loop);
            if (inliers>30)
            {
                monitor.addLoop(loop);
            } else {
                std::cerr << "Geometric check failed " << inliers << std::endl;
            }
        }
    }
}

void PlaceRecognizer
::queryLocation
( const PlaceRecognizerData& pr_data)
{
    Place new_loc;
    new_loc.keyframe_id = pr_data.keyframe_id;

    computeSURFFeatures(pr_data, new_loc);

    tr1::unordered_map<int,float> location_stats;
    assignWordsToFeatures(new_loc,
            pr_data.keyframe_id,
            pr_data.exclude_set,
            location_stats);

    float max_score = 0;
    int max_score_idx = -1;
    int best_match=-1;

    for (tr1::unordered_map<int,float>::iterator it=location_stats.begin();
            it!=location_stats.end(); ++it)
    {
        float v = it->second;
        if (v>max_score)
        {
            max_score = v;
            max_score_idx = it->first;
        }
    }
    if (max_score>2.)
    {
        std::cerr << "Word match detected" << std::endl;
        best_match = max_score_idx;
        const Place & matched_loc = GET_MAP_ELEM(best_match, location_map_);
        DetectedLoop loop;
        int inliers = geometricCheck(new_loc,
                                     matched_loc,
                                     loop);
        if(inliers>30){
            monitor.queryResponse(loop);
        }
    }
}

}

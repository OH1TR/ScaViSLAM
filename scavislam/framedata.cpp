#include "framedata.hpp"

namespace ScaViSLAM
{

template <class Camera>
ImageSet& FrameData<Camera>::
cur_left()
{
    return left[offset];
}

template <class Camera>
ImageSet& FrameData<Camera>::
prev_left()
{
    return left[(offset+1)%2];
}

template <class Camera>
const ImageSet& FrameData<Camera>::
cur_left() const
{
    return left[offset];
}

template <class Camera>
const ImageSet& FrameData<Camera>::
prev_left() const
{
    return left[(offset+1)%2];
}

template <class Camera>
void FrameData<Camera>::
nextFrame()
{
    offset = (offset+1)%2;
}

}

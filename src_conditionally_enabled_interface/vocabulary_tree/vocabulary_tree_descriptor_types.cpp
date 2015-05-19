#include <vocabulary_tree/vocabulary_tree_descriptor_types.h>
#include <limits>

#ifdef SIFT_USE_L2
const vocabulary_tree::descriptor::Sift::DistanceType
  vocabulary_tree::descriptor::Sift::WorstDistance =
    std::numeric_limits<vocabulary_tree::descriptor::Sift::DistanceType>::max();
#endif

#ifdef SIFT_USE_DOT
const vocabulary_tree::descriptor::Sift::DistanceType
  vocabulary_tree::descriptor::Sift::WorstDistance = 0;
#endif

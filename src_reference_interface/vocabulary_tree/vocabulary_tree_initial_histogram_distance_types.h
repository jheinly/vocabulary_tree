#pragma once
#ifndef VOCABULARY_TREE_INITIAL_HISTOGRAM_DISTANCE_TYPES_H
#define VOCABULARY_TREE_INITIAL_HISTOGRAM_DISTANCE_TYPES_H

#include <vocabulary_tree/vocabulary_tree_types.h>
#include <algorithm>
#include <cmath>

/*
This file defines classes that can be used as template arguments for
VocabularyTree's InitialHistogramDistance template argument. Conceptually, each
class defines a different metric that can be used to compute the initial
distance between two word histograms. However, based on the way that
VocabularyTree performs its initial scoring using the inverted index, distance
functions that need to consider the difference between corresponding histogram
bins where one of the bins may be zero cannot be correctly implemented (e.g. L1
and L2) as the add_term() function will only be called when both histograms have
non-zero values in the same bin.
*/

namespace vocabulary_tree {

namespace initial_histogram_distance {

// Dot product distance between two histograms (corresponding bins are
// multiplied together, and then summed to a single value).
class Dot : public VocabularyTreeTypes
{
  public:
    Dot()
    : m_magnitude(0)
    {}

    inline void add_term(
      const frequency_t frequency1,
      const frequency_t frequency2,
      const frequency_t weight)
    { m_magnitude += weight * frequency1 * frequency2; }

    inline frequency_t compute_magnitude() const
    { return m_magnitude; }

    inline void reset()
    { m_magnitude = 0; }

  private:
    frequency_t m_magnitude;
};

// Intersect distance between two histograms (the minimum values between
// corresponding bins are selected, and then summed to a single value). In other
// words, the two histograms are intersected, and the total intersected area is
// summed.
class Intersect : public VocabularyTreeTypes
{
  public:
    Intersect()
    : m_magnitude(0)
    {}

    inline void add_term(
      const frequency_t frequency1,
      const frequency_t frequency2,
      const frequency_t weight)
    { m_magnitude += weight * std::min(frequency1, frequency2); }

    inline frequency_t compute_magnitude() const
    { return m_magnitude; }

    inline void reset()
    { m_magnitude = 0; }

  private:
    frequency_t m_magnitude;
};

} // namespace initial_histogram_distance

} // namespace vocabulary_tree

#endif // VOCABULARY_TREE_INITIAL_HISTOGRAM_DISTANCE_TYPES_H

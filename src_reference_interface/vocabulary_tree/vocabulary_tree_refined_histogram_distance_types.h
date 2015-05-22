#pragma once
#ifndef VOCABULARY_TREE_REFINED_HISTOGRAM_DISTANCE_TYPES_H
#define VOCABULARY_TREE_REFINED_HISTOGRAM_DISTANCE_TYPES_H

#include <vocabulary_tree/vocabulary_tree_types.h>
#include <algorithm>
#include <cmath>

/*
This file defines classes that can be used as template arguments for
VocabularyTree's RefinedHistogramDistance template argument. Conceptually, each
class defines a different metric that can be used to compute the refined
distance between two word histograms.
*/

namespace vocabulary_tree {

namespace refined_histogram_distance {

// L1 distance between two histograms.
class L1 : public VocabularyTreeTypes
{
  public:
    L1()
    : m_magnitude(0)
    {}

    inline void add_term(
      const frequency_t frequency1,
      const frequency_t frequency2,
      const frequency_t weight)
    { m_magnitude += weight * abs(frequency1 - frequency2); }

    inline frequency_t compute_magnitude() const
    { return m_magnitude; }

  private:
    frequency_t m_magnitude;
};

// Squared L2 distance between two histograms.
class L2Squared : public VocabularyTreeTypes
{
  public:
    L2Squared()
    : m_magnitude(0)
    {}

    inline void add_term(
      const frequency_t frequency1,
      const frequency_t frequency2,
      const frequency_t weight)
    {
      const frequency_t difference = frequency1 - frequency2;
      m_magnitude += weight * difference * difference;
    }

    inline frequency_t compute_magnitude() const
    { return m_magnitude; }

  private:
    frequency_t m_magnitude;
};

class ChiSquared : public VocabularyTreeTypes
{
  public:
    ChiSquared()
    : m_magnitude(0)
    {}

    inline void add_term(
      const frequency_t frequency1,
      const frequency_t frequency2,
      const frequency_t weight)
    {
      const frequency_t difference = frequency1 - frequency2;
      m_magnitude +=
        weight * difference * difference / (frequency1 + frequency2);
    }

    inline frequency_t compute_magnitude() const
    { return m_magnitude; }

  private:
    frequency_t m_magnitude;
};

} // namespace refined_histogram_distance

} // namespace vocabulary_tree

#endif // VOCABULARY_TREE_REFINED_HISTOGRAM_DISTANCE_TYPES_H

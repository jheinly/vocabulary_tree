#pragma once
#ifndef VOCABULARY_TREE_HISTOGRAM_NORMALIZATION_TYPES_H
#define VOCABULARY_TREE_HISTOGRAM_NORMALIZATION_TYPES_H

#include <vocabulary_tree/vocabulary_tree_types.h>
#include <cmath>

/*
This file defines classes that can be used as template arguments for
VocabularyTree's HistogramNormalization template argument. Conceptually, each
class defines a different metric for computing the magnitude of a histogram,
which will then be used for normalization.
*/

namespace vocabulary_tree {

namespace histogram_normalization {

// No normalization.
class None : public VocabularyTreeTypes
{
  public:
    None(const frequency_t /*initial_magnitude*/ = 0)
    {}

    inline void add_term(const frequency_t /*frequency*/) const
    {}

    inline void update_term(
      const frequency_t /*old_frequency*/,
      const frequency_t /*new_frequency*/) const
    {}

    inline frequency_t compute_magnitude() const
    { return 1; }
};

// L1 distance, which is a sum of absolute values (though histogram frequencies
// are always positive, so the absolute value is unnecessary).
class L1 : public VocabularyTreeTypes
{
  public:
    L1(const frequency_t initial_magnitude = 0)
    : m_magnitude(initial_magnitude)
    {}

    inline void add_term(const frequency_t frequency)
    { m_magnitude += frequency; }

    inline void update_term(
      const frequency_t old_frequency,
      const frequency_t new_frequency)
    { m_magnitude += new_frequency - old_frequency; }

    inline frequency_t compute_magnitude() const
    { return m_magnitude; }
  
  private:
    frequency_t m_magnitude;
};

// L2 distance, which is the square root of the sum of squared values.
class L2 : public VocabularyTreeTypes
{
  public:
    L2(const frequency_t initial_magnitude = 0)
    : m_magnitude(initial_magnitude)
    {}

    inline void add_term(const frequency_t frequency)
    { m_magnitude += frequency * frequency; }

    inline void update_term(
      const frequency_t old_frequency,
      const frequency_t new_frequency)
    {
      m_magnitude +=
        new_frequency * new_frequency -
        old_frequency * old_frequency;
    }

    inline frequency_t compute_magnitude() const
    { return sqrt(m_magnitude); }

  private:
    frequency_t m_magnitude;
};

} // namespace histogram_normalization

} // namespace vocabulary_tree

#endif // VOCABULARY_TREE_HISTOGRAM_NORMALIZATION_TYPES_H

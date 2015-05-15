#pragma once
#ifndef VOCABULARY_TREE_CONDITIONALLY_ENABLE_H
#define VOCABULARY_TREE_CONDITIONALLY_ENABLE_H

#include <vocabulary_tree/vocabulary_tree_types.h>
#include <vector>

namespace vocabulary_tree {

namespace conditionally_enable {

namespace idf_weights {

// This class is used when IDF-weights are enabled via the enable_idf_weights
// template argument in VocabularyTree.
template<class VocabularyTree>
class IdfWeightsEnabled
{
  public:
    // Compute and update the IDF (inverse document frequency) weights for the
    // words in the database.
    void compute_idf_weights();

    // Reset the IDF (inverse document frequency) weights to 1.0 for the words
    // in the database.
    void reset_idf_weights();

  protected:
    // Clear the IDF weights.
    void clear_idf_weights()
    { m_word_idf_weights.clear(); }

    // Return the IDF weight for a particular word.
    inline VocabularyTreeTypes::frequency_t get_idf_weight(
      const VocabularyTreeTypes::word_t word) const
    { return m_word_idf_weights[word]; }

  private:
    // Each word will be assigned an IDF weight (default is 1.0).
    std::vector<VocabularyTreeTypes::frequency_t> m_word_idf_weights;
};

// This class is used when IDF-weights are disabled via the enable_idf_weights
// template argument in VocabularyTree.
class IdfWeightsDisabled
{
  protected:
    void reset_idf_weights() const
    {};

    void clear_idf_weights() const
    {};

    VocabularyTreeTypes::frequency_t get_idf_weight(
      const VocabularyTreeTypes::word_t /*word*/) const
    { return 1; }
};

} // namespace idf_weights

namespace histogram_normalization {

// This class is used when histogram normalization is enabled via the
// HistogramNormalization template argument being not equal to None in
// Vocabulary Tree.
class HistogramNormalizationEnabled
{
  public:
    // Return the inverse magnitude of the histogram.
    inline VocabularyTreeTypes::frequency_t get_inverse_magnitude() const
    { return m_inverse_magnitude; }

    // Set the inverse magnitude of the histogram.
    inline void set_inverse_magnitude(
      const VocabularyTreeTypes::frequency_t inverse_magnitude)
    { m_inverse_magnitude = inverse_magnitude; }

  private:
    // When histogram normalization is enabled, the inverse magnitude of each
    // histogram is computed and stored.
    VocabularyTreeTypes::frequency_t m_inverse_magnitude;
};

// This class is used when histogram normalization is disabled via the
// HistogramNormalization template argument being equal to None in
// Vocabulary Tree.
class HistogramNormalizationDisabled
{
  public:
    inline VocabularyTreeTypes::frequency_t get_inverse_magnitude() const
    { return 1; }

    inline void set_inverse_magnitude(
      const VocabularyTreeTypes::frequency_t /*inverse_magnitude*/) const
    {}
};

} // namespace histogram_normalization

} // namespace conditionally_enable

} // namespace vocabulary_tree

#endif // VOCABULARY_TREE_CONDITIONALLY_ENABLE_H

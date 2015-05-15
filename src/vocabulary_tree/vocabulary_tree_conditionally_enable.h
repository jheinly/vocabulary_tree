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
    ////////////////////////////////////////////////////////////////////////////
    // Public Processing Functions

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

namespace inverse_magnitudes {

// This class is used when histogram normalization is enabled via the
// HistogramNormalization template argument being not equal to None and the
// enable_document_modification template argument equalling true in
// Vocabulary Tree.
class InverseMagnitudesEnabled
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
// HistogramNormalization template argument being equal to None or the
// enable_document_modification template argument equalling false in
// Vocabulary Tree.
class InverseMagnitudesDisabled
{
  public:
    inline VocabularyTreeTypes::frequency_t get_inverse_magnitude() const
    { return 1; }

    inline void set_inverse_magnitude(
      const VocabularyTreeTypes::frequency_t /*inverse_magnitude*/) const
    {}
};

} // namespace inverse_magnitudes

namespace document_modification {

template<class VocabularyTree, typename Descriptor>
class DocumentModificationEnabled
{
  public:
    ////////////////////////////////////////////////////////////////////////////
    // Public Processing Convenience Functions

    // Add new words to an existing document in the database.
    void add_words_to_document(
      const typename Descriptor::DimensionType * const descriptors_to_add,
      const VocabularyTreeTypes::index_t num_descriptors_to_add,
      const VocabularyTreeTypes::document_id_t document_id);

    // Remove words from an existing document in the database.
    // NOTE: If all of the words for a document are removed, the document will
    //       not automatically be removed from the database.
    void remove_words_from_document(
      const typename Descriptor::DimensionType * const descriptors_to_remove,
      const VocabularyTreeTypes::index_t num_descriptors_to_remove,
      const VocabularyTreeTypes::document_id_t document_id);

    ////////////////////////////////////////////////////////////////////////////
    // Public Processing Functions

    // Add new words to an existing document in the database.
    void add_words_to_document(
      const typename VocabularyTree::WordHistogram & histogram_of_words_to_add,
      const VocabularyTreeTypes::document_id_t document_id);

    // Remove words from an existing document in the database.
    // NOTE: If all of the words for a document are removed, the document will
    //       not automatically be removed from the database.
    void remove_words_from_document(
      const typename VocabularyTree::WordHistogram & histogram_of_words_to_remove,
      const VocabularyTreeTypes::document_id_t document_id);
};

class DocumentModificationDisabled
{};

} // namespace document_modification

} // namespace conditionally_enable

} // namespace vocabulary_tree

#endif // VOCABULARY_TREE_CONDITIONALLY_ENABLE_H

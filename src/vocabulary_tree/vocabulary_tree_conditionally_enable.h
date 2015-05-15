#pragma once
#ifndef VOCABULARY_TREE_CONDITIONALLY_ENABLE_H
#define VOCABULARY_TREE_CONDITIONALLY_ENABLE_H

#include <vocabulary_tree/vocabulary_tree_types.h>
#include <vocabulary_tree/vocabulary_tree_structs.h>
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

template<class VocabularyTree>
void IdfWeightsEnabled<VocabularyTree>::compute_idf_weights()
{
  // IDF weight for a word
  //   = log(# Documents in Database / # Documents with Word)
  //   = log(# Documents in Database) - log(# Documents with Word)
  // By splitting the computation into its subtraction form, we avoid a division
  // operation on each iteration.

  const VocabularyTree * vt = static_cast<const VocabularyTree *>(this);

  // If there are no documents in the database, reset the idf weights.
  if (vt->num_documents_in_database() == 0)
  {
    reset_idf_weights();
    return;
  }

  const VocabularyTree::frequency_t database_weight = log(
    static_cast<VocabularyTree::frequency_t>(vt->num_documents_in_database()));

  for (VocabularyTree::index_t i = 0; i < vt->m_num_words_in_vocabulary; ++i)
  {
    const VocabularyTree::index_t num_documents_with_word =
      static_cast<VocabularyTree::index_t>(vt->m_word_inverted_indices[i].size());
    if (num_documents_with_word > 0)
    {
      const VocabularyTree::frequency_t word_weight =
        log(static_cast<VocabularyTree::frequency_t>(num_documents_with_word));
      m_word_idf_weights[i] = database_weight - word_weight;
    }
    else
    {
      m_word_idf_weights[i] = 0;
    }
  }
}

template<class VocabularyTree>
void IdfWeightsEnabled<VocabularyTree>::reset_idf_weights()
{
  const VocabularyTree * vt = static_cast<const VocabularyTree *>(this);

  m_word_idf_weights.resize(vt->m_num_words_in_vocabulary);
  for (VocabularyTree::index_t i = 0; i < vt->m_num_words_in_vocabulary; ++i)
  {
    m_word_idf_weights[i] = 1;
  }
}

} // namespace idf_weights

namespace inverse_magnitudes {

// This class is used when histogram normalization is enabled via the
// HistogramNormalization template argument not being equal to None and the
// enable_document_modification template argument equalling true in
// VocabularyTree.
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
// VocabularyTree.
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

// This class is used when document modification is enabled via the 
// enable_document_modification template argument in VocabularyTree.
template<
  class VocabularyTree,
  class VocabularyTreeStructs,
  typename Descriptor,
  typename HistogramNormalization,
  typename HistogramDistance,
  bool enable_document_modification,
  bool enable_idf_weights>
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
      const typename VocabularyTreeStructs::WordHistogram & histogram_of_words_to_add,
      const VocabularyTreeTypes::document_id_t document_id);

    // Remove words from an existing document in the database.
    // NOTE: If all of the words for a document are removed, the document will
    //       not automatically be removed from the database.
    void remove_words_from_document(
      const typename VocabularyTreeStructs::WordHistogram & histogram_of_words_to_remove,
      const VocabularyTreeTypes::document_id_t document_id);
};

// This class is used when document modification is disabled via the 
// enable_document_modification template argument in VocabularyTree.
class DocumentModificationDisabled
{};

template<
  class VocabularyTree,
  class VocabularyTreeStructs,
  typename Descriptor,
  typename HistogramNormalization,
  typename HistogramDistance,
  bool enable_document_modification,
  bool enable_idf_weights>
void DocumentModificationEnabled<
  VocabularyTree,
  VocabularyTreeStructs,
  Descriptor,
  HistogramNormalization,
  HistogramDistance,
  enable_document_modification,
  enable_idf_weights>::
add_words_to_document(
  const typename Descriptor::DimensionType * const descriptors_to_add,
  const VocabularyTreeTypes::index_t num_descriptors_to_add,
  const VocabularyTreeTypes::document_id_t document_id)
{
  const VocabularyTree * vt = static_cast<const VocabularyTree *>(this);

  vt->compute_words(
    descriptors_to_add,
    num_descriptors_to_add,
    m_words);
  vt->compute_word_histogram(
    m_words,
    m_word_histogram);
  vt->add_words_to_document(
    m_word_histogram,
    document_id);
}

template<
  class VocabularyTree,
  class VocabularyTreeStructs,
  typename Descriptor,
  typename HistogramNormalization,
  typename HistogramDistance,
  bool enable_document_modification,
  bool enable_idf_weights>
void DocumentModificationEnabled<
  VocabularyTree,
  VocabularyTreeStructs,
  Descriptor,
  HistogramNormalization,
  HistogramDistance,
  enable_document_modification,
  enable_idf_weights>::
remove_words_from_document(
  const typename Descriptor::DimensionType * const descriptors_to_remove,
  const VocabularyTreeTypes::index_t num_descriptors_to_remove,
  const VocabularyTreeTypes::document_id_t document_id)
{
  const VocabularyTree * vt = static_cast<const VocabularyTree *>(this);

  vt->compute_words(
    descriptors_to_remove,
    num_descriptors_to_remove,
    m_words);
  vt->compute_word_histogram(
    m_words,
    m_word_histogram);
  vt->remove_words_from_document(
    m_word_histogram,
    document_id);
}

template<
  class VocabularyTree,
  class VocabularyTreeStructs,
  typename Descriptor,
  typename HistogramNormalization,
  typename HistogramDistance,
  bool enable_document_modification,
  bool enable_idf_weights>
void DocumentModificationEnabled<
  VocabularyTree,
  VocabularyTreeStructs,
  Descriptor,
  HistogramNormalization,
  HistogramDistance,
  enable_document_modification,
  enable_idf_weights>::
add_words_to_document(
  const typename VocabularyTreeStructs::WordHistogram & histogram_of_words_to_add,
  const VocabularyTreeTypes::document_id_t document_id)
{
  VocabularyTree * vt = static_cast<VocabularyTree *>(this);

  // Find the storage index for this document.
  const auto & found_iter = vt->m_document_to_storage_indices.find(document_id);
  if (found_iter == vt->m_document_to_storage_indices.end())
  {
    throw std::runtime_error(
      "VocabularyTree::add_words_to_document called with non-existent document_id");
  }
  const VocabularyTree::storage_index_t storage_index = found_iter->second;

  VocabularyTree::frequency_t initial_magnitude = 0;
  if (!std::is_same<HistogramNormalization, histogram_normalization::None>::value)
  {
    initial_magnitude = static_cast<VocabularyTree::frequency_t>(
      1.0 / vt->m_document_storage[storage_index].get_inverse_magnitude());
  }
  HistogramNormalization normalization(initial_magnitude);

  for (const auto & histogram_entry : histogram_of_words_to_add.histogram_entries)
  {
    auto & current_inverted_indices =
      vt->m_word_inverted_indices[histogram_entry.word];
    bool found = false;

    // Search for an existing entry in the inverted index.
    for (auto & inverted_index_entry : current_inverted_indices)
    {
      if (inverted_index_entry.storage_index == storage_index)
      {
        const VocabularyTree::frequency_t new_frequency =
          inverted_index_entry.frequency + histogram_entry.frequency;
        if (!std::is_same<HistogramNormalization, histogram_normalization::None>::value)
        {
          normalization.update_term(
            inverted_index_entry.frequency,
            new_frequency);
        }
        inverted_index_entry.frequency = new_frequency;
        found = true;
        break;
      }
    }
    // If an existing entry was not found, create a new inverted index entry for
    // this word-document pair.
    if (!found)
    {
      current_inverted_indices.push_back(
        VocabularyTree::InvertedIndexEntry(storage_index, histogram_entry.frequency));
      if (!std::is_same<HistogramNormalization, histogram_normalization::None>::value)
      {
        normalization.add_term(histogram_entry.frequency);
      }
    }
  }

  if (!std::is_same<HistogramNormalization, histogram_normalization::None>::value)
  {
    vt->m_document_storage[storage_index].set_inverse_magnitude(
      static_cast<VocabularyTree::frequency_t>(1.0 / normalization.compute_magnitude()));
  }
}

template<
  class VocabularyTree,
  class VocabularyTreeStructs,
  typename Descriptor,
  typename HistogramNormalization,
  typename HistogramDistance,
  bool enable_document_modification,
  bool enable_idf_weights>
void DocumentModificationEnabled<
  VocabularyTree,
  VocabularyTreeStructs,
  Descriptor,
  HistogramNormalization,
  HistogramDistance,
  enable_document_modification,
  enable_idf_weights>::
remove_words_from_document(
  const typename VocabularyTreeStructs::WordHistogram & histogram_of_words_to_remove,
  const VocabularyTreeTypes::document_id_t document_id)
{
  VocabularyTree * vt = static_cast<VocabularyTree *>(this);

  // Find the storage index for this document.
  const auto & found_iter = vt->m_document_to_storage_indices.find(document_id);
  if (found_iter == vt->m_document_to_storage_indices.end())
  {
    throw std::runtime_error(
      "VocabularyTree::remove_words_from_document called with non-existent document_id");
  }
  const VocabularyTree::storage_index_t storage_index = found_iter->second;

  VocabularyTree::frequency_t initial_magnitude = 0;
  if (!std::is_same<HistogramNormalization, histogram_normalization::None>::value)
  {
    initial_magnitude = static_cast<VocabularyTree::frequency_t>(
      1.0 / vt->m_document_storage[storage_index].get_inverse_magnitude());
  }
  HistogramNormalization normalization(initial_magnitude);

  for (const auto & histogram_entry : histogram_of_words_to_remove.histogram_entries)
  {
    auto & current_inverted_indices =
      vt->m_word_inverted_indices[histogram_entry.word];
    bool found = false;
    for (auto iter = current_inverted_indices.begin();
      iter != current_inverted_indices.end();
      ++iter)
    {
      if (iter->storage_index == storage_index)
      {
        const VocabularyTree::frequency_t new_frequency =
          iter->frequency - histogram_entry.frequency;
        if (!std::is_same<HistogramNormalization, histogram_normalization::None>::value)
        {
          normalization.update_term(
            iter->frequency,
            new_frequency);
        }
        if (new_frequency == 0) // TODO: Make sure that this comparison works.
        {
          current_inverted_indices.erase(iter);
        }
        else
        {
          iter->frequency = new_frequency;
        }
        found = true;
        break;
      }
    }
    if (!found)
    {
      throw std::runtime_error(
        "VocabularyTree::remove_words_from_document could not find inverted index entry for document");
    }
  }

  if (!std::is_same<HistogramNormalization, histogram_normalization::None>::value)
  {
    vt->m_document_storage[storage_index].set_inverse_magnitude(
      static_cast<VocabularyTree::frequency_t>(1.0 / normalization.compute_magnitude()));
  }
}

} // namespace document_modification

} // namespace conditionally_enable

} // namespace vocabulary_tree

#endif // VOCABULARY_TREE_CONDITIONALLY_ENABLE_H

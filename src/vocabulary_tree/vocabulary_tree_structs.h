#pragma once
#ifndef VOCABULARY_TREE_STRUCTS_H
#define VOCABULARY_TREE_STRUCTS_H

#include <vocabulary_tree/vocabulary_tree_types.h>
#include <vocabulary_tree/vocabulary_tree_conditionally_enable.h>
#include <vocabulary_tree/vocabulary_tree_histogram_normalization_types.h>
#include <type_traits>
#include <vector>

namespace vocabulary_tree {

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename HistogramDistance,
  bool enable_document_modification,
  bool enable_idf_weights>
class VocabularyTreeStructs : public VocabularyTreeTypes
{
  public:
    ////////////////////////////////////////////////////////////////////////////
    // Public Structures

    // This struct stores the occurrence frequency for a given word. Several of
    // these entries are combined to form a histogram of words (where each entry
    // represents one dimension of the histogram).
    struct HistogramEntry
    {
      HistogramEntry(
        const word_t word,
        const frequency_t frequency)
        : word(word),
        frequency(frequency)
      {}

      word_t word;
      frequency_t frequency;
    };

    // This struct stores a histogram of words. Specifically, it contains a list
    // of words, and their occurrence frequency (as HistogramEntry objects). The
    // inverse magnitude of the histogram is also stored for use in histogram
    // normalization.
    struct WordHistogram : public std::conditional<
      enable_document_modification &&
        !std::is_same<HistogramNormalization, histogram_normalization::None>::value,
      conditionally_enable::inverse_magnitudes::InverseMagnitudesEnabled,
      conditionally_enable::inverse_magnitudes::InverseMagnitudesDisabled>::type
    {
      std::vector<HistogramEntry> histogram_entries;
    };

    // This struct a single query result, in that it contains a document's ID,
    // (which had previously been stored in the database) as well as its score
    // with respect to the query document. A higher score indicates a better
    // match.
    struct QueryResult
    {
      static inline bool greater(
        const QueryResult & a,
        const QueryResult & b)
      {
        return a.score > b.score;
      }

      document_id_t document_id;
      frequency_t score;
    };

  protected:
    ////////////////////////////////////////////////////////////////////////////
    // Private Structures

    // This struct represents a node (either an iterior or a leaf) within the
    // vocabulary tree. Each node keeps track of its children (if it is an
    // interior node) or the word that it represents (if it is a leaf node).
    // NOTE: If this is a leaf node, then num_children will equal zero and word
    //       will contain the word index for this node. Otherwise, num_children
    //       will be non-zero and starting_index_for_children is the index of
    //       the children nodes within m_nodes as well as the starting index for
    //       the children's descriptors within m_descriptors.
    struct Node
    {
      Node()
        : starting_index_for_children(InvalidIndex),
        num_children(InvalidIndex),
        word(InvalidWord)
      {}

      index_t starting_index_for_children;
      index_t num_children;
      word_t word;
    };

    // This struct stores the occurrence frequency of a particular word for a
    // document in the database. Each word has its own list of inverted index
    // entries (m_word_inverted_indices), and each document is assigned a unique
    // storage index.
    struct InvertedIndexEntry
    {
      InvertedIndexEntry(
        const storage_index_t storage_index,
        const frequency_t frequency)
        : storage_index(storage_index),
        frequency(frequency)
      {}

      storage_index_t storage_index;
      frequency_t frequency;
    };

    // This struct represents the existance of a document stored in the
    // database. A list of database documents is maintained in this class
    // (m_document_storage), and each document is assigned a unique storage
    // index. This struct stores the inverse magnitude of the histogram of words
    // with which it is currently associated.
    struct DatabaseDocument : public std::conditional<
      enable_document_modification &&
        !std::is_same<HistogramNormalization, histogram_normalization::None>::value,
      conditionally_enable::inverse_magnitudes::InverseMagnitudesEnabled,
      conditionally_enable::inverse_magnitudes::InverseMagnitudesDisabled>::type
    {
      DatabaseDocument(const document_id_t document_id)
        : document_id(document_id)
      {}

      document_id_t document_id;
    };

    // This struct conveniently stores a single descriptor.
    struct DescriptorStorage
    {
      typename Descriptor::DimensionType values[Descriptor::NumDimensions];
    };
};

} // namespace vocabulary_tree

#endif // VOCABULARY_TREE_STRUCTS_H

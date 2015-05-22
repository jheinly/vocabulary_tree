#pragma once
#ifndef VOCABULARY_TREE_H
#define VOCABULARY_TREE_H

#include <vocabulary_tree/vocabulary_tree_types.h>
#include <vocabulary_tree/vocabulary_tree_descriptor_types.h>
#include <vocabulary_tree/vocabulary_tree_histogram_normalization_types.h>
#include <vocabulary_tree/vocabulary_tree_initial_histogram_distance_types.h>
#include <vocabulary_tree/vocabulary_tree_refined_histogram_distance_types.h>
#include <vocabulary_tree/indexed_storage.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

/*
This file defines a vocabulary tree and an associated database, in which
documents (represented by set of words) can be inserted, and then retrieved
based on their similarity to a query document.

NOTE: For ease of programming, define the class in the following manner:
        typedef vocabulary_tree::VocabularyTree<
          vocabulary_tree::descriptor::Sift,
          vocabulary_tree::histogram_normalization::L1,
          vocabulary_tree::initial_histogram_distance::Intersect,
          vocabulary_tree::refined_histogram_distance::ChiSquared> VocTree;
      This allows an instance of the class to be declared as:
        VocTree voc_tree;
      Relevant types can also be accessed as follows:
        VocTree::word_t
        VocTree::WordHistogram
        VocTree::QueryResult
        etc.

TODO: How should adding and removing words from an existing document be
      handled? For instance, storing the words as a histogram will cause
      needless computation as the magnitude of the histogram will be computed
      if histogram normalization is enabled. Therefore, it is cheaper to just
      pass a vector of visual words. However, if there are many repeated words
      in this vector, then there will be needless updates to the document's
      magnitude. In addition, once multi-threading is enabled, there will be
      needless locking/unlocking of the word's inverted index.

TODO: Make sure that inverse_magnitude cannot be zero, as that would result in
      divide-by-zero errors. Therefore, enforce that a document has a non-zero
      number of words.

TODO: Add support for threading (making the functions thread-safe). Ideally,
      support would be enabled via a template argument.
*/

namespace vocabulary_tree {

template<
  typename Descriptor = descriptor::Sift,
  typename HistogramNormalization = histogram_normalization::L1,
  typename InitialHistogramDistance = initial_histogram_distance::Intersect,
  typename RefinedHistogramDistance = refined_histogram_distance::ChiSquared>
class VocabularyTree : public VocabularyTreeTypes
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

      static inline bool less(
        const HistogramEntry & a,
        const HistogramEntry & b)
      { return a.word < b.word; }

      word_t word;
      frequency_t frequency;
    };

    // This struct stores a histogram of words. Specifically, it contains a list
    // of words, and their occurrence frequency (as HistogramEntry objects). The
    // inverse magnitude of the histogram is also stored for use in histogram
    // normalization.
    struct WordHistogram
    {
      std::vector<HistogramEntry> histogram_entries;
      frequency_t inverse_magnitude;
    };

    // This struct a single query result, in that it contains a document's ID,
    // (which had previously been stored in the database) as well as its score
    // with respect to the query document. A higher score indicates a better
    // match.
    struct QueryResult
    {
      static inline bool initial_sort(
        const QueryResult & a,
        const QueryResult & b)
      { return a.score > b.score; }

      static inline bool refined_sort(
        const QueryResult & a,
        const QueryResult & b)
      { return a.score < b.score; }

      document_id_t document_id;
      frequency_t score;
    };

    ////////////////////////////////////////////////////////////////////////////
    // Public Initialization & Destruction

    // Default constructor.
    VocabularyTree();

    // Destructor.
    ~VocabularyTree();

    // Load a vocabulary from file, where the file uses Noah Snavely's
    // VocabTree2 file format (https://github.com/snavely/VocabTree2).
    // NOTE: This function will ignore any stored leaf weights or inverted index
    //       entries in the loaded file.
    void load_vocabulary_from_file_snavely_vocab_tree_2_format(
      const std::string & vocabulary_file_path);

    ////////////////////////////////////////////////////////////////////////////
    // Public Processing Convenience Functions

    // Add a new document to the database.
    void add_document_to_database(
      const typename Descriptor::DimensionType * const descriptors,
      const index_t num_descriptors,
      const document_id_t new_document_id);

    // Add new words to an existing document in the database.
    void add_words_to_document(
      const typename Descriptor::DimensionType * const descriptors_to_add,
      const index_t num_descriptors_to_add,
      const document_id_t document_id);

    // Remove words from an existing document in the database.
    // NOTE: If all of the words for a document are removed, the document will
    //       not automatically be removed from the database.
    void remove_words_from_document(
      const typename Descriptor::DimensionType * const descriptors_to_remove,
      const index_t num_descriptors_to_remove,
      const document_id_t document_id);

    // Query the database by returning a list of the most similar documents.
    void query_database(
      const typename Descriptor::DimensionType * const query_descriptors,
      const index_t num_query_descriptors,
      const index_t max_num_results,
      std::vector<QueryResult> & query_results) const;

    ////////////////////////////////////////////////////////////////////////////
    // Public Processing Functions

    // Convert a list of descriptors into a list of words.
    void compute_words(
      const typename Descriptor::DimensionType * const descriptors,
      const index_t num_descriptors,
      std::vector<word_t> & words) const;

    // Convert a list of words into a word histogram.
    void compute_word_histogram(
      const std::vector<word_t> & words,
      WordHistogram & word_histogram) const;

    // Add a new document to the database.
    void add_document_to_database(
      const WordHistogram & word_histogram,
      const document_id_t new_document_id);

    // Remove an existing document from the database.
    void remove_document_from_database(
      const document_id_t document_id);

    // Test whether a document is currently stored in the database.
    bool is_document_in_database(
      const document_id_t document_id) const;

    // Add new words to an existing document in the database.
    void add_words_to_document(
      const WordHistogram & histogram_of_words_to_add,
      const document_id_t document_id);

    // Remove words from an existing document in the database.
    // NOTE: If all of the words for a document are removed, the document will
    //       not automatically be removed from the database.
    void remove_words_from_document(
      const WordHistogram & histogram_of_words_to_remove,
      const document_id_t document_id);

    // Query the database by returning a list of the most similar documents.
    void query_database(
      const WordHistogram & query_word_histogram,
      std::vector<QueryResult> & query_results,
      const index_t max_num_results) const;

    // Refine the list of initial query results.
    void refine_query_results(
      const WordHistogram & query_word_histogram,
      std::vector<QueryResult> & query_results,
      const index_t num_refined_results);

    // Compute and update the IDF (inverse document frequency) weights for the
    // words in the database.
    void compute_idf_weights();

    // Reset the IDF (inverse document frequency) weights to 1.0 for the words
    // in the database.
    void reset_idf_weights();

    // Remove all stored documents in the database.
    void clear_database();

    // Get the number of words in the currently loaded vocabulary.
    inline index_t num_words_in_vocabulary() const
    { return m_num_words_in_vocabulary; }

    // Get the number of documents currently stored in the database.
    inline index_t num_documents_in_database() const
    { return m_document_storage.num_entries(); }

  private:
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
    struct DatabaseDocument
    {
      DatabaseDocument(
        const document_id_t document_id,
        const WordHistogram & word_histogram)
      : document_id(document_id),
        word_histogram(word_histogram)
      {}

      document_id_t document_id;
      WordHistogram word_histogram;
    };

    // This struct conveniently stores a single descriptor.
    struct DescriptorStorage
    {
      typename Descriptor::DimensionType values[Descriptor::NumDimensions];
    };

    ////////////////////////////////////////////////////////////////////////////
    // Private Operators

    // Prevent this class from being copied or assigned.
    VocabularyTree(const VocabularyTree &);
    VocabularyTree & operator=(const VocabularyTree &);

    ////////////////////////////////////////////////////////////////////////////
    // Private Functions

    // Given an input descriptor and an interior node, this function returns the
    // index of the child node with which the descriptor is most similar.
    index_t compute_best_child_node(
      const typename Descriptor::DimensionType * const descriptor,
      const Node & node) const;

    // This function assists in loading a vocabulary stored in a VocabTree2 file
    // format (load_vocabulary_from_file_snavely_vocab_tree_2_format).
    void load_vocabulary_from_file_snavely_vocab_tree_2_format_helper(
      FILE * file,
      const index_t node_index,
      const int branch_factor,
      word_t * const next_available_word);

    ////////////////////////////////////////////////////////////////////////////
    // Private Variables

    // The number of words in the current vocabulary.
    index_t m_num_words_in_vocabulary;

    // A list of nodes that make up the vocabulary tree. The first node is the
    // root (m_nodes[0]), and a node's children are stored contiguously in the
    // vector.
    std::vector<Node> m_nodes;

    // The set of descriptors that make up the nodes in the vocabulary tree.
    // A node's index directly corresponds to the index of its descriptor.
    // For example, m_nodes[1] stores its descriptor in m_descriptors[1].
    std::vector<DescriptorStorage> m_descriptors;

    // Each word will have a list of the inverted index entries (documents)
    // that belong to it.
    std::vector<std::vector<InvertedIndexEntry> > m_word_inverted_indices;

    // Each word will be assigned an IDF weight (default is 1.0).
    std::vector<frequency_t> m_word_idf_weights;

    // Each document stored in the database will have an entry in
    // m_document_storage.
    IndexedStorage<DatabaseDocument, storage_index_t> m_document_storage;

    // A mapping between a document ID and an existing document storage index.
    std::unordered_map<document_id_t, storage_index_t> m_document_to_storage_indices;

    ////////////////////////////////////////////////////////////////////////////
    // Private Temporary Variables

    // This is used to create a histogram from a list of words. Its size is
    // equal to the number of words in the vocabulary, and when not in use,
    // each element is equal to zero.
    mutable std::vector<int> m_histogram_counts;

    // This is used to keep track of the distances between a query document and
    // all of the database document.
    mutable std::vector<InitialHistogramDistance> m_histogram_distances;

    // This is used in the convenience functions.
    mutable std::vector<word_t> m_words;

    // This is used in the convenience functions.
    mutable WordHistogram m_word_histogram;
};

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
VocabularyTree()
: m_num_words_in_vocabulary(0)
{}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
~VocabularyTree()
{
  m_descriptors.clear();
  m_nodes.clear();
  m_word_inverted_indices.clear();
  m_word_idf_weights.clear();
  m_document_storage.clear();
  m_document_to_storage_indices.clear();
  m_histogram_counts.clear();
  m_histogram_distances.clear();
  m_words.clear();
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
load_vocabulary_from_file_snavely_vocab_tree_2_format(
  const std::string & vocabulary_file_path)
{
  FILE * file = nullptr;
#ifdef WIN32
  fopen_s(&file, vocabulary_file_path.c_str(), "rb");
#else
  file = fopen(vocabulary_file_path.c_str(), "rb");
#endif

  if (file == nullptr)
  {
    std::cerr << "ERROR: failed to open file," << std::endl;
    std::cerr << vocabulary_file_path << std::endl;
    throw std::runtime_error("Failed to open vocabulary tree file");
  }

  int branch_factor = 0;
  fread(&branch_factor, sizeof(int), 1, file);

  int tree_depth = 0;
  fread(&tree_depth, sizeof(int), 1, file);

  int descriptor_dimension = 0;
  fread(&descriptor_dimension, sizeof(int), 1, file);

  if (descriptor_dimension != Descriptor::NumDimensions)
  {
    throw std::runtime_error(
      "Loaded descriptor_dimension does not match Descriptor::NumDimensions");
  }

  // Example: branch_factor = 10
  //   tree_depth = 0; num_nodes =        11; num_leaves =        10
  //   tree_depth = 1; num_nodes =       111; num_leaves =       100
  //   tree_depth = 2; num_nodes =     1,111; num_leaves =     1,000
  //   tree_depth = 3; num_nodes =    11,111; num_leaves =    10,000
  //   tree_depth = 4; num_nodes =   111,111; num_leaves =   100,000
  //   tree_depth = 5; num_nodes = 1,111,111; num_leaves = 1,000,000
  index_t num_nodes_estimate = 1;
  index_t num_leaves_estimate = 1;
  for (int i = 0; i < tree_depth + 1; ++i)
  {
    num_leaves_estimate *= branch_factor;
    num_nodes_estimate += num_leaves_estimate;
  }

  m_descriptors.clear();
  m_descriptors.resize(num_nodes_estimate);

  m_nodes.clear();
  m_nodes.reserve(num_nodes_estimate);

  // Initialize the root node.
  m_nodes.push_back(Node());
  const index_t starting_index_for_children =
    static_cast<index_t>(m_nodes.size());
  m_nodes[0].starting_index_for_children = starting_index_for_children;

  // NOTE: Unused for root node.
  char interior = 0;
  fread(&interior, sizeof(char), 1, file);

  // NOTE: Unused for root node.
  fread(
    m_descriptors[0].values,
    sizeof(typename Descriptor::DimensionType),
    Descriptor::NumDimensions,
    file);
  // After reading in the descriptor for the root node, set it to zero as it is
  // unused in any further operation.
  memset(
    m_descriptors[0].values,
    0,
    Descriptor::NumDimensions * sizeof(typename Descriptor::DimensionType));

  // NOTE: Unused for root node.
  float weight = 0;
  fread(&weight, sizeof(float), 1, file);

  // Read in the flags that indicate which children exist.
  std::vector<char> children_flags;
  children_flags.resize(branch_factor);
  fread(&children_flags[0], sizeof(char), branch_factor, file);

  // Count the number of children that exist, and reserve space for them in the
  // vector of Nodes.
  unsigned int num_children = 0;
  for (int i = 0; i < branch_factor; ++i)
  {
    if (children_flags[i] != 0)
    {
      ++num_children;
      m_nodes.push_back(Node());
    }
  }

  m_nodes[0].num_children = num_children;

  word_t next_available_word = 0;

  // For each of the children, call the helper function to read their
  // information from file.
  index_t child_offset = 0;
  for (int i = 0; i < branch_factor; ++i)
  {
    if (children_flags[i] != 0)
    {
      load_vocabulary_from_file_snavely_vocab_tree_2_format_helper(
        file,
        starting_index_for_children + child_offset,
        branch_factor,
        &next_available_word);
      ++child_offset;
    }
  }

  m_num_words_in_vocabulary = next_available_word;
  m_descriptors.resize(m_nodes.size());

  clear_database();
  reset_idf_weights();

  m_histogram_counts.clear();
  m_histogram_counts.resize(m_num_words_in_vocabulary, 0);

  fclose(file);
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
add_document_to_database(
  const typename Descriptor::DimensionType * const descriptors,
  const index_t num_descriptors,
  const document_id_t new_document_id)
{
  compute_words(
    descriptors,
    num_descriptors,
    m_words);
  compute_word_histogram(
    m_words,
    m_word_histogram);
  add_document_to_database(
    m_word_histogram,
    new_document_id);
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
add_words_to_document(
  const typename Descriptor::DimensionType * const descriptors_to_add,
  const index_t num_descriptors_to_add,
  const document_id_t document_id)
{
  compute_words(
    descriptors_to_add,
    num_descriptors_to_add,
    m_words);
  compute_word_histogram(
    m_words,
    m_word_histogram);
  add_words_to_document(
    m_word_histogram,
    document_id);
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
remove_words_from_document(
  const typename Descriptor::DimensionType * const descriptors_to_remove,
  const index_t num_descriptors_to_remove,
  const document_id_t document_id)
{
  compute_words(
    descriptors_to_remove,
    num_descriptors_to_remove,
    m_words);
  compute_word_histogram(
    m_words,
    m_word_histogram);
  remove_words_from_document(
    m_word_histogram,
    document_id);
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
query_database(
  const typename Descriptor::DimensionType * const query_descriptors,
  const index_t num_query_descriptors,
  const index_t max_num_results,
  std::vector<QueryResult> & query_results) const
{
  compute_words(
    query_descriptors,
    num_query_descriptors,
    m_words);
  compute_word_histogram(
    m_words,
    m_word_histogram);
  query_database(
    m_word_histogram,
    max_num_results,
    query_results);
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
compute_words(
  const typename Descriptor::DimensionType * const descriptors,
  const index_t num_descriptors,
  std::vector<word_t> & words) const
{
  words.resize(num_descriptors);

  // Iterate over each descriptor in the input set.
  for (index_t descriptor_idx = 0; descriptor_idx < num_descriptors; ++descriptor_idx)
  {
    // Get a pointer to the current descriptor.
    const typename Descriptor::DimensionType * const descriptor =
      descriptors + descriptor_idx * Descriptor::NumDimensions;

    // Start traversal at the root node.
    index_t current_node_idx = 0;

    // Keep traversing the tree's nodes until the descriptor is assigned a word.
    for (;;)
    {
      // Get a reference to the current node.
      const Node & node = m_nodes[current_node_idx];

      // If the current node has no children, assign this node's visual word to
      // the descriptor and break out of the traversal.
      if (node.num_children == 0)
      {
        words[descriptor_idx] = node.word;
        break;
      }

      // Traverse to the best-matching child node.
      current_node_idx = compute_best_child_node(descriptor, node);
    }
  }
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
compute_word_histogram(
  const std::vector<word_t> & words,
  WordHistogram & word_histogram) const
{
  word_histogram.histogram_entries.clear();

  // Create a dense histogram of the input visual words (m_histogram_counts has
  // a size equal to the number of words in the vocabulary).
  for (const auto word : words)
  {
    ++m_histogram_counts[word];
  }

  HistogramNormalization normalization;

  // Iterate through each input word.
  for (const auto word : words)
  {
    // If the input word was not already processed, create a histogram entry
    // for it in word_histogram.
    if (m_histogram_counts[word] != 0)
    {
      const frequency_t frequency =
        static_cast<frequency_t>(m_histogram_counts[word]);
      word_histogram.histogram_entries.push_back(
        HistogramEntry(word, frequency));
      m_histogram_counts[word] = 0;

      // Keep track of the histogram's magnitude.
      normalization.add_term(frequency);
    }
  }

  // Compute the inverse magnitude.
  word_histogram.inverse_magnitude =
    static_cast<frequency_t>(1.0 / normalization.compute_magnitude());

  // Sort the histogram entries.
  std::sort(
    word_histogram.histogram_entries.begin(),
    word_histogram.histogram_entries.end(),
    HistogramEntry::less);
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
add_document_to_database(
  const WordHistogram & word_histogram,
  const document_id_t new_document_id)
{
  // Add the new document to the indexed storage data structure.
  if (is_document_in_database(new_document_id))
  {
    throw std::runtime_error(
      "VocabularyTree:add_document_to_database called with existing document_id");
  }
  const storage_index_t storage_index = m_document_storage.add(
    DatabaseDocument(new_document_id, word_histogram));

  // Update the mapping between document ids and storage indices.
  m_document_to_storage_indices[new_document_id] = storage_index;

  // Iterate through the document's histogram entries.
  for (const auto & histogram_entry : word_histogram.histogram_entries)
  {
    // Create an entry in the inverted index for the current histogram entry.
    // This stores that the word appeared in the document.
    m_word_inverted_indices[histogram_entry.word].push_back(
      InvertedIndexEntry(storage_index, histogram_entry.frequency));
  }
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
remove_document_from_database(
  const document_id_t document_id)
{
  // Find the storage index for this document.
  const auto & found_iter = m_document_to_storage_indices.find(document_id);
  if (found_iter == m_document_to_storage_indices.end())
  {
    throw std::runtime_error(
      "VocabularyTree::remove_document_from_database called with non-existent document_id");
  }
  const storage_index_t storage_index = found_iter->second;
  const WordHistogram & word_histogram =
    m_document_storage[storage_index].word_histogram;

  // Iterate through the histogram entries for this document.
  for (const auto & histogram_entry : word_histogram.histogram_entries)
  {
    // Find the inverted index entry for this histogram entry, and erase it.
    std::vector<InvertedIndexEntry> & current_inverted_indices =
      m_word_inverted_indices[histogram_entry.word];
    bool found = false;
    for (auto iter = current_inverted_indices.begin();
      iter != current_inverted_indices.end();
      ++iter)
    {
      if (iter->storage_index == storage_index)
      {
        current_inverted_indices.erase(iter);
        found = true;
        break;
      }
    }
    if (!found)
    {
      throw std::runtime_error(
        "VocabularyTree::remove_document_from_database could not find inverted index entry for document");
    }
  }

  // Remove the document from storage.
  m_document_to_storage_indices.erase(document_id);
  m_document_storage.remove(storage_index);
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
bool VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
is_document_in_database(
  const document_id_t document_id) const
{
  const auto found = m_document_to_storage_indices.find(document_id);
  if (found == m_document_to_storage_indices.end())
  {
    return false;
  }
  else
  {
    return true;
  }
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
add_words_to_document(
  const WordHistogram & histogram_of_words_to_add,
  const document_id_t document_id)
{
  // Find the storage index for this document.
  const auto & found_iter = m_document_to_storage_indices.find(document_id);
  if (found_iter == m_document_to_storage_indices.end())
  {
    throw std::runtime_error(
      "VocabularyTree::add_words_to_document called with non-existent document_id");
  }
  const storage_index_t storage_index = found_iter->second;

  WordHistogram & existing_word_histogram =
    m_document_storage[storage_index].word_histogram;

  const frequency_t initial_magnitude = static_cast<frequency_t>(
    1.0 / existing_word_histogram.inverse_magnitude);
  HistogramNormalization normalization(initial_magnitude);

  std::vector<HistogramEntry> & existing_histogram_entries =
    existing_word_histogram.histogram_entries;
  const std::vector<HistogramEntry> & histogram_entries_to_add =
    histogram_of_words_to_add.histogram_entries;

  size_t existing_entry_idx = 0;
  size_t entry_to_add_idx = 0;

  // Both word histograms are sorted based on the word indices, so iterate
  // through them similar to how merge sort would combine two sorted lists.
  while (existing_entry_idx < existing_histogram_entries.size() &&
    entry_to_add_idx < histogram_entries_to_add.size())
  {
    HistogramEntry & existing_entry =
      existing_histogram_entries[existing_entry_idx];
    const HistogramEntry & entry_to_add =
      histogram_entries_to_add[entry_to_add_idx];

    // If the word to add is less than the existing word, we have found the
    // position where the new word should be inserted.
    if (entry_to_add.word < existing_entry.word)
    {
      normalization.add_term(entry_to_add.frequency);
      
      // Add the new entry to the existing word histogram in storage.
      existing_histogram_entries.insert(
        existing_histogram_entries.begin() + existing_entry_idx,
        entry_to_add);
      // Create a new inverted index entry for this word.
      m_word_inverted_indices[entry_to_add.word].push_back(
        InvertedIndexEntry(storage_index, entry_to_add.frequency));

      // Advance to the next word to add, as well as advance the existing entry
      // index as we have just inserted an element.
      ++existing_entry_idx;
      ++entry_to_add_idx;
    }
    else if (entry_to_add.word > existing_entry.word)
    {
      // We have not found a valid insert position, so advance to the next entry
      // in the existing word histogram.
      ++existing_entry_idx;
    }
    else // if (entry_to_add.word == existing_entry.word)
    {
      // If the words are equal, we can add their frequencies.
      const frequency_t updated_frequency =
        existing_entry.frequency + entry_to_add.frequency;
      
      normalization.update_term(existing_entry.frequency, updated_frequency);

      // Update the entry in the existing word histogram in storage.
      existing_entry.frequency = updated_frequency;

      // Update the entry in the inverted index.
      std::vector<InvertedIndexEntry> & current_inverted_indices =
        m_word_inverted_indices[entry_to_add.word];
      bool found = false;
      for (auto & inverted_index_entry : current_inverted_indices)
      {
        if (inverted_index_entry.storage_index == storage_index)
        {
          inverted_index_entry.frequency = updated_frequency;
          found = true;
          break;
        }
      }
      if (!found)
      {
        // There was a matching entry in the existing word histogram, but we
        // failed to find a matching entry in the inverted index.
        throw std::runtime_error(
          "VocabularyTree::add_words_to_document could not find inverted index entry");
      }
      
      // Advance to the next word to add as well as the next existing entry
      // index.
      ++existing_entry_idx;
      ++entry_to_add_idx;
    }
  }
  // The previous loop could have ended because we reached the end of the
  // existing word histogram. So, loop over any remaining entries to add.
  while (entry_to_add_idx < histogram_entries_to_add.size())
  {
    const HistogramEntry & entry_to_add =
      histogram_entries_to_add[entry_to_add_idx];

    normalization.add_term(entry_to_add.frequency);

    // Add the entry to the existing word histogram in storage.
    existing_histogram_entries.push_back(entry_to_add);
    // Create a new inverted index entry for this word.
    m_word_inverted_indices[entry_to_add.word].push_back(
      InvertedIndexEntry(storage_index, entry_to_add.frequency));

    ++entry_to_add_idx;
  }

  existing_word_histogram.inverse_magnitude =
    static_cast<frequency_t>(1.0 / normalization.compute_magnitude());
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
remove_words_from_document(
  const WordHistogram & histogram_of_words_to_remove,
  const document_id_t document_id)
{
  // Find the storage index for this document.
  const auto & found_iter = m_document_to_storage_indices.find(document_id);
  if (found_iter == m_document_to_storage_indices.end())
  {
    throw std::runtime_error(
      "VocabularyTree::remove_words_from_document called with non-existent document_id");
  }
  const storage_index_t storage_index = found_iter->second;

  WordHistogram & existing_word_histogram =
    m_document_storage[storage_index].word_histogram;

  const frequency_t initial_magnitude = static_cast<frequency_t>(
    1.0 / existing_word_histogram.inverse_magnitude);
  HistogramNormalization normalization(initial_magnitude);

  std::vector<HistogramEntry> & existing_histogram_entries =
    existing_word_histogram.histogram_entries;
  const std::vector<HistogramEntry> & histogram_entries_to_remove =
    histogram_of_words_to_remove.histogram_entries;

  size_t existing_entry_idx = 0;
  size_t entry_to_remove_idx = 0;

  // Both word histograms are sorted based on the word indices, so iterate
  // through them similar to how merge sort would combine two sorted lists.
  while (existing_entry_idx < existing_histogram_entries.size() &&
    entry_to_remove_idx < histogram_entries_to_remove.size())
  {
    HistogramEntry & existing_entry =
      existing_histogram_entries[existing_entry_idx];
    const HistogramEntry & entry_to_remove =
      histogram_entries_to_remove[entry_to_remove_idx];

    if (entry_to_remove.word == existing_entry.word)
    {
      // If the words are equal, subtract the frequency to remove from the
      // existing frequency.
      const frequency_t updated_frequency =
        existing_entry.frequency - entry_to_remove.frequency;

      normalization.update_term(existing_entry.frequency, updated_frequency);

      if (updated_frequency == 0) // TODO: Make sure that this comparison works.
      {
        // Erase the existing histogram entry.
        existing_histogram_entries.erase(
          existing_histogram_entries.begin() + existing_entry_idx);

        // Erase the entry in the inverted index.
        std::vector<InvertedIndexEntry> & current_inverted_indices =
          m_word_inverted_indices[entry_to_remove.word];
        bool found = false;
        for (auto iter = current_inverted_indices.begin();
          iter != current_inverted_indices.end();
          ++iter)
        {
          if (iter->storage_index == storage_index)
          {
            current_inverted_indices.erase(iter);
            found = true;
            break;
          }
        }
        if (!found)
        {
          // There was a matching entry in the existing word histogram, but we
          // failed to find a matching entry in the inverted index.
          throw std::runtime_error(
            "VocabularyTree::remove_words_from_document could not find inverted index entry");
        }

        // Advance to the next entry to remove. We don't need to advance the
        // existing entry index as we have just removed the entry at the current
        // position.
        ++entry_to_remove_idx;
      }
      else if (updated_frequency > 0)
      {
        // Update the frequency in the existing word histogram in storage.
        existing_entry.frequency = updated_frequency;

        // Update the entry in the inverted index.
        std::vector<InvertedIndexEntry> & current_inverted_indices =
          m_word_inverted_indices[entry_to_remove.word];
        bool found = false;
        for (auto & inverted_index_entry : current_inverted_indices)
        {
          if (inverted_index_entry.storage_index == storage_index)
          {
            inverted_index_entry.frequency = updated_frequency;
            found = true;
            break;
          }
        }
        if (!found)
        {
          // There was a matching entry in the existing word histogram, but we
          // failed to find a matching entry in the inverted index.
          throw std::runtime_error(
            "VocabularyTree::remove_words_from_document could not find inverted index entry");
        }

        // Advace to the next word to remove, as well as advance the existing
        // entry index as we have just updated the current existing entry.
        ++existing_entry_idx;
        ++entry_to_remove_idx;
      }
      else // if (updated_frequency < 0)
      {
        throw std::runtime_error(
          "VocabularyTree::remove_words_from_document could not remove the requested frequency");
      }
    }
    else if (entry_to_remove.word > existing_entry.word)
    {
      // We have not found the matching existing word yet, so advance to the
      // next entry in the existing word histogram.
      ++existing_entry_idx;
    }
    else // if (entry_to_remove.word < existing_entry.word)
    {
      // If the word to add is less than the existing word, then there is an
      // error as the word to be removed should have been contained within the
      // existing word histogram.
      throw std::runtime_error(
        "VocabularyTree::remove_words_from_document could not find existing word entry");
    }
  }
  if (entry_to_remove_idx < histogram_entries_to_remove.size())
  {
    // If we've processed all of the existing word histogram entries, but there
    // are still words to remove, there is an error.
    throw std::runtime_error(
      "VocabularyTree::remove_words_from_document could not find existing word entry");
  }

  existing_word_histogram.inverse_magnitude =
    static_cast<frequency_t>(1.0 / normalization.compute_magnitude());
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
query_database(
  const WordHistogram & query_word_histogram,
  std::vector<QueryResult> & query_results,
  const index_t max_num_results) const
{
  const storage_index_t document_storage_capacity =
    m_document_storage.capacity();
  m_histogram_distances.resize(document_storage_capacity);
  for (storage_index_t i = 0; i < document_storage_capacity; ++i)
  {
    m_histogram_distances[i].reset();
  }

  for (const auto & histogram_entry : query_word_histogram.histogram_entries)
  {
    const frequency_t query_frequency =
      histogram_entry.frequency * query_word_histogram.inverse_magnitude;

    const frequency_t idf_weight = m_word_idf_weights[histogram_entry.word];

    const std::vector<InvertedIndexEntry> & current_inverted_indices =
      m_word_inverted_indices[histogram_entry.word];
    for (const auto & inverted_index_entry : current_inverted_indices)
    {
      const storage_index_t storage_index = inverted_index_entry.storage_index;
      const frequency_t frequency =
        inverted_index_entry.frequency *
        m_document_storage[storage_index].word_histogram.inverse_magnitude;

      m_histogram_distances[storage_index].add_term(
        query_frequency, frequency, idf_weight);
    }
  }

  query_results.resize(document_storage_capacity);
  for (storage_index_t i = 0; i < document_storage_capacity; ++i)
  {
    query_results[i].document_id = m_document_storage[i].document_id;
    query_results[i].score = m_histogram_distances[i].compute_magnitude();
  }

  // Determine the number of results that can be returned.
  const index_t num_results =
    std::min<index_t>(max_num_results, m_document_storage.num_entries());
  
  // Find the num_results best query results.
  std::partial_sort(
    query_results.begin(),
    query_results.begin() + num_results,
    query_results.end(),
    QueryResult::initial_sort);
  query_results.resize(num_results);

  // Remove any database results that have 0 overlap with the query document.
  for (index_t i = 0; i < num_results; ++i)
  {
    if (query_results[i].score == 0) // TODO: Make sure this comparison works.
    {
      query_results.resize(i);
      break;
    }
  }
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
refine_query_results(
  const WordHistogram & query_word_histogram,
  std::vector<QueryResult> & query_results,
  const index_t num_refined_results)
{
  for (auto & query_result : query_results)
  {
    // Find the storage index for this query result.
    const auto & found_iter =
      m_document_to_storage_indices.find(query_result.document_id);
    if (found_iter == m_document_to_storage_indices.end())
    {
      throw std::runtime_error(
        "VocabularyTree::refine_query_results could not find document");
    }
    const storage_index_t storage_index = found_iter->second;
    const WordHistogram & result_word_histogram =
      m_document_storage[storage_index].word_histogram;

    RefinedHistogramDistance refined_distance;

    // TODO: Iterate through the histograms and update refined_distance.
    // TODO: Make sure to use the inverse magnitudes and IDF weights.
    // TODO: Store the resulting distance back into query_result.
  }

  // Determine the number of results that can be returned.
  const index_t num_results = std::min<index_t>(
    num_refined_results,
    static_cast<index_t>(query_results.size()));

  // Find the num_results best query results.
  std::partial_sort(
    query_results.begin(),
    query_results.begin() + num_results,
    query_results.end(),
    QueryResult::refined_sort);
  query_results.resize(num_results);
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
compute_idf_weights()
{
  // IDF weight for a word
  //   = log(# Documents in Database / # Documents with Word)
  //   = log(# Documents in Database) - log(# Documents with Word)
  // By splitting the computation into its subtraction form, we avoid a division
  // operation on each iteration.

  // If there are no documents in the database, reset the idf weights.
  if (num_documents_in_database() == 0)
  {
    reset_idf_weights();
    return;
  }

  const frequency_t database_weight =
    log(static_cast<frequency_t>(num_documents_in_database()));

  for (index_t i = 0; i < m_num_words_in_vocabulary; ++i)
  {
    const index_t num_documents_with_word =
      static_cast<index_t>(m_word_inverted_indices[i].size());
    if (num_documents_with_word > 0)
    {
      const frequency_t word_weight =
        log(static_cast<frequency_t>(num_documents_with_word));
      m_word_idf_weights[i] = database_weight - word_weight;
    }
    else
    {
      m_word_idf_weights[i] = 0;
    }
  }
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
reset_idf_weights()
{
  m_word_idf_weights.resize(m_num_words_in_vocabulary);
  for (index_t i = 0; i < m_num_words_in_vocabulary; ++i)
  {
    m_word_idf_weights[i] = 1;
  }
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
clear_database()
{
  m_word_inverted_indices.resize(m_num_words_in_vocabulary);
  for (index_t i = 0; i < m_num_words_in_vocabulary; ++i)
  {
    m_word_inverted_indices[i].clear();
  }
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
VocabularyTreeTypes::index_t VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
compute_best_child_node(
  const typename Descriptor::DimensionType * const descriptor,
  const Node & node) const
{
  index_t best_idx = 0;
  typename Descriptor::DistanceType best_distance = Descriptor::WorstDistance;

  // Iterate over this node's children, and keep track of the best match
  // (the child with the smallest distance to the current descriptor).
  for (index_t i = 0; i < node.num_children; ++i)
  {
    const typename Descriptor::DistanceType distance = Descriptor::compute_distance(
      descriptor,
      m_descriptors[node.starting_index_for_children + i].values);
    if (Descriptor::is_first_distance_better(distance, best_distance))
    {
      best_idx = i;
      best_distance = distance;
    }
  }

  // Return the index of the best-matching child node.
  return node.starting_index_for_children + best_idx;
}

template<
  typename Descriptor,
  typename HistogramNormalization,
  typename InitialHistogramDistance,
  typename RefinedHistogramDistance>
void VocabularyTree<
  Descriptor,
  HistogramNormalization,
  InitialHistogramDistance,
  RefinedHistogramDistance>::
load_vocabulary_from_file_snavely_vocab_tree_2_format_helper(
  FILE * file,
  const index_t node_index,
  const int branch_factor,
  word_t * const next_available_word)
{
  // Read the flag which indicates whether this node is an interior node or a
  // leaf node.
  char interior = 0;
  fread(&interior, sizeof(char), 1, file);

  // Read this node's descriptor into its corresponding position within the
  // descriptors vector.
  fread(
    m_descriptors[node_index].values,
    sizeof(typename Descriptor::DimensionType),
    Descriptor::NumDimensions,
    file);

  // NOTE: Node weights are ignored.
  float weight = 0;
  fread(&weight, sizeof(float), 1, file);

  if (interior == 1)
  {
    // We have reached an interior node.

    const index_t starting_index_for_children =
      static_cast<index_t>(m_nodes.size());
    m_nodes[node_index].starting_index_for_children =
      starting_index_for_children;

    // Read in the flags that indicate which children exist.
    std::vector<char> children_flags;
    children_flags.resize(branch_factor);
    fread(&children_flags[0], sizeof(char), branch_factor, file);

    // Count the number of children that exist, and reserve space for them in
    // the vector of Nodes.
    unsigned int num_children = 0;
    for (int i = 0; i < branch_factor; ++i)
    {
      if (children_flags[i] != 0)
      {
        ++num_children;
        m_nodes.push_back(Node());
      }
    }

    m_nodes[node_index].num_children = num_children;

    // Recursively call the helper function on this node's children to read the
    // information from file.
    index_t child_offset = 0;
    for (int i = 0; i < branch_factor; ++i)
    {
      if (children_flags[i] != 0)
      {
        load_vocabulary_from_file_snavely_vocab_tree_2_format_helper(
          file,
          starting_index_for_children + child_offset,
          branch_factor,
          next_available_word);
        ++child_offset;
      }
    }
  }
  else // if (interior == 0)
  {
    // We have reached a leaf node.

    m_nodes[node_index].num_children = 0;

    // Assign the next available word to this node.
    m_nodes[node_index].word = *next_available_word;
    ++(*next_available_word);

    // NOTE: Any stored image data is ignored.
    int num_images = 0;
    fread(&num_images, sizeof(int), 1, file);
    for (int i = 0; i < num_images; ++i)
    {
      int image = 0;
      fread(&image, sizeof(int), 1, file);
      float count = 0;
      fread(&count, sizeof(float), 1, file);
    }
  }
}

} // namespace vocabulary_tree

#endif // VOCABULARY_TREE_H

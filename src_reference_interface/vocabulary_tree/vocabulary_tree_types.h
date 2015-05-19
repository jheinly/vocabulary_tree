#pragma once
#ifndef VOCABULARY_TREE_TYPES_H
#define VOCABULARY_TREE_TYPES_H

namespace vocabulary_tree {

// This class is used to store types definitions and static const values
// associated with the main VocabularyTree class and its use.
class VocabularyTreeTypes
{
  public:
    typedef unsigned int word_t;
    typedef unsigned int document_id_t;
    typedef float frequency_t;
    typedef unsigned int index_t;

  protected:
    typedef unsigned int storage_index_t;

    static const word_t InvalidWord;
    static const index_t InvalidIndex;
};

} // namespace vocabulary_tree

#endif // VOCABULARY_TREE_TYPES_H

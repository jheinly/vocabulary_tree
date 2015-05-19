#include <vocabulary_tree/vocabulary_tree_types.h>
#include <limits>

const vocabulary_tree::VocabularyTreeTypes::word_t
  vocabulary_tree::VocabularyTreeTypes::InvalidWord =
    std::numeric_limits<vocabulary_tree::VocabularyTreeTypes::word_t>::max();

const vocabulary_tree::VocabularyTreeTypes::index_t
  vocabulary_tree::VocabularyTreeTypes::InvalidIndex =
    std::numeric_limits<vocabulary_tree::VocabularyTreeTypes::index_t>::max();

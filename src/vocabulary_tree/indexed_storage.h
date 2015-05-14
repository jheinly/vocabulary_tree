#pragma once
#ifndef INDEXED_STORAGE_H
#define INDEXED_STORAGE_H

#include <vector>

namespace vocabulary_tree {

/*
This file defines the IndexedStorage container class, where entries are assigned
a unique storage index upon their insertion. Adding, accessing, and removing
entries are all designed to be efficient operations.
*/

template<typename T, typename IndexType = size_t>
class IndexedStorage
{
  public:
    // Default constructor.
    IndexedStorage()
    : m_num_entries(0)
    {}
    
    // Destructor.
    ~IndexedStorage()
    {}
    
    // Return the current number of entries that are stored.
    inline IndexType num_entries() const
    { return m_num_entries; }

    // Return the maximum number of entries that could be stored without
    // potentially reallocating the internal storage. Additionally, all valid
    // storage indices should be less than this function's returned value.
    inline IndexType capacity() const
    { return static_cast<IndexType>(m_storage.size()); }
    
    // Access an existing entry via its storage index.
    inline const T & operator[](const IndexType index) const
    { return m_storage[index]; }
    
    // Access an existing entry via its storage index.
    inline T & operator[](const IndexType index)
    { return m_storage[index]; }
    
    // Add a new entry to storage, and return its unique storage index.
    IndexType add(const T & t)
    {
      ++m_num_entries;
      if (m_free_indices.size() > 0)
      {
        const IndexType index = m_free_indices.back();
        m_free_indices.pop_back();
        m_storage[index] = t;
        return index;
      }
      const IndexType index = static_cast<IndexType>(m_storage.size());
      m_storage.push_back(t);
      return index;
    }
    
    // Remove an existing entry from storage via its storage index.
    void remove(const IndexType index)
    {
      --m_num_entries;
      m_free_indices.push_back(index);
    }
    
    // Reserve storage for the provided number of elements.
    void reserve(const IndexType count)
    { m_storage.reserve(count); }
    
    // Remove all elements that are currently stored.
    void clear()
    {
      m_num_entries = 0;
      m_storage.clear();
      m_free_indices.clear();
    }
    
  private:
    // Prevent this class from being copied or assigned.
    IndexedStorage(const IndexedStorage &);
    IndexedStorage & operator=(const IndexedStorage &);
    
    // The current number of entries that are stored.
    IndexType m_num_entries;

    // A list containing the stored entries, where each entry is stored at its
    // assigned storage index. Not all of the list elements may be occupied by
    // a valid entry (the indices of the unoccupied elements is stored in
    // m_free_indices).
    std::vector<T> m_storage;

    // A list of the unoccupied element indices in the storage list (m_storage).
    std::vector<IndexType> m_free_indices;
};

} // namespace vocabulary_tree

#endif // INDEXED_STORAGE_H

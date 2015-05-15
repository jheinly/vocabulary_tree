# NOTE: This file should not be executed directly using CMake, but instead
#       should be referenced via find_package(vocabulary_tree). This will
#       request the location of the vocabulary_tree-config.cmake file contained
#       in this same directory.

cmake_minimum_required(VERSION 3.0)

# Create a static library module with the given source files.
cmh_add_library_module(
  src/vocabulary_tree/indexed_storage.h
  src/vocabulary_tree/vocabulary_tree.h
  src/vocabulary_tree/vocabulary_tree_conditionally_enable.h
  src/vocabulary_tree/vocabulary_tree_descriptor_types.h
  src/vocabulary_tree/vocabulary_tree_descriptor_types.cpp
  src/vocabulary_tree/vocabulary_tree_histogram_distance_types.h
  src/vocabulary_tree/vocabulary_tree_histogram_normalization_types.h
  src/vocabulary_tree/vocabulary_tree_types.h
  src/vocabulary_tree/vocabulary_tree_types.cpp
)

# Set the include directories necessary to use this module.
cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
)

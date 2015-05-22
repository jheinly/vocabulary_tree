# NOTE: This file should not be executed directly using CMake, but instead
#       should be referenced via find_package(vocabulary_tree). This will
#       request the location of the vocabulary_tree-config.cmake file contained
#       in this same directory.

cmake_minimum_required(VERSION 3.0)

# The conditionally enabled interface allows functionality of the library to be
# enabled or disabled via template arguments.
set(VOCABULARY_TREE_USE_CONDITIONALLY_ENABLED_INTERFACE TRUE
  CACHE BOOL "Whether or not to use the conditionally enabled interface to the code.")

if(VOCABULARY_TREE_USE_CONDITIONALLY_ENABLED_INTERFACE)
  # Create a static library module with the given source files.
  set(SRC_DIR src_conditionally_enabled_interface/vocabulary_tree)
  cmh_add_library_module(
    ${SRC_DIR}/indexed_storage.h
    ${SRC_DIR}/vocabulary_tree.h
    ${SRC_DIR}/vocabulary_tree_conditionally_enable.h
    ${SRC_DIR}/vocabulary_tree_descriptor_types.h
    ${SRC_DIR}/vocabulary_tree_descriptor_types.cpp
    ${SRC_DIR}/vocabulary_tree_histogram_distance_types.h
    ${SRC_DIR}/vocabulary_tree_histogram_normalization_types.h
    ${SRC_DIR}/vocabulary_tree_structs.h
    ${SRC_DIR}/vocabulary_tree_types.h
    ${SRC_DIR}/vocabulary_tree_types.cpp
  )
  unset(SRC_DIR)
  # Set the include directories necessary to use this module.
  cmh_target_include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/src_conditionally_enabled_interface
  )
else() # if(NOT VOCABULARY_TREE_USE_CONDITIONALLY_ENABLED_INTERFACE)
  # Create a static library module with the given source files.
  set(SRC_DIR src_reference_interface/vocabulary_tree)
  cmh_add_library_module(
    ${SRC_DIR}/indexed_storage.h
    ${SRC_DIR}/vocabulary_tree.h
    ${SRC_DIR}/vocabulary_tree_descriptor_types.h
    ${SRC_DIR}/vocabulary_tree_descriptor_types.cpp
    ${SRC_DIR}/vocabulary_tree_histogram_normalization_types.h
    ${SRC_DIR}/vocabulary_tree_initial_histogram_distance_types.h
    ${SRC_DIR}/vocabulary_tree_refined_histogram_distance_types.h
    ${SRC_DIR}/vocabulary_tree_types.h
    ${SRC_DIR}/vocabulary_tree_types.cpp
  )
  unset(SRC_DIR)
  # Set the include directories necessary to use this module.
  cmh_target_include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/src_reference_interface
  )
endif() # VOCABULARY_TREE_USE_CONDITIONALLY_ENABLED_INTERFACE

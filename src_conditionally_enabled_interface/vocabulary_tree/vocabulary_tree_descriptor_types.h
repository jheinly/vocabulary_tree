#pragma once
#ifndef VOCABULARY_TREE_DESCRIPTOR_TYPES_H
#define VOCABULARY_TREE_DESCRIPTOR_TYPES_H

/*
This file defines structs that can be used as template arguments for
VocabularyTree's Descriptor template argument. Conceptually, each struct defines
a particular representation for a descriptor (its storage type and dimension),
as well as a distance metric in order to compute distances between descriptors.

NOTE: SIFT_USE_DOT gives worse results than SIFT_USE_L2. However, SIFT_USE_SSE
      has no effect on the results (except for speed), so the issue is not in
      the SSE code. The issue could be that in order for the dot distance to
      work, the two descriptors are assumed to have the same length. However, by
      discretizing each dimension and storing it as an unsigned char, the length
      of the descriptors can vary.
*/

#define SIFT_USE_L2
#define SIFT_USE_SSE

#ifndef SIFT_USE_L2
  #define SIFT_USE_DOT
#endif

#ifdef SIFT_USE_SSE
  #include <emmintrin.h>
#endif

namespace vocabulary_tree {

namespace descriptor {

// SIFT descriptor that is stored using 128 unsigned chars.
struct Sift
{
  typedef unsigned char DimensionType;
  static const int NumDimensions = 128;

  typedef unsigned int DistanceType;
  static const DistanceType WorstDistance;

  static inline bool is_first_distance_better(
    const DistanceType distance1,
    const DistanceType distance2)
  {
#ifdef SIFT_USE_L2
    return distance1 < distance2;
#endif
#ifdef SIFT_USE_DOT
    return distance1 > distance2;
#endif
  }

  static inline DistanceType compute_distance(
    const DimensionType * const descriptor1,
    const DimensionType * const descriptor2)
  {
#ifdef SIFT_USE_L2
#ifdef SIFT_USE_SSE

    // Set 16, 8-bit values to zero.
    __m128i zero8 = _mm_set1_epi8(0);

    // Set 8, 16-bit values to zero.
    __m128i zero16 = _mm_set1_epi16(0);

    // Set 4, 32-bit values to zero.
    __m128i sum32 = _mm_set1_epi32(0);

    __m128i a8;
    __m128i b8;
    __m128i a16;
    __m128i b16;
    __m128i sub16;
    __m128i mul16;
    __m128i mul32;

    // We will process 16 values at a time.
    const int dim_16 = NumDimensions / 16;
    for (int i = 0; i < dim_16; ++i)
    {
      // Load 16, 8-bit values from each descriptor without assuming anything
      // about the alignment of the underlying pointers.
      const int offset = 16 * i;
      a8 = _mm_loadu_si128((__m128i *)(descriptor1 + offset));
      b8 = _mm_loadu_si128((__m128i *)(descriptor2 + offset));

      /////////////////////////////////////////////////
      // Process the lower 8 values of the descriptors.

      // Interleave the lower 8, 8-bit values of the descriptors with zeros,
      // effectively converting the 8 values to 16-bit values.
      a16 = _mm_unpacklo_epi8(a8, zero8);
      b16 = _mm_unpacklo_epi8(b8, zero8);

      // Compute the difference between the 8, 16-bit values.
      sub16 = _mm_sub_epi16(a16, b16);

      // Compute the squared difference between the 8, 16-bit values.
      mul16 = _mm_mullo_epi16(sub16, sub16);

      // Interleave the lower 4, 16-bit values with zeros to effectively
      // convert to 4, 32-bit values.
      mul32 = _mm_unpacklo_epi16(mul16, zero16);

      // Add the current 4 values to the running sum that stores 4, 32-bit
      // values.
      sum32 = _mm_add_epi32(sum32, mul32);

      // Interleave the upper 4, 16-bit values with zeros to effectively
      // convert to 4, 32-bit values.
      mul32 = _mm_unpackhi_epi16(mul16, zero16);

      // Add the current 4 values to the running sum that stores 4, 32-bit
      // values.
      sum32 = _mm_add_epi32(sum32, mul32);

      /////////////////////////////////////////////////
      // Process the upper 8 values of the descriptors.

      // Interleave the upper 8, 8-bit values of the descriptors with zeros,
      // effectively converting the 8 values to 16-bit values.
      a16 = _mm_unpackhi_epi8(a8, zero8);
      b16 = _mm_unpackhi_epi8(b8, zero8);

      // Compute the difference between the 8, 16-bit values.
      sub16 = _mm_sub_epi16(a16, b16);

      // Compute the squared difference between the 8, 16-bit values.
      mul16 = _mm_mullo_epi16(sub16, sub16);

      // Interleave the lower 4, 16-bit values with zeros to effectively
      // convert to 4, 32-bit values.
      mul32 = _mm_unpacklo_epi16(mul16, zero16);

      // Add the current 4 values to the running sum that stores 4, 32-bit
      // values.
      sum32 = _mm_add_epi32(sum32, mul32);

      // Interleave the upper 4, 16-bit values with zeros to effectively
      // convert to 4, 32-bit values.
      mul32 = _mm_unpackhi_epi16(mul16, zero16);

      // Add the current 4 values to the running sum that stores 4, 32-bit
      // values.
      sum32 = _mm_add_epi32(sum32, mul32);
    }

    // Copy the running sum of the squared differences to 4, 32-bit values
    // without assuming anything about the alignment of the pointer.
    DistanceType sum[4];
    _mm_storeu_si128((__m128i *)sum, sum32);

    // Manually sum the 4, 32-bit values, and return the result.
    return sum[0] + sum[1] + sum[2] + sum[3];

#else // ifndef SIFT_USE_SSE

    DistanceType distance = 0;
    for (int i = 0; i < NumDimensions; ++i)
    {
      DistanceType difference =
        static_cast<DistanceType>(descriptor1[i]) -
        static_cast<DistanceType>(descriptor2[i]);
      distance += difference * difference;
    }
    return distance;

#endif // SIFT_USE_SSE
#endif // SIFT_USE_L2

#ifdef SIFT_USE_DOT
#ifdef SIFT_USE_SSE

    // Set 16, 8-bit values to zero.
    __m128i zero8 = _mm_set1_epi8(0);

    // Set 8, 16-bit values to zero.
    __m128i zero16 = _mm_set1_epi16(0);

    // Set 4, 32-bit values to zero.
    __m128i sum32 = _mm_set1_epi32(0);

    __m128i a8;
    __m128i b8;
    __m128i a16;
    __m128i b16;
    __m128i mul16;
    __m128i mul32;

    // We will process 16 values at a time.
    const int dim_16 = NumDimensions / 16;
    for (int i = 0; i < dim_16; ++i)
    {
      // Load 16, 8-bit values from each descriptor without assuming anything
      // about the alignment of the underlying pointers.
      const int offset = 16 * i;
      a8 = _mm_loadu_si128((__m128i *)(descriptor1 + offset));
      b8 = _mm_loadu_si128((__m128i *)(descriptor2 + offset));

      /////////////////////////////////////////////////
      // Process the lower 8 values of the descriptors.

      // Interleave the lower 8, 8-bit values of the descriptors with zeros,
      // effectively converting the 8 values to 16-bit values.
      a16 = _mm_unpacklo_epi8(a8, zero8);
      b16 = _mm_unpacklo_epi8(b8, zero8);

      // Multiply the 8, 16-bit values together.
      mul16 = _mm_mullo_epi16(a16, b16);

      // Interleave the lower 4, 16-bit values with zeros to effectively
      // convert to 4, 32-bit values.
      mul32 = _mm_unpacklo_epi16(mul16, zero16);

      // Add the current 4 values to the running sum that stores 4, 32-bit
      // values.
      sum32 = _mm_add_epi32(sum32, mul32);

      // Interleave the upper 4, 16-bit values with zeros to effectively
      // convert to 4, 32-bit values.
      mul32 = _mm_unpackhi_epi16(mul16, zero16);

      // Add the current 4 values to the running sum that stores 4, 32-bit
      // values.
      sum32 = _mm_add_epi32(sum32, mul32);

      /////////////////////////////////////////////////
      // Process the upper 8 values of the descriptors.

      // Interleave the upper 8, 8-bit values of the descriptors with zeros,
      // effectively converting the 8 values to 16-bit values.
      a16 = _mm_unpackhi_epi8(a8, zero8);
      b16 = _mm_unpackhi_epi8(b8, zero8);

      // Multiply the 8, 16-bit values together.
      mul16 = _mm_mullo_epi16(a16, b16);

      // Interleave the lower 4, 16-bit values with zeros to effectively
      // convert to 4, 32-bit values.
      mul32 = _mm_unpacklo_epi16(mul16, zero16);

      // Add the current 4 values to the running sum that stores 4, 32-bit
      // values.
      sum32 = _mm_add_epi32(sum32, mul32);

      // Interleave the upper 4, 16-bit values with zeros to effectively
      // convert to 4, 32-bit values.
      mul32 = _mm_unpackhi_epi16(mul16, zero16);

      // Add the current 4 values to the running sum that stores 4, 32-bit
      // values.
      sum32 = _mm_add_epi32(sum32, mul32);
    }

    // Copy the running sum of the multiplied values to 4, 32-bit values without
    // assuming anything about the alignment of the pointer.
    DistanceType sum[4];
    _mm_storeu_si128((__m128i *)sum, sum32);

    // Manually sum the 4, 32-bit values, and return the result.
    return sum[0] + sum[1] + sum[2] + sum[3];

#else // ifndef SIFT_USE_SSE

    DistanceType distance = 0;
    for (int i = 0; i < NumDimensions; ++i)
    {
      distance +=
        static_cast<DistanceType>(descriptor1[i]) *
        static_cast<DistanceType>(descriptor2[i]);
    }
    return distance;

#endif // SIFT_USE_SSE
#endif // SIFT_USE_DOT
  }
};

} // namespace descriptor

} // namespace vocabulary_tree

#endif // VOCABULARY_TREE_DESCRIPTOR_TYPES_H

/*
 * Quarisma Memory library — DLL export/import (same pattern as Parallel).
 */
#pragma once

#define MEMORY_VISIBILITY_ENUM

#if defined(MEMORY_STATIC_DEFINE)
#define MEMORY_API
#define MEMORY_VISIBILITY
#define MEMORY_IMPORT
#define MEMORY_HIDDEN

#elif defined(MEMORY_SHARED_DEFINE)
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef MEMORY_BUILDING_DLL
#define MEMORY_API __declspec(dllexport)
#else
#define MEMORY_API __declspec(dllimport)
#endif
#define MEMORY_VISIBILITY
#define MEMORY_IMPORT __declspec(dllimport)
#define MEMORY_HIDDEN
#elif defined(__GNUC__) && __GNUC__ >= 4
#define MEMORY_API __attribute__((visibility("default")))
#define MEMORY_VISIBILITY __attribute__((visibility("default")))
#define MEMORY_IMPORT __attribute__((visibility("default")))
#define MEMORY_HIDDEN __attribute__((visibility("hidden")))
#else
#define MEMORY_API
#define MEMORY_VISIBILITY
#define MEMORY_IMPORT
#define MEMORY_HIDDEN
#endif

#else
#define MEMORY_API
#define MEMORY_VISIBILITY
#define MEMORY_IMPORT
#define MEMORY_HIDDEN
#endif


---
name: new-test
description: Scaffold a new unit test file for this repo with the correct macro (MEMORYTEST or TEST/TEST_F), placed under Testing/Cxx/. Use when the user asks to add/write a test for a class or function in include/ or src/.
---

# new-test

Scaffolds a test file matching this repo's existing test conventions (see
root `/CLAUDE.md` "Test conventions"). Test files are discovered
automatically by both build systems — CMake's recursive glob and Bazel's
package-local glob under `Testing/Cxx/` — **no build-file edit is needed**
for a new file, only for a new subdirectory (register it in both CMake and
Bazel, per root `/CLAUDE.md`).

## Steps

1. **Find the right location.** For a source file at
   `include/<path>/<name>.h`, the test goes in
   `Testing/Cxx/Test<ClassName>.cpp` (CamelCase class name, even though the
   rest of the codebase is `snake_case` — this is test-file-only). Check for
   an existing test file for the same class first (`grep -rl` the class name
   under `Testing/Cxx/`) and add cases there instead of creating a duplicate
   file.

2. **Pick the test macro.** This repo's majority convention is `MEMORYTEST`
   (it expands to Google Test's `TEST` under the normal
   `MEMORY_HAS_GTEST=1` build — see `Testing/MemoryTest.h.in`), but plain
   `TEST`/`TEST_F` also appears in a minority of files. Open an existing
   `Test*.cpp` in `Testing/Cxx/` and copy its macro choice, namespace usage
   (`using namespace memory;`), and license header verbatim rather than
   composing from memory or introducing a third style.

3. **Write cases.** One behavior per case. At minimum include:
   - happy path
   - a boundary/edge case
   - an error/failure path (invalid input, null, empty collection) that
     checks the return value — this codebase does not throw for expected
     failures (see root `/CLAUDE.md` / `.augment/rules/coding.md`
     error-handling policy), so assert on the returned `bool`/`optional`/
     result, not on an exception.

   ```cpp
   using namespace memory;

   MEMORYTEST(my_class_test, handles_valid_input)
   {
       my_class obj;
       EXPECT_TRUE(obj.do_something());
   }

   MEMORYTEST(my_class_test, rejects_invalid_input)
   {
       my_class obj;
       EXPECT_FALSE(obj.do_something_with(-1));
   }
   ```

   ```cpp
   // Where the neighboring file already uses plain Google Test instead:
   using namespace memory;

   TEST(MyClass, handles_valid_input)
   {
       my_class obj;
       EXPECT_TRUE(obj.do_something());
   }

   // Fixture-based, when shared setup is needed:
   class MyClassTest : public ::testing::Test
   {
   protected:
       void SetUp() override { /* ... */ }
   };

   TEST_F(MyClassTest, handles_valid_input)
   {
       EXPECT_TRUE(obj_.do_something());
   }
   ```

4. **GPU tests**: name the file/cases `TestGpu*`/`TestCuda*`/`TestHip*` to
   match the exclusion patterns already in `Testing/Cxx/CMakeLists.txt` and
   `Testing/Cxx/BUILD.bazel`, so they're automatically skipped on backends
   that don't support them — check both files before assuming a new naming
   pattern will be excluded correctly on both build systems.

5. **Build and run** (see the `project-build` skill for more invocation
   patterns):
   ```
   cd Scripts
   python3 setup.py config.build.test
   python3 setup_bazel.py config.build.test
   ```

Don't introduce a third test-macro style — match what's already there.

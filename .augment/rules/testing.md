---
type: "always_apply"
---

### Test Scope
- Test happy paths and success cases
- Explicitly test boundary conditions and edge cases
- Test error handling and failure scenarios (return value validation)
- Test null pointers, empty collections, and invalid inputs
- Verify state changes and side effects
- Use the `MEMORYTEST` macro for new tests (expands to Google Test's `TEST` under the normal `MEMORY_HAS_GTEST=1` build — see `Testing/MemoryTest.h.in`); plain `TEST`/`TEST_F` also exist in this codebase — match whichever the neighboring file in `Testing/Cxx/` already uses

✅ **Correct Example**
```cpp
MEMORYTEST(my_class_test, handles_valid_input) {
  my_class obj;
  EXPECT_TRUE(obj.do_something());
}

MEMORYTEST(my_class_test, handles_invalid_input) {
  my_class obj;
  EXPECT_FALSE(obj.do_something_with(-1));
}

MEMORYTEST(my_class_test, handles_null_pointer) {
  my_class obj;
  EXPECT_FALSE(obj.process(nullptr));
}
```

---

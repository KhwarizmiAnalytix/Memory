---
type: "always_apply"
---

# Documentation File Creation Rules

## Rule 1: Do Not Create Markdown Files Unless Explicitly Requested
- **Never** create summary, verification, or documentation markdown files (*.md) unless the user explicitly asks for them
- This includes files like:
  - Summary files (e.g., SUMMARY.md, FIXES_COMPLETE.md)
  - Verification checklists (e.g., VERIFICATION_CHECKLIST.md)
  - Before/after comparison files (e.g., BEFORE_AFTER_COMPARISON.md)
  - Index or reference files (e.g., INDEX.md)
  - Any other documentation files not directly requested by the user
- Focus on making the requested changes to existing files only
- Provide summaries and explanations directly in your response to the user instead of creating files

## Rule 2: Default Location for Requested Documentation Files
- **If** the user explicitly requests a markdown documentation file **and** does not specify a location, then:
  - Place the file in the `Docs/` folder
  - Use clear, descriptive filenames
  - Follow existing naming conventions in the `Docs/` folder
- **If** the user specifies a location, use that location exactly as specified
- **If** the file is a core project file (like README.md, CONTRIBUTING.md, SECURITY.md, LICENSE), place it in the repository root unless otherwise specified

## Rule 3: Colon Alignment in CMake Configuration Messages
- `message("  LABEL : value")` configuration summary lines in `CMakeLists.txt` and `Cmake/lto.cmake` must align the colon at exactly 24 characters from the opening quote (inclusive) — see root `/CLAUDE.md` "CMake Configuration Message Alignment" for the exact format and example.

## Examples
- ❌ User asks to "fix documentation issues" → Do NOT create FIXES_COMPLETE.md
- ✓ User asks to "fix documentation issues" → Make the fixes and explain in your response
- ✓ User asks to "create a summary document of the changes" → Create Docs/CHANGES_SUMMARY.md
- ✓ User asks to "create a guide for contributors in the root" → Create CONTRIBUTING.md in the repo root

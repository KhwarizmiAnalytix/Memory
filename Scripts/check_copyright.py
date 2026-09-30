"""Check for missing copyright headers in C++ files."""

import argparse
import sys
from pathlib import Path


def check_copyright(file_path):
    """Check if file has copyright or SPDX header."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(1000)  # Check first 1000 chars
            return "Copyright" in content or "SPDX-License" in content
    except Exception:
        return True  # Assume has copyright if can't read


def main():
    parser = argparse.ArgumentParser(description="Check for missing copyright headers")
    parser.add_argument("files", nargs="*", help="Files to check")
    args = parser.parse_args()

    issues = []
    for filepath in args.files:
        path = Path(filepath)
        if not check_copyright(path):
            issues.append(
                {
                    "file": filepath,
                    "line": 1,
                    "column": 1,
                    "message": "Missing copyright/SPDX header",
                    "code": "COPYRIGHT",
                }
            )

    if issues:
        for issue in issues:
            msg = (
                f"{issue['file']}:{issue['line']}:{issue['column']}: "
                f"{issue['code']}: {issue['message']}"
            )
            print(msg)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

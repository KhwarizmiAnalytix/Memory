---
type: "always_apply"
---

# 🐍 Python Coding Rules
**Google Python Style + Cross-Platform Compliance**

Applies to this repository's own Python (`Scripts/setup.py`, `Scripts/setup_bazel.py`, and other tooling under `Scripts/`) — not to any vendored code under `ThirdParty/` (see the `ThirdParty.md` rule).

---

## 1. 📘 General Guidelines
- Follow the **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)**.
- Target **Python 3.9+** for full typing and compatibility.
- Code must run on **Windows**, **macOS**, and **Linux** (this repo is actively built on Windows — see `README.md`).
- Write **modular, readable, and testable** code.

---

## 2. 📁 File and Module Organization
- One module per file (`.py`).
- Filenames use lowercase with underscores: `file_handler.py`.
- Use **`pathlib`** for file paths — avoids OS-specific issues.

```python
from pathlib import Path

data_dir = Path.home() / "Documents" / "project_data"
```

✅ Works on all systems.

---

## 3. 🧱 Imports
Order imports as follows:
1. Standard library
2. Third-party libraries
3. Local modules

Each group separated by a blank line.

```python
import os
import sys

import requests

from myapp import config
```

---

## 4. 📄 Docstrings
Use triple double quotes (`\"\"\"`) for all public functions, classes, and modules.
Follow **Google-style docstrings**.

```python
def get_user_info(user_id: int) -> dict:
    \"\"\"Fetches user information by ID.

    Args:
        user_id: Unique identifier of the user.

    Returns:
        Dictionary containing user data.

    Raises:
        ValueError: If the ID is invalid.
    \"\"\"
    ...
```

---

## 5. 🧩 Functions and Classes
- Functions and variables: `snake_case`
- Classes: `CapWords`
- Constants: `UPPER_CASE`
- Keep functions short and focused on one purpose.

```python
class FileHandler:
    \"\"\"Handles file operations.\"\"\"

    def read_text(self, filepath: str) -> str:
        \"\"\"Reads text from a file.\"\"\"
        with open(filepath, encoding="utf-8") as file:
            return file.read()
```

---

## 6. 🌐 Cross-Platform Practices
✅ **DO:**
- Use `pathlib`, `os`, `subprocess`, `tempfile`
- Use environment variables (`os.environ`)
- Handle newline differences with `newline=""` in file I/O

🚫 **DON'T:**
- Use hard-coded file paths (`C:\\`, `/home/`)
- Use OS-specific commands (`cls`, `clear`)

```python
import subprocess

subprocess.run(["echo", "Hello World!"], check=True)
```

---

## 7. ⚙️ Error Handling
Catch **specific exceptions** and include meaningful error messages.

```python
try:
    process_data()
except FileNotFoundError as e:
    print(f"File missing: {e}")
```

---

## 8. 🧪 Testing
- Use `unittest` or `pytest`
- Test filenames start with `test_`
- Keep tests independent of the environment

```python
import unittest

class TestMath(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(2 + 2, 4)
```

Run tests:
```bash
pytest tests/
```

---

## 9. 🧼 Code Quality
Use automatic tools to enforce quality and formatting — this repo runs these via `lintrunner` (`.lintrunner.toml`: `RUFF`, `PYFMT`, `MYPY`/`MYPYSTRICT`, `FLAKE8`, `PYREFLY`, `IMPORT_LINTER`, among others); invoke them directly when iterating on one file:

### Format with `black` / `ruff format` (see `pyproject.toml` for this repo's line length)
```bash
ruff format Scripts/
```

### Lint with `ruff` / `mypy`
```bash
ruff check Scripts/
mypy --config-file mypy.ini Scripts/
```

---

## 10. 🔒 Security and Portability
- No hard-coded credentials or absolute paths.
- Use environment variables or `.env` files.
- Use only cross-platform standard library modules unless absolutely needed.

---

## ✅ Summary Checklist

| Rule | Description | Example |
|------|--------------|----------|
| Style | Google Python Style | `snake_case`, `\"\"\"Docstrings\"\"\"` |
| Imports | Group by category | stdlib → 3rd → local |
| Paths | Use `pathlib` | ✅ |
| Exceptions | Catch specific | `except ValueError:` |
| Tests | Use `pytest` or `unittest` | `test_*.py` |
| Lint | `ruff`, `mypy`, `lintrunner` | Static checks |
| Format | `ruff format` | Enforces style |
| Security | Use env vars | No hard-coded secrets |

This repo's actual tool configuration lives in `pyproject.toml`, `mypy.ini` / `mypy-strict.ini`, `pyrefly.toml`, and `.lintrunner.toml` — check those files rather than assuming defaults when a specific rule (line length, ignored checks) matters.

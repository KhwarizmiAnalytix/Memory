#!/usr/bin/env python3
"""
IWYU Analysis Runner for XSigma Project
Runs IWYU analysis on source files and captures output to log file.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def find_source_files(source_dir: str) -> list[str]:
    """Find all C++ source files in the given directory."""
    source_files: list[Path] = []
    source_path = Path(source_dir)

    for ext in ["*.cpp", "*.cc", "*.c", "*.hpp", "*.h"]:
        source_files.extend(source_path.rglob(ext))

    return [str(f) for f in source_files]


def get_compile_flags_for_file(compile_commands_file: str, source_file: str) -> list[str]:
    """Extract compile flags for a specific source file from compile_commands.json."""
    if not os.path.exists(compile_commands_file):
        return []

    try:
        with open(compile_commands_file) as f:
            compile_commands = json.load(f)

        # Normalize paths for comparison (resolve to absolute paths)
        source_file_abs = os.path.abspath(os.path.normpath(source_file))

        for entry in compile_commands:
            entry_file_abs = os.path.abspath(os.path.normpath(entry.get("file", "")))
            if entry_file_abs == source_file_abs:
                # Extract include paths and defines from the command
                command = entry.get("command", "")
                flags = []

                # Split command and extract relevant flags
                parts = command.split()
                i = 0
                while i < len(parts):
                    part = parts[i]
                    if part.startswith("-I"):
                        if len(part) > 2:
                            flags.append(part)
                        elif i + 1 < len(parts):
                            flags.append(f"-I{parts[i + 1]}")
                            i += 1
                    elif part.startswith("-D"):
                        flags.append(part)
                    elif part.startswith("-std="):
                        flags.append(part)
                    i += 1

                return flags
    except Exception as e:
        print(f"Warning: Could not parse compile commands: {e}")

    return []


def run_iwyu_on_file(
    iwyu_executable: str,
    iwyu_args: list[str],
    source_file: str,
    log_file: str,
    compile_commands_file: str | None = None,
) -> bool:
    """Run IWYU on a single source file and log the output."""
    print(f"Analyzing: {source_file}")

    # Get compile flags for this file
    compile_flags = []
    if compile_commands_file:
        compile_flags = get_compile_flags_for_file(compile_commands_file, source_file)

    # Build IWYU command
    cmd = (
        [iwyu_executable]
        + iwyu_args
        + compile_flags
        + [
            "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
            "-D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING",
            source_file,
        ]
    )

    # Log analysis start
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n================================================================================\n")
        f.write(f"ANALYZING: {source_file}\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("================================================================================\n")

    try:
        # Run IWYU and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout per file
        )

        # Log the output
        with open(log_file, "a", encoding="utf-8") as f:
            if result.stdout:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n")

            if result.stderr:
                f.write("STDERR:\n")
                f.write(result.stderr)
                f.write("\n")

            f.write(f"Exit code: {result.returncode}\n")
            f.write("ANALYSIS COMPLETED\n")
            f.write("================================================================================\n\n")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("ERROR: IWYU analysis timed out after 60 seconds\n")
            f.write("================================================================================\n\n")
        return False

    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"ERROR: {str(e)}\n")
            f.write("================================================================================\n\n")
        return False


def main() -> int:
    """Run IWYU analysis on source files and log results."""
    parser = argparse.ArgumentParser(description="Run IWYU analysis on XSigma source files")
    parser.add_argument("--source-dir", required=True, help="Source directory to analyze")
    parser.add_argument("--log-file", required=True, help="Output log file")
    parser.add_argument("--iwyu-executable", required=True, help="Path to IWYU executable")
    parser.add_argument("--mapping-file", help="IWYU mapping file")
    parser.add_argument("--compile-commands", help="Path to compile_commands.json file")
    parser.add_argument("--max-files", type=int, default=10, help="Maximum number of files to analyze")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.source_dir):
        print(f"Error: Source directory does not exist: {args.source_dir}")
        return 1

    if not os.path.exists(args.iwyu_executable):
        print(f"Error: IWYU executable does not exist: {args.iwyu_executable}")
        return 1

    # Build IWYU arguments
    iwyu_args = [
        "-Xiwyu",
        "--cxx17ns",
        "-Xiwyu",
        "--max_line_length=120",
        "-Xiwyu",
        "--verbose=1",
        "-Xiwyu",
        "--comment_style=short",
        "-Xiwyu",
        "--error=0",
        "-Xiwyu",
        "--no_fwd_decls",
        "-Xiwyu",
        "--quoted_includes_first",
        "-Xiwyu",
        "--transitive_includes_only",
    ]

    if args.mapping_file and os.path.exists(args.mapping_file):
        iwyu_args.extend(["-Xiwyu", f"--mapping_file={args.mapping_file}"])

    # Find source files
    source_files = find_source_files(args.source_dir)
    print(f"Found {len(source_files)} source files")

    # Limit the number of files to analyze
    if len(source_files) > args.max_files:
        print(f"Limiting analysis to first {args.max_files} files")
        source_files = source_files[: args.max_files]

    # Initialize log file
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write("================================================================================\n")
        f.write("IWYU ANALYSIS SESSION STARTED\n")
        f.write("================================================================================\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source directory: {args.source_dir}\n")
        f.write(f"IWYU executable: {args.iwyu_executable}\n")
        f.write(f"Mapping file: {args.mapping_file}\n")
        f.write(f"Files to analyze: {len(source_files)}\n")
        f.write(f"IWYU arguments: {' '.join(iwyu_args)}\n")
        f.write("================================================================================\n")

    # Analyze files
    successful = 0
    failed = 0

    for source_file in source_files:
        if run_iwyu_on_file(
            args.iwyu_executable,
            iwyu_args,
            source_file,
            args.log_file,
            args.compile_commands,
        ):
            successful += 1
        else:
            failed += 1

    # Log summary
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write("================================================================================\n")
        f.write("IWYU ANALYSIS SESSION COMPLETED\n")
        f.write("================================================================================\n")
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Files analyzed: {len(source_files)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n")
        f.write("================================================================================\n\n")

    print(f"IWYU analysis completed. Results logged to: {args.log_file}")
    print(f"Files analyzed: {len(source_files)}, Successful: {successful}, Failed: {failed}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

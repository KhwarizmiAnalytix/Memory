"""Documented setup actions and validated options shared by CI and local builds."""

import argparse
import contextlib
import io
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .options import OPTIONS

ACTIONS = {
    "config",
    "build",
    "test",
    "install",
    "clean",
    "cleanbuild",
    "verifycache",
    "verifytoolchain",
}
GENERATORS = (
    "default",
    "Ninja",
    "Ninja Multi-Config",
    "Xcode",
    "Visual Studio 16 2019",
    "Visual Studio 17 2022",
    "Visual Studio 18 2026",
)
COMPILERS = ("default", "msvc", "clang", "clang-cl", "gcc")
# Dotted Visual Studio tokens -> CMake generator (shared with setup.py's legacy path).
VS_GENERATORS = {
    "vs17": "Visual Studio 15 2017 Win64",
    "vs19": "Visual Studio 16 2019",
    "vs22": "Visual Studio 17 2022",
    "vs26": "Visual Studio 18 2026",
}


def dotted_compiler(token: str) -> tuple[str, str, str] | None:
    """Translate a dotted compiler token into (family, C compiler, C++ compiler).

    Accepts versioned names and paths (``gcc-13``, ``clang-18``,
    ``/opt/bin/g++``) as well as the exact family names. This is the single
    translation used by both setup.py's legacy pipeline and named-option
    dispatch, so the two modes cannot disagree about a dotted selection.
    """
    if token == "msvc":
        return "msvc", "cl", "cl"
    if token == "clang-cl":
        return "clang-cl", "clang-cl", "clang-cl"
    if "clang" in token and token != "clangtidy":
        return "clang", token, token.replace("clang", "clang++")
    if ("gcc" in token or "g++" in token) and token != "cppcheck":
        if "g++" in token:
            return "gcc", token.replace("g++", "gcc"), token
        return "gcc", token, token.replace("gcc", "g++")
    return None


def dotted_generator(token: str) -> str | None:
    """Translate a dotted generator token (``ninja``, ``xcode``, ``vs22``...)."""
    if token == "ninja":
        return "Ninja"
    if token == "xcode":
        return "Xcode"
    return VS_GENERATORS.get(token)


def canonical_build_type(value: str) -> str:
    """Return CMake's canonical configuration spelling."""
    return {
        "debug": "Debug",
        "release": "Release",
        "relwithdebinfo": "RelWithDebInfo",
        "minsizerel": "MinSizeRel",
    }.get(value.lower(), value)


def boolean(value: str) -> str:
    """Normalize documented Boolean values without accepting arbitrary definitions."""
    if value.lower() in ("on", "true", "1", "yes"):
        return "ON"
    if value.lower() in ("off", "false", "0", "no"):
        return "OFF"
    raise argparse.ArgumentTypeError("expected on or off")


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def argument_parser(bazel: bool = False) -> argparse.ArgumentParser:
    """Build the public option registry used by parsing, help, and CI validation."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--compiler",
        choices=COMPILERS,
        help="Compiler selection; default respects the build system/environment",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Build/test target; repeat for multiple targets",
    )
    if bazel:
        parser.add_argument(
            "--compilation-mode",
            choices=("fastbuild", "dbg", "opt"),
            help="Bazel compilation mode",
        )
        parser.add_argument(
            "--test-output",
            choices=("errors", "all", "summary", "streamed"),
            default=None,
            help="Test output policy (default: errors)",
        )
        parser.add_argument(
            "--batch", action="store_true", help="Run Bazel without a persistent server"
        )
        parser.add_argument(
            "--output-root", help="Directory for Bazel's output/cache state"
        )
        parser.add_argument(
            "--cxx-standard", choices=("17", "20", "23"), help="C++ language standard"
        )
        for name in ("tbb", "mimalloc", "numa", "memkind", "profiler", "shared"):
            parser.add_argument(
                "--" + name, type=boolean, help="Enable/disable " + name + " (on/off)"
            )
        project = Path(__file__).resolve().parents[2].name
        if project in ("Vectorization",):
            parser.add_argument(
                "--cpu-backend",
                choices=("no", "sse", "avx", "avx2", "avx512", "neon", "sve"),
                help="CPU SIMD backend",
            )
        if project in ("Vectorization", "Memory"):
            parser.add_argument(
                "--gpu-backend",
                choices=("none", "cuda", "hip", "metal"),
                help="GPU backend",
            )
        if project == "Logging":
            parser.add_argument(
                "--backend",
                choices=("native", "loguru", "glog", "spdlog"),
                type=str.lower,
                help="Logging backend",
            )
        if project == "Parallel":
            parser.add_argument(
                "--backend",
                choices=("sequential", "stdthread", "openmp", "tbb"),
                help="Parallel backend",
            )
        return parser
    parser.add_argument(
        "--source-dir", help="Project or installed-package consumer source directory"
    )
    parser.add_argument(
        "--build-dir", help="Build directory; relative paths use the invoking directory"
    )
    parser.add_argument(
        "--build-type",
        type=canonical_build_type,
        choices=("Debug", "Release", "RelWithDebInfo", "MinSizeRel"),
        help="Build and test configuration",
    )
    parser.add_argument(
        "--generator",
        choices=GENERATORS,
        help="Build generator; default respects CMAKE_GENERATOR",
    )
    parser.add_argument("--c-compiler", help="C compiler executable")
    parser.add_argument("--cxx-compiler", help="C++ compiler executable")
    parser.add_argument("--architecture", help="Visual Studio target architecture")
    parser.add_argument("--preload-file", help="Initial project configuration file")
    parser.add_argument(
        "--parallel", type=positive, default=4, help="Number of build jobs (default: 4)"
    )
    parser.add_argument(
        "--install-prefix", help="Installation prefix for config/install actions"
    )
    parser.add_argument(
        "--test-filter", help="Only run tests matching this regular expression"
    )
    parser.add_argument("--test-label", help="Only run tests with a matching label")
    parser.add_argument(
        "--test-timeout",
        type=positive,
        default=300,
        help="Per-test timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--memcheck", action="store_true", help="Run the CTest memory-check dashboard"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show verbose build/test output"
    )
    for option in OPTIONS:
        kind = option["kind"]
        converter = (
            boolean
            if kind in ("bool", "cuda-compat")
            else (
                str.upper
                if kind == "upper-enum"
                else str.lower
                if kind == "enum"
                else str
            )
        )
        aliases = ["--" + option["name"]]
        if "-" in option["name"]:
            aliases.append("--" + option["name"].replace("-", "_"))
        parser.add_argument(
            *aliases,
            dest=option["name"].replace("-", "_"),
            type=converter,
            choices=option.get("choices"),
            help=(
                f"Set {option['name']} "
                f"({'on/off' if kind in ('bool', 'cuda-compat') else 'setting'})"
            ),
        )
    return parser


def print_help() -> None:
    """Print the supported public actions and named options."""
    print("\nSupported setup actions (combine with dots):")
    print("  config.build.test, install, cleanbuild, verifycache, verifytoolchain")
    print(
        "  python Scripts/setup.py config.build.test.release.ninja.clang"
        " --build-dir build"
    )
    print(
        "  python Scripts/setup.py install.release"
        " --build-dir build --install-prefix install"
    )
    print("\nOfficial named options:")
    print(argument_parser().format_help())


def print_bazel_help() -> None:
    print("\nSupported Bazel actions: config.build.test, clean, coverage")
    print(
        "  python Scripts/setup_bazel.py build.test"
        " --target //Testing/... --compiler default"
    )
    print(argument_parser(bazel=True).format_help())
    print(
        "Dotted features: cuda, hip, metal, tbb, nomimalloc, numa,"
        " memkind, profiler, shared, static"
    )
    print(
        "Dotted tools: coverage, valgrind, benchmark, spell,"
        " cppcheck, clangtidy, iwyu, vv"
    )
    print(
        "Named options override dotted choices."
        " Compiler/environment and fastbuild are the defaults."
    )
    print(
        "SDK prefixes: CUDA_ROOT/CUDA_PATH, ROCM_ROOT/ROCM_PATH,"
        " TBB_ROOT/TBB_DIR, PROFILER_ROOT"
    )


def _reject_raw(arguments: list[str], parser: argparse.ArgumentParser) -> None:
    for argument in arguments:
        if argument in ("--", "cmake", "ctest", "bazel") or argument.startswith(
            ("-D", "--cmake", "--copt", "--define", "--repo_env", "--test_output")
        ):
            parser.error(
                f"unsupported argument {argument!r}; use documented actions and options"
            )


def parse_options(
    arguments: list[str], bazel: bool = False
) -> tuple[argparse.Namespace, list[str]]:
    """Parse only declared options, retaining documented dotted legacy flags."""
    parser = argument_parser(bazel)
    _reject_raw(arguments, parser)
    options, remaining = parser.parse_known_args(arguments)
    legacy_prefixes = (
        "--backend.",
        "--gpu.",
        "--gpu_backend.",
        "--cpu_backend.",
        "--sanitizer.",
        "--lto.",
        "--linker.",
        "--packet-size=",
    )
    for argument in remaining:
        if argument.startswith("-") and not argument.startswith(legacy_prefixes):
            parser.error(f"unrecognized option {argument!r}; see --help")
    return options, remaining


def validate_legacy(
    tokens: list[str],
    flags_factory: Callable[[list[str]], Any],
    parser: argparse.ArgumentParser,
) -> None:
    """Reject misspelled actions rather than silently ignoring them."""
    flags = flags_factory([])
    allowed = set(
        next(value for name, value in vars(flags).items() if name.endswith("__key"))
    )
    allowed.update(
        ACTIONS
        | {
            "release",
            "debug",
            "relwithdebinfo",
            "minsizerel",
            "ninja",
            "xcode",
            "vs17",
            "vs19",
            "vs22",
            "vs26",
            "clang",
            "clang-cl",
            "gcc",
            "msvc",
            "shared",
            "address",
            "undefined",
            "thread",
            "memory",
            "leak",
            "ccache",
            "sccache",
            "buildcache",
            "none",
            "no",
            "sse",
            "avx",
            "avx2",
            "avx512",
            "neon",
            "sve",
            "cuda",
            "hip",
            "metal",
            "native",
            "loguru",
            "glog",
            "spdlog",
            "sequential",
            "stdthread",
            "openmp",
            "tbb",
            "v",
            "vv",
            "torch",
            "noinstall",
            "tao",
            "logfatal",
            "throw",
            "nocxademangle",
            "nomimalloc",
            "batch",
            "clean",
        }
    )
    for token in tokens:
        if (
            token in allowed
            or token.startswith(
                (
                    "lto.",
                    "linker.",
                    "backend.",
                    "gpu.",
                    "gpu_backend.",
                    "cpu_backend.",
                    "parallel_backend.",
                    "packet_size.",
                )
            )
            or re.fullmatch(r"(?:cxx|c\+\+)\d+|psize\d+|sve(?:128|256|512)", token)
        ):
            continue
        if re.search(r"(?:clang|gcc|g\+\+)", token) and token not in ("cmake", "ctest"):
            continue
        parser.error(f"unrecognized action or flag {token!r}; see --help")


def _run(command: list[str], **kwargs: Any) -> "subprocess.CompletedProcess[Any]":
    return subprocess.run(command, check=True, **kwargs)


def _definitions(options: Any) -> list[str]:
    definitions = []
    for option in OPTIONS:  # option: Option
        value = getattr(options, option["name"].replace("-", "_"))
        if value is None:
            continue
        if option["kind"] == "cuda-compat":
            value = "-allow-unsupported-compiler" if value == "ON" else ""
        if option["kind"] == "linker-flags":
            value = "" if value == "default" else "-fuse-ld=" + value
        definitions.append(f"-D{option['cache']}={value}")
    if options.install_prefix:
        definitions.append(f"-DCMAKE_INSTALL_PREFIX={options.install_prefix}")
    return definitions


def _explicit_legacy_definitions(
    tokens: list[str], flags_factory: Callable[[list[str]], Any]
) -> list[str]:
    """Translate explicit dotted feature choices without overriding project defaults."""
    with contextlib.redirect_stdout(io.StringIO()):
        baseline = flags_factory(["config"])
        selected = flags_factory(tokens or ["config"])
        baseline_flags: list[str] = []
        selected_flags: list[str] = []
        baseline.create_cmake_flags(baseline_flags, "Release", os.name)
        selected.create_cmake_flags(selected_flags, "Release", os.name)
    defaults = dict(flag.split("=", 1) for flag in baseline_flags)
    return [
        flag
        for flag in selected_flags
        if defaults.get(flag.split("=", 1)[0]) != flag.split("=", 1)[1]
    ]


def _configuration(build: Path, requested: str | None) -> str:
    if requested:
        return requested
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        match = re.search(
            r"^CMAKE_BUILD_TYPE:\w+=(.+)$",
            cache.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if match:
            return canonical_build_type(match.group(1))
    return "Release"


def verify_cache(build: Path, target: str | None, parallel: int) -> None:
    """Require a real cache launcher and hits from a clean rebuild."""
    if not target:
        raise ValueError("verifycache requires --target")
    commands = _run(
        ["cmake", "--build", str(build), "--target", target, "--", "-t", "commands"],
        capture_output=True,
        text=True,
    ).stdout
    (build / "compile-commands.txt").write_text(commands, encoding="utf-8")
    if not re.search(
        r'buildcache(?:\.exe)?"? .* (?:-c |/c |/Fo)',
        commands,
        re.IGNORECASE,  # codespell:ignore fo
    ):
        raise RuntimeError("BuildCache is missing from the generated compiler commands")
    _run(["cmake", "--build", str(build), "--parallel", str(parallel)])
    _run(["cmake", "--build", str(build), "--clean-first", "--parallel", str(parallel)])
    statistics = _run(
        ["buildcache", "--show-stats"], capture_output=True, text=True
    ).stdout
    print(statistics, end="")
    (build / "cache-stats.txt").write_text(statistics, encoding="utf-8")
    if not re.search(r"(?:Direct|Local|Remote) hits: +[1-9]", statistics):
        raise RuntimeError("The clean rebuild did not produce any BuildCache hits")


def verify_toolchain(build: Path, expected: str, generator: str = "Ninja") -> None:
    """Verify the realized generator and compiler rather than trusting a label."""
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    if not re.search(
        rf"^CMAKE_GENERATOR:INTERNAL={re.escape(generator)}$", cache, re.MULTILINE
    ):
        raise RuntimeError(f"Expected the {generator} generator")
    files = list((build / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake"))
    if len(files) != 1:
        raise RuntimeError("Expected one CMake C++ compiler description")
    compiler = files[0].read_text(encoding="utf-8")
    expected_ids = {
        "msvc": ("MSVC",),
        "clang": ("Clang", "AppleClang"),
        "clang-cl": ("Clang",),
        "gcc": ("GNU",),
    }.get(expected, ())
    if not any(
        f'set(CMAKE_CXX_COMPILER_ID "{expected_id}")' in compiler
        for expected_id in expected_ids
    ):
        raise RuntimeError(f"CMake did not select the requested compiler {expected}")
    frontend = {"clang": "GNU", "clang-cl": "MSVC"}.get(expected)
    if (
        frontend
        and f'set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT "{frontend}")' not in compiler
    ):
        raise RuntimeError(f"Expected the {frontend} frontend")
    print(f"Verified {expected} with {generator}")


def dispatch(
    arguments: list[str],
    root: "str | Path",
    flags_factory: Callable[[list[str]], Any],
    configuration_factory: Callable[[list[str]], Any],
    parse_legacy: Callable[[list[str]], list[str]],
) -> int | None:
    """Execute documented named options, leaving existing dotted usage compatible."""
    if arguments == ["--help"]:
        return None
    options, remaining = parse_options(arguments)
    tokens = parse_legacy(remaining)
    parser = argument_parser()
    validate_legacy(tokens, flags_factory, parser)
    modern = (
        "msvc" in tokens
        or len(remaining) != len(arguments)
        or bool(
            set(tokens)
            & {"install", "cleanbuild", "clean", "verifycache", "verifytoolchain"}
        )
    )
    if not modern:
        return None
    phases = set(tokens) & ACTIONS
    if not phases:
        parser.error("specify an action such as config.build.test")
    root = Path(root).resolve()
    source = Path(options.source_dir).resolve() if options.source_dir else root
    build = (
        Path(options.build_dir).resolve() if options.build_dir else root / "build_ninja"
    )
    requested_type = options.build_type or next(
        (
            canonical_build_type(token)
            for token in tokens
            if token in ("debug", "release", "relwithdebinfo", "minsizerel")
        ),
        None,
    )
    # Last dotted choice wins, matching the legacy pipeline's in-order processing.
    dotted_compilers = [c for c in map(dotted_compiler, tokens) if c is not None]
    dotted_generators = [g for g in map(dotted_generator, tokens) if g is not None]
    compiler = options.compiler or (
        dotted_compilers[-1][0] if dotted_compilers else "clang"
    )
    generator = options.generator or (
        dotted_generators[-1] if dotted_generators else "Ninja"
    )
    definitions = _explicit_legacy_definitions(tokens, flags_factory) + _definitions(
        options
    )
    try:
        if "config" in phases:
            command = ["cmake", "-S", str(source), "-B", str(build)]
            if generator != "default":
                command.extend(["-G", generator])
            if options.architecture:
                command.extend(["-A", options.architecture])
            if options.preload_file:
                command.extend(["-C", str(Path(options.preload_file).resolve())])
            if requested_type or not options.preload_file:
                command.append("-DCMAKE_BUILD_TYPE=" + (requested_type or "Release"))
            if options.compiler:
                cc, cxx = (dotted_compiler(compiler) or (None, None, None))[1:]
            elif dotted_compilers:
                cc, cxx = dotted_compilers[-1][1:]
            elif not options.generator and generator in VS_GENERATORS.values():
                # Dotted vsNN with no compiler: let Visual Studio pick (legacy behavior).
                cc, cxx = None, None
            else:
                cc, cxx = "clang", "clang++"
            cc, cxx = options.c_compiler or cc, options.cxx_compiler or cxx
            if generator.startswith("Visual Studio") and compiler == "clang-cl":
                command.extend(["-T", "ClangCL"])
                cc, cxx = None, None
            if cc:
                command.append("-DCMAKE_C_COMPILER=" + cc)
            if cxx:
                command.append("-DCMAKE_CXX_COMPILER=" + cxx)
            _run(command + definitions)
        build_type = _configuration(build, requested_type)
        if "verifytoolchain" in phases:
            verify_toolchain(build, compiler, generator)
        if "clean" in phases:
            _run(
                [
                    "cmake",
                    "--build",
                    str(build),
                    "--target",
                    "clean",
                    "--config",
                    build_type,
                ]
            )
        if "build" in phases or "cleanbuild" in phases:
            command = [
                "cmake",
                "--build",
                str(build),
                "--config",
                build_type,
                "--parallel",
                str(options.parallel),
            ]
            if "cleanbuild" in phases:
                command.append("--clean-first")
            if options.target:
                command.extend(["--target", *options.target])
            if options.verbose:
                command.append("--verbose")
            _run(command)
        analysis = set(tokens) & {"cppcheck", "coverage", "spell", "benchmark"}
        configuration: Any | None = None
        if analysis:
            configuration = configuration_factory(tokens)
            values = getattr(
                configuration, "_" + type(configuration).__name__ + "__value"
            )
            values["build_enum"] = build_type
        original_dir = Path.cwd()
        try:
            if analysis:
                os.chdir(build)
                if "cppcheck" in analysis and configuration is not None:
                    result = configuration.cppcheck(str(source), str(build))
                    if result:
                        return int(result)
            if "verifycache" in phases:
                verify_cache(
                    build,
                    options.target[0] if options.target else None,
                    options.parallel,
                )
            if "test" in phases:
                command = [
                    "ctest",
                    "--test-dir",
                    str(build),
                    "-C",
                    build_type,
                    "--output-on-failure",
                    "--no-tests=error",
                    "--timeout",
                    str(options.test_timeout),
                ]
                if options.test_filter:
                    command.extend(["-R", options.test_filter])
                if options.test_label:
                    command.extend(["-L", options.test_label])
                if options.memcheck or "valgrind" in tokens:
                    command.extend(["-T", "memcheck"])
                if options.verbose:
                    command.append("-VV")
                _run(command)
            if "install" in phases:
                command = ["cmake", "--install", str(build), "--config", build_type]
                if options.install_prefix:
                    command.extend(["--prefix", options.install_prefix])
                _run(command)
            if analysis and configuration is not None:
                for action in ("coverage", "spell", "benchmark"):
                    if action in analysis:
                        method = getattr(configuration, action, None)
                        if method is None:
                            continue
                        result = (
                            method(str(build))
                            if action == "benchmark"
                            else method(str(source))
                            if action == "spell"
                            else method(str(source), str(build))
                        )
                        if result:
                            return int(result)
        finally:
            os.chdir(original_dir)
        return 0
    except subprocess.CalledProcessError as error:
        return error.returncode
    except (OSError, RuntimeError, ValueError) as error:
        print(error)
        return 1


def dispatch_bazel(
    arguments: list[str],
    root: str | None = None,
    force: bool = False,
    configuration_factory: Callable[[list[str]], Any] | None = None,
    parse_legacy: Callable[[list[str]], list[str]] | None = None,
) -> int | None:
    """Normalize named options and dotted tokens into one Bazel pipeline."""
    if arguments == ["--help"]:
        print_bazel_help()
        return 0 if force else None
    options, remaining = parse_options(arguments, bazel=True)
    root_path = Path(root or Path(__file__).resolve().parents[2]).resolve()
    if not arguments:
        argument_parser(True).error("specify a build action; see --help")
    if configuration_factory is None:
        # Keep this entry point usable by callers and CLI regression tests.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "memory_bazel_setup", root_path / "Scripts/setup_bazel.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load Bazel setup module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        configuration_factory = module.MemoryBazelConfiguration
        parse_legacy = module.parse_args
    if parse_legacy is None:
        raise RuntimeError("parse_legacy must be provided or loaded from setup_bazel")
    try:
        legacy = parse_legacy(remaining)
        if not isinstance(legacy, list) or (remaining and not legacy):
            raise ValueError(
                "Bazel arguments could not be parsed into an action; see --help"
            )
        config = configuration_factory(legacy)
        config.targets = options.target or ["//..."]
        config.output_root = (
            str(Path(options.output_root).resolve()) if options.output_root else None
        )
        if options.compilation_mode:
            config.build_type = {
                "dbg": "debug",
                "opt": "release",
                "fastbuild": "fastbuild",
            }[options.compilation_mode]
        if options.compiler:
            config._set_compiler(options.compiler)
        if options.cxx_standard:
            config.cxx_standard = "cxx" + options.cxx_standard
        if options.gpu_backend:
            config.gpu_backend = options.gpu_backend
        for option, attribute in (
            ("tbb", "enable_tbb"),
            ("numa", "enable_numa"),
            ("memkind", "enable_memkind"),
            ("profiler", "enable_profiler"),
            ("shared", "shared_libs"),
        ):
            value = getattr(options, option)
            if value is not None:
                setattr(config, attribute, value == "ON")
        if options.mimalloc is not None:
            config.disable_mimalloc = options.mimalloc == "OFF"
        config.use_batch = options.batch or config.use_batch
        if options.test_output:
            config.test_output = options.test_output
        if not any(
            (
                config.run_build,
                config.run_tests,
                config.run_clean,
                config.run_config,
                config.run_coverage,
            )
        ):
            # Feature tokens such as "release" still mean "build". An empty parse
            # does not: that is a missing action, not a request to build.
            if not legacy:
                raise ValueError("No Bazel action specified; see --help")
            config.run_build = True
        original_dir = Path.cwd()
        try:
            os.chdir(root_path)
            config.execute()
        finally:
            os.chdir(original_dir)
        return 0
    except subprocess.CalledProcessError as error:
        return error.returncode
    except (OSError, RuntimeError, ValueError) as error:
        print(error)
        return 1

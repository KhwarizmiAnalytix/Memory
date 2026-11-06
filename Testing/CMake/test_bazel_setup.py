"""Regression coverage for the shared named/dotted Bazel execution path."""

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "Scripts"
SPEC = importlib.util.spec_from_file_location("bazel_setup", SCRIPTS / "setup_bazel.py")
assert SPEC is not None and SPEC.loader is not None
setup = importlib.util.module_from_spec(SPEC)
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC.loader.exec_module(setup)
    from helpers import cli


class BazelSetupTests(unittest.TestCase):
    """Regression coverage for the shared named/dotted Bazel execution path."""

    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.prefix = patch.object(setup, "bazel_prefix", side_effect=lambda: ["bazel"])
        self.prefix.start()
        self.addCleanup(self.prefix.stop)

    def dispatch(self, arguments):
        return cli.dispatch_bazel(
            arguments,
            ROOT,
            configuration_factory=setup.MemoryBazelConfiguration,
            parse_legacy=setup.parse_args,
        )

    def commands(self, arguments):
        with patch.object(subprocess, "run", return_value=Mock(returncode=0)) as run:
            self.assertEqual(self.dispatch(arguments), 0)
        return [call.args[0] for call in run.call_args_list]

    def test_named_target_preserves_dotted_features_and_defaults(self):
        plain = self.commands(["build.test.tbb.nomimalloc.shared"])
        named = self.commands(
            [
                "build.test.tbb.nomimalloc.shared",
                "--target=//Testing/...",
                "--compiler=default",
            ]
        )
        self.assertEqual(len(plain), 2)
        self.assertEqual(len(named), 2)
        for first, second in zip(plain, named):
            self.assertEqual(first[:-1], second[:-1])
            self.assertEqual(second[-1], "//Testing/...")
            self.assertIn("--define=memory_enable_tbb=true", second)
            self.assertIn("--define=memory_enable_mimalloc=false", second)
            self.assertIn("--define=build_shared_libs=true", second)
            self.assertFalse(any(arg.startswith("--repo_env") for arg in second))

    def test_fastbuild_emits_compilation_mode(self):
        command = self.commands(["build"])[0]
        self.assertEqual(command[1:4], ["build", "-c", "fastbuild"])
        named = self.commands(["build", "--compilation-mode=fastbuild"])[0]
        self.assertEqual(named[1:4], ["build", "-c", "fastbuild"])

    def test_empty_parse_does_not_default_to_build(self):
        with patch.object(subprocess, "run") as run:
            self.assertEqual(self.dispatch(["--compiler=clang"]), 1)
        run.assert_not_called()

    def test_feature_token_without_action_still_builds(self):
        command = self.commands(["release"])[0]
        self.assertEqual(command[1], "build")
        self.assertEqual(command[2:4], ["-c", "opt"])

    def test_shared_library_env_uses_the_platform_loader_path(self):
        spec = importlib.util.spec_from_file_location(
            "verify_bazel_package", SCRIPTS / "verify_bazel_package.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        lib = Path("/opt/memory/lib")
        with patch.object(module.sys, "platform", "linux"):
            env = module.library_env(
                lib, {"LD_LIBRARY_PATH": "/usr/lib", "PATH": "/usr/bin"}
            )
        self.assertEqual(
            env["LD_LIBRARY_PATH"], "/opt/memory/lib" + os.pathsep + "/usr/lib"
        )
        self.assertEqual(env["PATH"], "/usr/bin")
        with patch.object(module.sys, "platform", "darwin"):
            env = module.library_env(lib, {"PATH": "/usr/bin"})
        self.assertEqual(env["DYLD_LIBRARY_PATH"], "/opt/memory/lib" + os.pathsep)
        self.assertEqual(env["PATH"], "/usr/bin")
        with patch.object(module.sys, "platform", "win32"):
            env = module.library_env(Path("C:/memory/lib"), {"PATH": "C:/Windows"})
        self.assertTrue(env["PATH"].startswith("C:/memory/lib" + os.pathsep))

    def test_unknown_build_type_does_not_drop_compilation_mode(self):
        config = setup.MemoryBazelConfiguration(["build"])
        config.build_type = "minsizerel"
        with self.assertRaises(ValueError):
            config.build_bazel_command("build")

    def test_llvm_coverage_requires_explicit_clang(self):
        config = setup.MemoryBazelConfiguration(["coverage"])
        self.assertIsNone(config.compiler)
        for compiler in (None, "gcc", "msvc", "clang-cl"):
            config.compiler = compiler
            with patch.object(
                setup.MemoryBazelConfiguration, "_find_recent_profraw_files"
            ) as profraw:
                self.assertIsNone(config._build_llvm_coverage_report(0.0))
            profraw.assert_not_called()
        config._set_compiler("clang")
        with patch.object(
            setup.MemoryBazelConfiguration,
            "_find_recent_profraw_files",
            return_value=[],
        ) as profraw:
            self.assertIsNone(config._build_llvm_coverage_report(0.0))
        profraw.assert_called_once()

    def test_named_choices_override_dotted_choices(self):
        command = self.commands(
            [
                "build.cuda.shared.nomimalloc",
                "--gpu-backend=none",
                "--shared=off",
                "--mimalloc=on",
                "--compilation-mode=dbg",
                "--compiler=gcc",
            ]
        )[0]
        self.assertIn("--define=memory_gpu_backend=none", command)
        self.assertNotIn("--define=memory_gpu_backend=cuda", command)
        self.assertNotIn("--define=build_shared_libs=true", command)
        self.assertNotIn("--define=memory_enable_mimalloc=false", command)
        self.assertIn("--repo_env=CC=gcc", command)
        self.assertEqual(command[2:4], ["-c", "dbg"])

    def test_config_does_not_clean_or_kill_processes(self):
        with tempfile.TemporaryDirectory() as temp:
            # The helper resolve()s the root (macOS: /tmp -> /private/tmp); expect the same.
            output_root = Path(temp) / "bazel state"
            commands = self.commands(
                ["config.build.test", "--batch", f"--output-root={output_root}"]
            )
            expected = f"--output_user_root={output_root.resolve()}"
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertEqual(command[:3], ["bazel", "--batch", expected])
        config = self.commands(["config"])
        self.assertEqual(len(config), 1)
        self.assertIn("--nobuild", config[0])

    def test_cxx_standard_uses_platform_compiler_syntax(self):
        config = setup.MemoryBazelConfiguration(["build", "cxx23"])
        config.system = "Windows"
        self.assertIn("--cxxopt=/std:c++23", config.build_bazel_command("build"))
        config.system = "Linux"
        self.assertIn("--cxxopt=-std=c++23", config.build_bazel_command("build"))

    def test_unknown_feature_fails_without_running_bazel(self):
        with patch.object(subprocess, "run") as run:
            self.assertEqual(self.dispatch(["build.typo", "--target=//..."]), 1)
        run.assert_not_called()

    def test_named_coverage_builds_and_validates_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "coverage.info"
            report.write_text("SF:src/allocator.cpp\nDA:1,1\nend_of_record\n")
            with (
                patch.object(subprocess, "run", return_value=Mock(returncode=0)) as run,
                patch.object(
                    setup.MemoryBazelConfiguration,
                    "_build_llvm_coverage_report",
                    return_value=str(report),
                ) as build_report,
                patch.object(
                    setup.MemoryBazelConfiguration, "_generate_coverage_html"
                ) as html,
            ):
                self.assertEqual(
                    self.dispatch(["build.test.coverage", "--compiler=clang"]), 0
                )
            command = run.call_args.args[0]
            self.assertEqual(command[1], "coverage")
            self.assertIn("--nocache_test_results", command)
            self.assertLess(
                command.index("--combined_report=lcov"), command.index("--")
            )
            build_report.assert_called_once()
            html.assert_called_once_with(str(report))
            report.write_text("")
            with (
                patch.object(subprocess, "run", return_value=Mock(returncode=0)),
                patch.object(
                    setup.MemoryBazelConfiguration,
                    "_build_llvm_coverage_report",
                    return_value=str(report),
                ),
            ):
                self.assertEqual(self.dispatch(["coverage", "--compiler=clang"]), 1)

    def test_report_queries_share_startup_options_and_configuration(self):
        config = setup.MemoryBazelConfiguration(["coverage", "release", "clang"])
        config.output_root = "/tmp/isolated bazel"
        config.use_batch = True
        with patch.object(
            subprocess, "run", return_value=Mock(stdout="/tmp/out", returncode=0)
        ) as run:
            self.assertEqual(
                config._bazel_info("output_path", "", match_config=True), "/tmp/out"
            )
        command = run.call_args.args[0]
        self.assertEqual(
            command[:4],
            ["bazel", "--batch", "--output_user_root=/tmp/isolated bazel", "info"],
        )
        self.assertIn("opt", command)
        self.assertIn("--repo_env=CC=clang", command)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Determine lower bounds for core dependencies by running the test suite.

The script reads runtime dependencies from ``pyproject.toml``, builds isolated
virtual environments in a temporary directory, and walks available PyPI
versions downward from a known-good baseline. Each dependency is tested in
isolation while the remaining dependencies stay pinned to their baseline
versions. ``jax`` and ``jaxlib`` are always moved together.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDED_DEPS = {"pytest"}
JAX_GROUP = ("jax", "jaxlib")


@dataclass(frozen=True)
class TrialResult:
    target: tuple[str, ...]
    version_map: dict[str, str]
    passed: bool
    returncode: int
    log_path: Path
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the project test suite against older dependency versions to "
            "determine lower bounds."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing pyproject.toml and tests/.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to create trial virtual environments.",
    )
    parser.add_argument(
        "--test-command",
        default="python -m pytest",
        help=(
            "Command to execute the regression suite inside each trial "
            "environment. Default: python -m pytest"
        ),
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Parent directory for trial virtual environments and logs.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON summary.",
    )
    parser.add_argument(
        "--max-versions",
        type=int,
        default=None,
        help="Optional cap on how many historical versions to test per target.",
    )
    parser.add_argument(
        "--deps",
        nargs="*",
        default=None,
        help=(
            "Optional list of dependency names to test. Use 'jax' to test the "
            "jax/jaxlib pair."
        ),
    )
    parser.add_argument(
        "--include-prerelease",
        action="store_true",
        help="Include prereleases when enumerating candidate versions from PyPI.",
    )
    return parser.parse_args()


def load_project_dependencies(pyproject_path: Path) -> list[str]:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    deps = data["project"]["dependencies"]
    names: list[str] = []
    for dep in deps:
        name = dep.split("[", 1)[0]
        for sep in ("<", ">", "=", "!", "~", ";"):
            if sep in name:
                name = name.split(sep, 1)[0]
        names.append(name.strip())
    return names


def get_core_dependencies(pyproject_path: Path) -> list[str]:
    deps = load_project_dependencies(pyproject_path)
    return [dep for dep in deps if dep not in DEFAULT_EXCLUDED_DEPS]


def get_installed_version(python_exe: str, package: str) -> str:
    cmd = [
        python_exe,
        "-c",
        (
            "from importlib.metadata import version; "
            f"print(version({package!r}))"
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def build_baseline_versions(python_exe: str, dependencies: Iterable[str]) -> dict[str, str]:
    return {dep: get_installed_version(python_exe, dep) for dep in dependencies}


def fetch_versions_from_pypi(package: str, include_prerelease: bool) -> list[str]:
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url) as response:
            payload = json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"failed to query PyPI for {package}: {exc}") from exc

    versions: list[str] = []
    for raw_version in payload.get("releases", {}):
        parsed = parse_version(raw_version)
        if parsed is None:
            continue
        if not include_prerelease and parsed["prerelease"]:
            continue
        versions.append(raw_version)
    versions.sort(key=version_sort_key)
    return versions


def parse_version(version: str) -> dict[str, object] | None:
    normalized = version.strip()
    prerelease = False
    for marker in ("a", "b", "rc", "dev", "post"):
        if marker in normalized:
            prerelease = True
            break

    head = []
    token = ""
    for char in normalized:
        if char.isdigit():
            token += char
            continue
        if char == ".":
            if token:
                head.append(int(token))
                token = ""
            else:
                return None
            continue
        if token:
            head.append(int(token))
        break
    else:
        if token:
            head.append(int(token))

    if not head:
        return None

    return {"parts": tuple(head), "prerelease": prerelease}


def version_sort_key(version: str) -> tuple[int, ...]:
    parsed = parse_version(version)
    if parsed is None:
        return tuple()
    return parsed["parts"]  # type: ignore[return-value]


def candidate_versions(
    package: str,
    baseline_version: str,
    include_prerelease: bool,
    max_versions: int | None,
) -> list[str]:
    versions = fetch_versions_from_pypi(package, include_prerelease)
    filtered = [version for version in versions if version_sort_key(version) <= version_sort_key(baseline_version)]
    filtered.sort(key=version_sort_key, reverse=True)
    if max_versions is not None:
        filtered = filtered[:max_versions]
    return filtered


def resolve_targets(core_dependencies: list[str], requested: list[str] | None) -> list[tuple[str, ...]]:
    available = set(core_dependencies)
    targets: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    names = requested or core_dependencies
    for name in names:
        if name == "jax":
            target = JAX_GROUP
        elif name == "jaxlib":
            target = JAX_GROUP
        else:
            target = (name,)

        missing = [dep for dep in target if dep not in available]
        if missing:
            raise ValueError(f"unknown dependency target: {', '.join(missing)}")

        if target not in seen:
            targets.append(target)
            seen.add(target)
    return targets


def create_trial_environment(parent: Path, python_exe: str, name: str) -> tuple[Path, Path]:
    env_dir = parent / name
    if env_dir.exists():
        shutil.rmtree(env_dir)

    subprocess.run([python_exe, "-m", "venv", str(env_dir)], check=True)

    if os.name == "nt":
        env_python = env_dir / "Scripts" / "python.exe"
    else:
        env_python = env_dir / "bin" / "python"

    subprocess.run([str(env_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    return env_dir, env_python


def build_install_spec_map(
    baseline_versions: dict[str, str],
    target: tuple[str, ...],
    candidate_version: str,
) -> dict[str, str]:
    specs = dict(baseline_versions)
    for dep in target:
        specs[dep] = candidate_version
    return specs


def install_trial_dependencies(env_python: Path, version_map: dict[str, str]) -> None:
    packages = [f"{name}=={version}" for name, version in sorted(version_map.items())]
    cmd = [str(env_python), "-m", "pip", "install", "pytest", *packages]
    subprocess.run(cmd, check=True)


def run_trial(
    env_python: Path,
    project_root: Path,
    test_command: str,
    log_path: Path,
) -> tuple[bool, int]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root / "src")
    if test_command == "python -m pytest":
        command = [str(env_python), "-m", "pytest"]
        proc = subprocess.run(
            command,
            cwd=project_root,
            env=env,
            text=True,
            capture_output=True,
        )
        rendered_command = shlex.join(command)
    else:
        command = test_command.replace("python", f'"{env_python}"', 1)
        proc = subprocess.run(
            command,
            cwd=project_root,
            env=env,
            shell=True,
            text=True,
            capture_output=True,
        )
        rendered_command = command
    log_path.write_text(
        textwrap.dedent(
            f"""\
            COMMAND: {rendered_command}
            RETURN CODE: {proc.returncode}

            STDOUT
            ======
            {proc.stdout}

            STDERR
            ======
            {proc.stderr}
            """
        ),
        encoding="utf-8",
    )
    return proc.returncode == 0, proc.returncode


def evaluate_candidate(
    trial_root: Path,
    python_exe: str,
    project_root: Path,
    test_command: str,
    baseline_versions: dict[str, str],
    target: tuple[str, ...],
    candidate_version: str,
) -> TrialResult:
    label = "-".join(target)
    version_map = build_install_spec_map(baseline_versions, target, candidate_version)
    safe_version = candidate_version.replace(".", "_")
    env_name = f"{label}-{safe_version}"
    log_path = trial_root / f"{env_name}.log"

    try:
        _, env_python = create_trial_environment(trial_root, python_exe, env_name)
        install_trial_dependencies(env_python, version_map)
        passed, returncode = run_trial(env_python, project_root, test_command, log_path)
        return TrialResult(target, version_map, passed, returncode, log_path)
    except subprocess.CalledProcessError as exc:
        message = f"installation failed: {' '.join(map(str, exc.cmd))}"
        log_path.write_text(message, encoding="utf-8")
        return TrialResult(target, version_map, False, exc.returncode, log_path, error=message)


def find_lower_bound_for_target(
    trial_root: Path,
    python_exe: str,
    project_root: Path,
    test_command: str,
    baseline_versions: dict[str, str],
    target: tuple[str, ...],
    include_prerelease: bool,
    max_versions: int | None,
) -> tuple[str | None, list[TrialResult]]:
    anchor_package = target[0]
    baseline_version = baseline_versions[anchor_package]
    versions = candidate_versions(anchor_package, baseline_version, include_prerelease, max_versions)
    trial_results: list[TrialResult] = []
    last_passing_version: str | None = None

    for version in versions:
        print(f"[{'+'.join(target)}] testing {version}")
        result = evaluate_candidate(
            trial_root=trial_root,
            python_exe=python_exe,
            project_root=project_root,
            test_command=test_command,
            baseline_versions=baseline_versions,
            target=target,
            candidate_version=version,
        )
        trial_results.append(result)
        if result.passed:
            last_passing_version = version
            continue
        if last_passing_version is not None:
            break

    return last_passing_version, trial_results


def print_summary(
    baseline_versions: dict[str, str],
    lower_bounds: dict[str, str | None],
    results: dict[str, list[TrialResult]],
) -> None:
    print("Baseline versions:")
    for dep, version in sorted(baseline_versions.items()):
        print(f"  {dep}=={version}")

    print("\nDetected lower bounds:")
    for target_name, version in sorted(lower_bounds.items()):
        display = version if version is not None else "no passing candidate found"
        print(f"  {target_name}: {display}")

    print("\nTrial logs:")
    for target_name, target_results in sorted(results.items()):
        for result in target_results:
            outcome = "PASS" if result.passed else "FAIL"
            print(f"  {target_name} {outcome} -> {result.log_path}")


def write_report(
    report_json: Path,
    baseline_versions: dict[str, str],
    lower_bounds: dict[str, str | None],
    results: dict[str, list[TrialResult]],
) -> None:
    payload = {
        "baseline_versions": baseline_versions,
        "lower_bounds": lower_bounds,
        "results": {
            target_name: [
                {
                    "target": list(result.target),
                    "version_map": result.version_map,
                    "passed": result.passed,
                    "returncode": result.returncode,
                    "log_path": str(result.log_path),
                    "error": result.error,
                }
                for result in target_results
            ]
            for target_name, target_results in results.items()
        },
    }
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    pyproject_path = project_root / "pyproject.toml"
    core_dependencies = get_core_dependencies(pyproject_path)
    targets = resolve_targets(core_dependencies, args.deps)
    baseline_versions = build_baseline_versions(args.python, core_dependencies)

    trial_root_parent = args.tmp_dir.resolve() if args.tmp_dir else Path(tempfile.mkdtemp(prefix="chillprop-lower-bounds-"))
    trial_root_parent.mkdir(parents=True, exist_ok=True)
    trial_root = trial_root_parent / "trials"
    trial_root.mkdir(parents=True, exist_ok=True)

    lower_bounds: dict[str, str | None] = {}
    results: dict[str, list[TrialResult]] = {}

    for target in targets:
        target_name = "+".join(target)
        lower_bound, trial_results = find_lower_bound_for_target(
            trial_root=trial_root,
            python_exe=args.python,
            project_root=project_root,
            test_command=args.test_command,
            baseline_versions=baseline_versions,
            target=target,
            include_prerelease=args.include_prerelease,
            max_versions=args.max_versions,
        )
        lower_bounds[target_name] = lower_bound
        results[target_name] = trial_results

    print_summary(baseline_versions, lower_bounds, results)

    if args.report_json:
        write_report(args.report_json.resolve(), baseline_versions, lower_bounds, results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Isolated execution of an untrusted manifold-bench submission."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np

from .protocol import Prediction, load_prediction
from .zoo import Dataset, public_config


MAX_FILES = 2_000
MAX_SUBMISSION_BYTES = 256 * 1024 * 1024
OUTPUT_TAIL_BYTES = 16_384
CPU_THREAD_LIMIT = 4
# A scored row transformed alone versus embedded among unrelated rows may differ
# only by floating-point reduction-order noise; genuine batch transduction moves
# outputs far above this scale-relative tolerance.
BATCH_CONTEXT_TOLERANCE = 1e-3
BANNED_PYTHON_MODULES = (
    "sklearn",
    "sae_lens",
    "dictionary_learning",
    "gamfit",
    "manifold_sae",
)
BANNED_NON_PYTHON_MARKERS = (*BANNED_PYTHON_MODULES, "sae-lens")
DEPENDENCY_FILES = {"package.json", "pyproject.toml", "requirements.txt"}


class SubmissionError(RuntimeError):
    pass


# A zero must say WHY. "The submission exceeded the stated runtime" and "the
# submission ran fine and discovered no structure" are opposite verdicts implying
# opposite next actions, and both used to render as 0.000 with no way to tell them
# apart from the artifact. Both Haiku rollouts on this env scored exactly 0.000 --
# one on a KeyError reading config.json, one on a fit timeout -- and neither score
# distinguished itself from a genuine failure to recover any manifold structure.
CONTRACT_FAULTS = (
    ("runtime_exceeded", ("exceeded",)),
    ("nondeterministic_transform", ("depends on unrelated rows", "inconsistent feature dictionary")),
    ("mutated_grader_input", ("modified train.npz", "modified config.json", "modified eval.npz")),
    ("forbidden_dependency", ("forbidden end-to-end modeling dependency",)),
    ("unsafe_tree", ("symlink", "special file", "exceeds file limits")),
    ("packaging", ("must be a regular executable file", "must be a regular file",
                   "must be a regular directory")),
    ("protocol", ("predictions.npz", "prediction arrays", "presence scores",
                  "learned feature count", "reconstruction must", "contributions must",
                  "bias must")),
    # Gross violations of the declared hard invariants, set directly by the
    # verifier from the Integrity measurement (not from an error message).
    ("additive_identity_violated", ("additive_identity_violated",)),
    ("permutation_equivariance_violated", ("permutation_equivariance_violated",)),
    ("crashed", ("exited",)),
)


def classify_fault(message: str) -> str:
    """Name the contract fault behind an invalid suite.

    Ordered most-specific first: a timeout also produces a non-zero exit, and
    reporting that as a crash would send a reader to debug the wrong thing.
    """
    lowered = message.casefold()
    for name, markers in CONTRACT_FAULTS:
        if any(marker.casefold() in lowered for marker in markers):
            return name
    return "unclassified"


class SandboxUnavailable(RuntimeError):
    """The trusted execution jail is unavailable or failed its active probe."""


class PreparedSubmission:
    def __init__(self, root: Path, build_seconds: float):
        self.root = root
        self.build_seconds = build_seconds

    def close(self) -> None:
        shutil.rmtree(self.root.parent, ignore_errors=True)


class RunResult:
    def __init__(
        self,
        match: Prediction,
        score: Prediction,
        permuted: Prediction,
        permutation_inverse: np.ndarray,
        fit_seconds: float,
        transform_seconds: float,
        log_tail: str,
    ):
        self.match = match
        self.score = score
        self.permuted = permuted
        self.permutation_inverse = permutation_inverse
        self.fit_seconds = fit_seconds
        self.transform_seconds = transform_seconds
        self.log_tail = log_tail


def _regular_tree(root: Path) -> tuple[int, int]:
    count = 0
    total = 0
    for path in root.rglob("*"):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        ):
            raise SubmissionError(
                f"submission contains a symlink or special file: {path.relative_to(root)}"
            )
        if stat.S_ISREG(info.st_mode):
            count += 1
            total += info.st_size
    if count > MAX_FILES or total > MAX_SUBMISSION_BYTES:
        raise SubmissionError(f"submission exceeds file limits: files={count}, bytes={total}")
    return count, total


def _constant_string(node: ast.AST) -> str | None:
    """Evaluate only literal string concatenation used as an import name.

    v13 flagged ``__import__("sklearn")`` but not ``__import__("sk" + "learn")``
    because it required a single ``ast.Constant`` argument. Folding literal
    string concatenation closes that obfuscation without executing anything.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant_string(node.left)
        right = _constant_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _forbidden_python_import(text: str) -> str | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        module: str | None = None
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            parent = node.module or ""
            names = [parent, *(f"{parent}.{alias.name}" for alias in node.names)]
        else:
            names = []
        for name in names:
            module = name.casefold()
            marker = next(
                (
                    item
                    for item in BANNED_PYTHON_MODULES
                    if module == item or module.startswith(f"{item}.")
                ),
                None,
            )
            if marker is not None:
                return marker
        if not isinstance(node, ast.Call) or not node.args:
            continue
        function = node.func
        dynamic_import = isinstance(function, ast.Name) and function.id == "__import__"
        dynamic_import |= isinstance(function, ast.Attribute) and function.attr == "import_module"
        import_name = _constant_string(node.args[0]) if dynamic_import else None
        if import_name is not None:
            module = import_name.casefold()
            marker = next(
                (
                    item
                    for item in BANNED_PYTHON_MODULES
                    if module == item or module.startswith(f"{item}.")
                ),
                None,
            )
            if marker is not None:
                return marker
    return None


def _forbidden_archive_member(names: list[str]) -> str | None:
    """Return a banned package named as an archive path component.

    A wheel/zip keeps the forbidden package name in its central directory even
    when the outer file is renamed (``sklearn`` -> ``data.dat``). Names only are
    inspected; no member is decompressed or executed.
    """
    for name in names:
        for component in name.casefold().replace("-", "_").split("/"):
            for marker in BANNED_PYTHON_MODULES:
                normalized = marker.replace("-", "_")
                if component == normalized or component.startswith(
                    (f"{normalized}.", f"{normalized}-")
                ):
                    return marker
    return None


def _scan_source(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Renamed-archive dependency scan: inspect the central directory names
        # of any zip/wheel regardless of its outer name, before the text-size
        # skip below (wheels usually exceed the text cap).
        if zipfile.is_zipfile(path):
            try:
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
            except (OSError, zipfile.BadZipFile):
                names = []
            marker = _forbidden_archive_member(names)
            if marker is not None:
                raise SubmissionError(
                    f"forbidden end-to-end modeling dependency {marker!r} in {path.relative_to(root)}"
                )
        if path.stat().st_size > 2_000_000:
            continue
        try:
            text = path.read_text(errors="strict").lower()
        except (UnicodeDecodeError, OSError):
            continue
        suffix = path.suffix.casefold()
        first_line = text.partition("\n")[0].casefold()
        is_python = suffix == ".py" or (not suffix and "python" in first_line)
        if is_python:
            marker = _forbidden_python_import(text)
        elif path.name.casefold() in DEPENDENCY_FILES:
            lowered = text.casefold()
            marker = next((item for item in BANNED_NON_PYTHON_MARKERS if item in lowered), None)
        else:
            marker = None
        if marker is not None:
            raise SubmissionError(
                f"forbidden end-to-end modeling dependency {marker!r} in {path.relative_to(root)}"
            )


def _require_entrypoints(root: Path) -> None:
    for name in ("fit", "transform"):
        path = root / name
        if not path.is_file() or path.is_symlink() or not os.access(path, os.X_OK):
            raise SubmissionError(f"{name} must be a regular executable file")
    build = root / "build.sh"
    if build.exists() and (not build.is_file() or build.is_symlink()):
        raise SubmissionError("build.sh must be a regular file")


def _cpu_pinned(command: list[str]) -> list[str]:
    """Restrict a submission to exactly the disclosed core budget.

    The public contract promises four cores, and both the wall-clock limits and
    the ``RLIMIT_CPU`` guard below are calibrated for that width.  On a wider
    host a submission that cannot meet the contract on four cores would pass on
    wall time, and a multiprocess submission would exhaust the CPU guard at a
    fraction of its wall budget.  Affinity is inherited by every child, so
    pinning the process group here makes grading identical on any instance size.
    """
    if sys.platform != "linux":
        return command
    taskset = shutil.which("taskset")
    available = os.cpu_count() or CPU_THREAD_LIMIT
    if taskset is None or available <= CPU_THREAD_LIMIT:
        return command
    return [taskset, "-c", f"0-{CPU_THREAD_LIMIT - 1}", *command]


def _resource_limits(timeout_seconds: int) -> None:
    def lower(kind: int, target: int) -> None:
        _, hard = resource.getrlimit(kind)
        limit = target if hard == resource.RLIM_INFINITY else min(target, hard)
        resource.setrlimit(kind, (limit, limit))

    # macOS reports RLIMIT_AS but rejects the production-sized setting from a
    # pre-exec child.  The deployed Linux verifier always applies it.
    if sys.platform != "darwin":
        lower(resource.RLIMIT_AS, 8 * 1024 * 1024 * 1024)
    # RLIMIT_CPU counts process CPU time across all worker threads, whereas the
    # public command budget is wall time on four available cores.  Giving a
    # multithreaded learner only ``timeout_seconds`` of aggregate CPU silently
    # turns a 75 s wall-time allowance into about 19 s at full utilization and
    # the kernel reports the resulting failure only as SIGKILL.  Keep the CPU
    # guard as defense in depth, but scale it to the disclosed core budget; the
    # process-group wall timer below remains the authoritative deadline.
    lower(resource.RLIMIT_CPU, timeout_seconds * CPU_THREAD_LIMIT + 10)
    lower(resource.RLIMIT_FSIZE, 600 * 1024 * 1024)
    lower(resource.RLIMIT_NOFILE, 256)


def _reject_unsafe_entries(root: Path) -> None:
    """Refuse links and special files before trusted code touches the run tree.

    Untrusted code owns this directory between commands, and every trusted
    pathname operation afterwards -- reading the log, hashing the inputs, writing
    the next eval batch, chowning for the next command -- resolves names inside
    it. A submission that replaces a name with a link to a host path turns each
    of those into a primitive against the host. A FIFO is a denial-of-service
    primitive against the same trusted reads, and devices or sockets have no
    legitimate place in persisted model state. Reject the whole class rather
    than trying to make each call site safe.
    """
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            path = Path(parent) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not (
                stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
            ):
                raise SubmissionError(
                    f"run directory contains a symlink or special file after execution: "
                    f"{path.relative_to(root)}"
                )


def _chown_tree(root: Path, uid: int, gid: int) -> None:
    # follow_symlinks=False throughout: os.chown() follows by default, so a
    # candidate-planted link would hand ownership of a host path to the sandbox
    # account. The tree is symlink-audited before this runs; this is the belt to
    # that suspenders.
    os.chown(root, uid, gid, follow_symlinks=False)
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            os.chown(Path(parent) / name, uid, gid, follow_symlinks=False)


def _sandbox_command(command: list[str], cwd: Path, account) -> list[str]:
    """Return a minimal-mount, networkless Bubblewrap command.

    UID separation alone is not isolation: a submission can still read any
    world-readable host path and reach the network through a renamed binary or a
    compiled socket call. The jail exposes only generic runtimes/toolchains and
    this one writable run directory; the repository, grader, other trials, and
    host network do not exist inside it.

    Exception: with MANIFOLD_BENCH_SANDBOX=external (packaged harbor tasks
    only) the platform owns the network jail and this returns the setpriv-only
    uid jail instead -- see the fenced HYPERFOCAL PATCH below.
    """
    if sys.platform != "linux" or os.geteuid() != 0:
        raise SandboxUnavailable("sandbox requires a Linux root verifier")
    # HYPERFOCAL PATCH(sandbox-external): in packaged harbor tasks the grader
    # cannot own the namespace jail -- bwrap as root creates mount/pid/net
    # namespaces directly, which needs CAP_SYS_ADMIN, and harbor grants kernel
    # capabilities only to its egress-control sidecar, never the task
    # container, so every bwrap invocation there fails with EPERM before the
    # first suite runs. The image bakes MANIFOLD_BENCH_SANDBOX=external
    # (hyperfocal.yaml packaging.dockerfileExtra) and the verifier phase runs
    # under harbor-native no-network (packaging.network.verifier), so the
    # PLATFORM owns egress while this branch keeps the uid jail: setpriv drops
    # to the sandbox account, private state is root-locked by sealPrivateState
    # (and asserted unreadable before grading), and commands run in the real
    # run directory via Popen cwd -- the contract's entry points use relative
    # paths, but interpreters absolutize script paths on re-open (F19), so the
    # grader-owned run-tree parents are 0711 in this mode (traversal without
    # listing, see _submission_tree_parent) and their random mkdtemp suffixes
    # keep one run's tree undiscoverable from another. preflight_sandbox()
    # still executes its
    # probe as the submission uid in this mode, but the probe asserts only
    # that sealed host state is unreadable: harbor verifiers legitimately run
    # with the network reachable (owner decision 15), so a reachable network
    # is the normal packaged condition, not evidence of a missing jail — see
    # HYPERFOCAL PATCH(preflight-network) in _preflight_probe below.
    # Known deltas vs bwrap, accepted for packaged tasks and
    # absent from native EC2 rollouts (which never set the variable): the host
    # /tmp is shared rather than a fresh per-command tmpfs, and world-readable
    # host paths remain visible to the submission uid.
    if os.environ.get("MANIFOLD_BENCH_SANDBOX") == "external":
        external_setpriv = shutil.which("setpriv")
        if external_setpriv is None:
            raise SandboxUnavailable("required setpriv sandbox tool is unavailable")
        return [
            external_setpriv,
            "--reuid",
            str(account.pw_uid),
            "--regid",
            str(account.pw_gid),
            "--clear-groups",
            "--inh-caps=-all",
            "--no-new-privs",
            *command,
        ]
    # end HYPERFOCAL PATCH(sandbox-external)
    bwrap = shutil.which("bwrap")
    setpriv = shutil.which("setpriv")
    if bwrap is None or setpriv is None:
        raise SandboxUnavailable("required bwrap/setpriv sandbox tools are unavailable")

    actual = [
        bwrap,
        "--unshare-pid",
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-cgroup",
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--dir",
        "/opt",
        "--dir",
        "/opt/hyperfocal",
        "--dir",
        "/etc",
    ]
    for source in (
        "/usr",
        "/bin",
        "/sbin",
        "/lib",
        "/lib64",
        "/opt/hyperfocal/manifold-bench",
    ):
        if os.path.exists(source):
            actual.extend(("--ro-bind", source, source))
    for source in ("/etc/passwd", "/etc/group", "/etc/ld.so.cache"):
        if os.path.exists(source):
            actual.extend(("--ro-bind", source, source))
    # Amazon Linux resolves the linker through /etc/alternatives. Omitting this
    # mount turns every legitimate compiled submission into a build failure.
    if os.path.isdir("/etc/alternatives"):
        actual.extend(("--ro-bind", "/etc/alternatives", "/etc/alternatives"))
    actual.extend(
        (
            "--bind",
            str(cwd),
            "/work",
            "--chdir",
            "/work",
            "--",
            setpriv,
            "--reuid",
            str(account.pw_uid),
            "--regid",
            str(account.pw_gid),
            "--clear-groups",
            "--inh-caps=-all",
            "--no-new-privs",
            *command,
        )
    )
    return actual


def _preflight_probe(secret: Path, *, external: bool) -> str:
    """Return the preflight probe source for the active sandbox mode.

    Both modes assert that the root-sealed ``secret`` is unreadable from the
    submission uid — that readability check is the integrity property grading
    actually depends on (the same sealing protects grader/, reference/ and
    rollout-analysis/, independently re-asserted by
    assertPrivateStateUnreadable() in environment/src/index.ts before any
    grade).

    HYPERFOCAL PATCH(preflight-network): the reachable-network assert runs
    only in bwrap mode, where this grader owns the jail (--unshare-net) and a
    reachable network proves the jail is broken. Under
    MANIFOLD_BENCH_SANDBOX=external (packaged harbor tasks) the platform owns
    egress and harbor verifiers legitimately run with the network reachable
    (owner decision 15), so the old assert fail-closed every packaged grade
    (F18: SystemExit('network is reachable') -> grader_error, reward 0) while
    detecting nothing wrong. Verify-time egress also confers no advantage the
    agent phase did not already have — the agent had full network while
    producing the submission, and the held-out data and reference answers the
    jail protects are local, root-0700, and covered by the readability assert
    above. Native EC2 rollouts never set the variable and keep the assert.
    """
    probe = (
        "from pathlib import Path\n"
        # Visibility means READABILITY, not existence: under the bwrap jail
        # the host /tmp is a fresh tmpfs (FileNotFoundError), while under
        # MANIFOLD_BENCH_SANDBOX=external the host /tmp is shared and the
        # root-0700 preflight dir makes any submission-uid stat/open raise
        # PermissionError — which IS the sealed state. Path.exists() re-raises
        # PermissionError on Python 3.12, so the old existence assert crashed
        # every packaged-mode grade before the first suite (dress-rehearsal
        # finding F14, 2026-07-17). A jail that is actually absent (probe
        # running with root-equivalent access) still reads the secret and
        # fails loudly.\n
        "try:\n"
        f" visible = Path({str(secret)!r}).read_text() == 'must not be visible'\n"
        "except (FileNotFoundError, PermissionError):\n"
        " visible = False\n"
        "assert not visible, 'host file is visible'\n"
    )
    if not external:
        probe += (
            "import socket\n"
            "s=socket.socket(); s.settimeout(1.0)\n"
            "try:\n"
            " s.connect(('1.1.1.1',53))\n"
            "except OSError:\n"
            " pass\n"
            "else:\n"
            " raise SystemExit('network is reachable')\n"
            "finally:\n"
            " s.close()\n"
        )
    return probe


def preflight_sandbox() -> None:
    """Prove the uid jail seals host files before touching a submission — and,
    in bwrap mode where this grader owns the network jail, that egress is
    absent (see _preflight_probe for why external mode does not assert
    no-network).

    A missing/broken namespace tool is grader infrastructure failure, never a
    candidate zero. This trusted probe converts every setup failure into the
    distinct ``SandboxUnavailable`` path before candidate execution begins.
    """
    parent = Path(tempfile.mkdtemp(prefix="manifold-bench-preflight-"))
    work = parent / "work"
    work.mkdir()
    secret = parent / "host-secret"
    secret.write_text("must not be visible")
    probe = _preflight_probe(
        secret,
        external=os.environ.get("MANIFOLD_BENCH_SANDBOX") == "external",
    )
    try:
        try:
            _run_command(
                ["/opt/hyperfocal/manifold-bench/bin/python", "-c", probe],
                work,
                15,
                sandboxed=True,
            )
        except (SubmissionError, SandboxUnavailable) as error:
            raise SandboxUnavailable(f"sandbox preflight failed: {error}") from error
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def _run_command(
    command: list[str],
    cwd: Path,
    timeout_seconds: int,
    *,
    sandboxed: bool,
) -> tuple[float, str]:
    env = {
        "PATH": "/opt/hyperfocal/manifold-bench/bin:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/work",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": str(CPU_THREAD_LIMIT),
        "OPENBLAS_NUM_THREADS": str(CPU_THREAD_LIMIT),
        "MKL_NUM_THREADS": str(CPU_THREAD_LIMIT),
        "NUMEXPR_NUM_THREADS": str(CPU_THREAD_LIMIT),
    }
    (cwd / "tmp").mkdir(exist_ok=True)
    actual = command
    if sandboxed:
        try:
            import pwd

            account = pwd.getpwnam("manifoldsub")
        except (ImportError, KeyError) as error:
            raise SubmissionError("sandbox account manifoldsub is unavailable") from error
        _chown_tree(cwd, account.pw_uid, account.pw_gid)
        env["HOME"] = "/work"
        env["TMPDIR"] = "/tmp"
        actual = _sandbox_command(command, cwd, account)
    actual = _cpu_pinned(actual)
    # The log lives OUTSIDE the candidate-writable directory. It used to be
    # cwd/".runner.log", which the submission could delete and replace with a
    # symlink to any host path; the trusted read_bytes() below then followed it
    # and returned the host file's contents in the graded result. The child still
    # writes through the inherited descriptor, so nothing about capture changes.
    log_directory = Path(tempfile.mkdtemp(prefix="manifold-bench-log-"))
    log_path = log_directory / "runner.log"
    started = time.monotonic()
    with log_path.open("ab") as log:
        try:
            process = subprocess.Popen(
                actual,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=lambda: _resource_limits(timeout_seconds),
            )
        except OSError as error:
            raise SubmissionError(f"cannot execute {' '.join(command)}: {error}") from error
        try:
            code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, 9)
            process.wait()
            raise SubmissionError(f"{' '.join(command)} exceeded {timeout_seconds}s") from error
    elapsed = time.monotonic() - started
    try:
        tail = log_path.read_bytes()[-OUTPUT_TAIL_BYTES:].decode("utf8", errors="replace")
    finally:
        shutil.rmtree(log_directory, ignore_errors=True)
    # Audit before reporting: a non-zero exit is still a submission that may have
    # left links behind, and the caller hashes and rewrites this tree next.
    _reject_unsafe_entries(cwd)
    if code != 0:
        raise SubmissionError(f"{' '.join(command)} exited {code}: {tail}")
    return elapsed, tail


def _submission_tree_parent(prefix: str) -> Path:
    """mkdtemp a grader-owned parent for a tree the submission uid executes in.

    HYPERFOCAL PATCH(external-run-parent): mkdtemp creates the parent 0700, and
    under MANIFOLD_BENCH_SANDBOX=external the submission runs in the REAL run
    directory (no bwrap ``--bind cwd /work``), so every absolutized path the
    submission's own interpreter opens must traverse this root-owned parent.
    ``execve("./fit")`` succeeds (resolved via the cwd fd), but CPython 3.8+
    absolutizes the script path before reopening it, and that open EACCESes on
    the 0700 parent as the submission uid -> python3 exit 2 -> every suite
    ``INVALID fault=crashed`` -> reward 0 regardless of submission quality
    (F19, proven by a check run). chmod 0711 restores traversal while
    keeping the no-listing cross-run isolation intent: the run tree itself is
    chowned to the submission account anyway, sibling parents keep random
    mkdtemp suffixes that 0711 does not let other uids enumerate, and suites
    run sequentially with rmtree in ``finally`` so exposure stays bounded.
    Native EC2 rollouts never set the variable and keep mkdtemp's 0700
    byte-identical (bwrap bind-mounts the run dir at /work, so the parent
    never appears in any path the submission opens). The preflight parent is
    deliberately NOT created through this helper -- its probe's sealed-secret
    PermissionError depends on that parent staying 0700.
    """
    parent = Path(tempfile.mkdtemp(prefix=prefix))
    if os.environ.get("MANIFOLD_BENCH_SANDBOX") == "external":
        os.chmod(parent, 0o711)
    return parent


def prepare_submission(submission: Path, *, sandboxed: bool) -> PreparedSubmission:
    if not submission.is_dir() or submission.is_symlink():
        raise SubmissionError("submission path must be a regular directory")
    _regular_tree(submission)
    _scan_source(submission)
    _require_entrypoints(submission)
    parent = _submission_tree_parent("manifold-bench-built-")
    root = parent / "submission"
    shutil.copytree(submission, root, symlinks=False)
    build_seconds = 0.0
    build = root / "build.sh"
    if build.exists():
        build_seconds, _ = _run_command(["/bin/bash", "build.sh"], root, 180, sandboxed=sandboxed)
    _regular_tree(root)
    _scan_source(root)
    _require_entrypoints(root)
    return PreparedSubmission(root, build_seconds)


def _write_fresh(path: Path, write) -> None:
    """Write a grader-owned file into the run directory without following a link.

    ``unlink`` removes the name itself rather than the target, so a candidate
    symlink parked at this path is destroyed instead of followed. Without this,
    ``np.savez(root / "eval.npz", ...)`` truncated whatever host file the link
    pointed at and wrote an NPZ archive over it.
    """
    if path.is_symlink() or path.exists():
        path.unlink()
    write(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise SubmissionError(f"{path.name} is not a regular file after writing")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_prediction(prediction: Prediction, rows: np.ndarray) -> Prediction:
    return Prediction(
        reconstruction=prediction.reconstruction[rows],
        presence=prediction.presence[rows],
        contributions=prediction.contributions[rows],
        bias=prediction.bias,
    )


def _prediction_discrepancy(left: Prediction, right: Prediction, x: np.ndarray) -> float:
    """Scale-relative maximum discrepancy between two predictions of the same rows."""
    if left.n_features != right.n_features:
        return float("inf")
    scale = float(np.median(np.linalg.norm(x, axis=1))) + 1e-8
    return max(
        float(np.max(np.abs(left.reconstruction - right.reconstruction))) / scale,
        float(np.max(np.abs(left.contributions - right.contributions))) / scale,
        float(np.max(np.abs(left.presence - right.presence)))
        / (float(np.max(np.abs(left.presence))) + 1e-8),
        float(np.max(np.abs(left.bias - right.bias))) / scale,
    )


def run_suite(
    prepared: PreparedSubmission,
    dataset: Dataset,
    *,
    sandboxed: bool,
) -> RunResult:
    spec = dataset.spec
    parent = _submission_tree_parent(f"manifold-bench-{spec.name}-")
    root = parent / "run"
    try:
        shutil.copytree(prepared.root, root, symlinks=False)
        _write_fresh(root / "train.npz", lambda p: np.savez(p, x=dataset.train.x))
        _write_fresh(
            root / "config.json",
            lambda p: p.write_text(json.dumps(public_config(spec), sort_keys=True) + "\n"),
        )
        train_hash = _sha256(root / "train.npz")
        config_hash = _sha256(root / "config.json")
        fit_seconds, fit_log = _run_command(
            ["./fit"], root, spec.fit_timeout_seconds, sandboxed=sandboxed
        )
        if _sha256(root / "train.npz") != train_hash:
            raise SubmissionError("fit modified train.npz")
        if _sha256(root / "config.json") != config_hash:
            raise SubmissionError("fit modified config.json")

        config_hash_after_fit = config_hash

        def transform(
            rows: np.ndarray, timeout_seconds: int | None = None
        ) -> tuple[Prediction, float, str]:
            # Every deployment call starts from the same persisted fit artifact.
            # Reusing one candidate-writable directory let a transform remember
            # the standalone score rows, then recognize and replay them during
            # the permutation/context probes. That tests a mutable conversation
            # with the verifier, not deterministic out-of-sample inference.
            call_parent = _submission_tree_parent(f"manifold-call-{spec.name}-")
            call_root = call_parent / "run"
            try:
                shutil.copytree(root, call_root, symlinks=False)
                _write_fresh(call_root / "eval.npz", lambda p: np.savez(p, x=rows))
                eval_hash = _sha256(call_root / "eval.npz")
                prediction_path = call_root / "predictions.npz"
                prediction_path.unlink(missing_ok=True)
                seconds, log = _run_command(
                    ["./transform"],
                    call_root,
                    timeout_seconds or spec.transform_timeout_seconds,
                    sandboxed=sandboxed,
                )
                if _sha256(call_root / "eval.npz") != eval_hash:
                    raise SubmissionError("transform modified eval.npz")
                if _sha256(call_root / "config.json") != config_hash_after_fit:
                    raise SubmissionError("transform modified config.json")
                try:
                    prediction = load_prediction(prediction_path, rows, spec.max_features)
                except ValueError as error:
                    raise SubmissionError(str(error)) from error
                return prediction, seconds, log
            finally:
                shutil.rmtree(call_parent, ignore_errors=True)

        # Transform the matching and scoring partitions in *separate* calls so a
        # candidate cannot use the hidden scoring rows while producing matching
        # outputs (or vice versa).  A reusable, deployment-ready representation
        # must assign each observation a representation from persisted fit state,
        # independent of which other rows share the call.
        match_pred, match_seconds, match_log = transform(dataset.match.x)
        score_pred, score_seconds, score_log = transform(dataset.score.x)
        if match_pred.n_features != score_pred.n_features or not np.array_equal(
            match_pred.bias, score_pred.bias
        ):
            raise SubmissionError(
                "transform emitted an inconsistent feature dictionary across calls"
            )

        # Row-order determinism: permuting the scored rows must permute the
        # outputs the same way and nothing else.
        permutation = np.random.default_rng(spec.seed + 701).permutation(dataset.score.x.shape[0])
        inverse = np.argsort(permutation)
        permuted, _, permuted_log = transform(dataset.score.x[permutation])

        # Batch-context invariance: a scored row's representation must not depend
        # on unrelated rows in the same batch.  Re-transform the scored rows
        # embedded among the (unrelated) matching rows and require the scored
        # slice to reproduce the standalone result.  A transductive method that
        # adapts to batch composition fails this cheaply, while row-order noise
        # stays far below the tolerance.
        cut = dataset.match.x.shape[0]
        joined = np.concatenate([dataset.match.x, dataset.score.x], axis=0)
        # Hide scored rows at deterministic private positions among unrelated
        # matching rows. A visible suffix made it possible to branch on batch
        # length and special-case exactly the slice the verifier compared.
        context_order = np.random.default_rng(spec.seed + 907).permutation(len(joined))
        context = joined[context_order]
        score_positions = np.flatnonzero(context_order >= cut)
        score_indices = context_order[score_positions] - cut
        score_positions = score_positions[np.argsort(score_indices)]
        # The published per-transform limit is stated for an evaluation batch.  This
        # probe deliberately submits a larger batch, so hold it to the same rows
        # per second rather than to a silently stricter deadline.
        context_budget = max(
            spec.transform_timeout_seconds,
            math.ceil(
                spec.transform_timeout_seconds
                * len(context)
                / max(len(dataset.score.x), 1)
            ),
        )
        context_pred, _, context_log = transform(context, context_budget)
        context_error = _prediction_discrepancy(
            score_pred, _select_prediction(context_pred, score_positions), dataset.score.x
        )
        if context_error > BATCH_CONTEXT_TOLERANCE:
            raise SubmissionError(
                "transform output for a row depends on unrelated rows in the same batch "
                f"(batch-context discrepancy {context_error:.3e} exceeds "
                f"{BATCH_CONTEXT_TOLERANCE:.0e}); the representation must be reusable per row"
            )

        return RunResult(
            match=match_pred,
            score=score_pred,
            permuted=permuted,
            permutation_inverse=inverse,
            fit_seconds=fit_seconds,
            transform_seconds=max(match_seconds, score_seconds),
            log_tail=(fit_log + match_log + score_log + permuted_log + context_log)[
                -OUTPUT_TAIL_BYTES:
            ],
        )
    finally:
        shutil.rmtree(parent, ignore_errors=True)

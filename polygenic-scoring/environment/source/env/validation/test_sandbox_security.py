"""Regression tests for the two grader-integrity defects fixed in
grader/submission_runner.py:

  (A) build.sh ran BEFORE any sandbox, with the grader's inherited environment
      and full filesystem access -> a malicious build.sh could read the held-out
      corpus/*/truth/y_test.csv and plant it for `predict`.
  (B) the sandbox defaulted to mode 'auto' and returned an EMPTY prefix (NO
      isolation) whenever `bwrap` was unavailable -> fit/predict silently ran
      unconfined (fail-OPEN).

These tests are host-independent: they mock bwrap where needed. The runtime
enforcement of the bwrap mounts is
exercised at grade time by `preflight_sandbox`, which this suite tests via its
leak-detection logic. What we assert here:

  1. fail-closed: a missing sandbox raises SandboxUnavailable; there is no
     unconfined mode or policy selector.
  2. the build/fit confinement argv binds ONLY the intended dirs and never the
     corpus/truth/grader, with no network (--unshare-net).
  3. build runs in an ISOLATED, FROZEN copy (source untouched) with a SANITIZED
     env, so build.sh cannot `import grader`/`import datagen` nor mutate the
     source in place.
  4. preflight aborts scoring when the sandbox leaks (fs read or network).

Run:  python validation/test_sandbox_security.py   (also collectable by pytest)
"""
from __future__ import annotations
import inspect
import os
import shutil
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grader import submission_runner as sr
from grader.submission_runner import (
    SandboxUnavailable, resolve_sandbox, _bwrap_prefix,
    build_submission, preflight_sandbox,
)

FAKE_BWRAP = "/opt/fake/bin/bwrap"
FAKE_SETPRIV = "/opt/fake/bin/setpriv"


def _install_fake_identity():
    original = sr.sys_platform_linux, sr.os.geteuid, sr.pwd.getpwnam
    sr.sys_platform_linux = lambda: True
    sr.os.geteuid = lambda: 0
    sr.pwd.getpwnam = lambda _name: SimpleNamespace(
        pw_uid=os.getuid(), pw_gid=os.getgid(), pw_name=sr.SANDBOX_USER)
    return original


def _restore_fake_identity(original):
    sr.sys_platform_linux, sr.os.geteuid, sr.pwd.getpwnam = original


def _mk_submission(build_sh=None, extra=None):
    d = tempfile.mkdtemp(prefix="svpgs_sub_")
    for name in ("fit", "predict"):
        p = os.path.join(d, name)
        with open(p, "w") as fh:
            fh.write("#!/bin/sh\nexit 0\n")
        os.chmod(p, 0o755)
    if build_sh is not None:
        p = os.path.join(d, "build.sh")
        with open(p, "w") as fh:
            fh.write(build_sh)
        os.chmod(p, 0o755)
    for name, content in (extra or {}).items():
        with open(os.path.join(d, name), "w") as fh:
            fh.write(content)
    return d


# --- 1. fail-closed on missing sandbox (defect B) --------------------------

def test_fail_closed_when_sandbox_required_but_missing(monkeypatch=None):
    """The native bwrap path must raise when bwrap is absent."""
    orig_which = sr._which
    identity = _install_fake_identity()
    sr._which = lambda prog: None  # simulate bwrap absent
    try:
        raised = False
        try:
            with mock.patch.dict(os.environ, {"SVPGSBENCH_SANDBOX": "bwrap"}):
                resolve_sandbox()
        except SandboxUnavailable:
            raised = True
        assert raised, "native bwrap path must fail closed when bwrap is missing"
    finally:
        sr._which = orig_which
        _restore_fake_identity(identity)


def test_native_optin_uses_bwrap_when_present():
    orig_which = sr._which
    identity = _install_fake_identity()
    sr._which = lambda prog: {"bwrap": FAKE_BWRAP, "setpriv": FAKE_SETPRIV}.get(prog)
    try:
        with mock.patch.dict(os.environ, {"SVPGSBENCH_SANDBOX": "bwrap"}):
            assert resolve_sandbox() == (FAKE_BWRAP, FAKE_SETPRIV)
    finally:
        sr._which = orig_which
        _restore_fake_identity(identity)


# --- 1b. platform-owned setpriv jail is the DEFAULT ------------------------
# This is the harbor-release product branch: it must grade under stock harbor with
# NOTHING set, because the harbor task container cannot create the namespaces bwrap
# needs. So the bwrap self-sandbox is SKIPPED BY DEFAULT and the runner uses a
# setpriv-only uid jail, relying on the 0700 root seal for read-protection. Native
# EC2 rollouts opt back into the full bwrap jail with SVPGSBENCH_SANDBOX=bwrap.

def test_default_unset_is_external_setpriv_only():
    """With NOTHING set, the shipped default is the setpriv-only platform jail and
    bwrap being absent must NOT fail closed."""
    orig_which = sr._which
    identity = _install_fake_identity()
    sr._which = lambda prog: {"setpriv": FAKE_SETPRIV}.get(prog)  # bwrap absent
    saved = os.environ.pop("SVPGSBENCH_SANDBOX", None)
    try:
        assert sr._external_sandbox() is True, "default must be external"
        assert resolve_sandbox() == (None, FAKE_SETPRIV)
    finally:
        if saved is not None:
            os.environ["SVPGSBENCH_SANDBOX"] = saved
        sr._which = orig_which
        _restore_fake_identity(identity)


def test_external_mode_resolves_setpriv_only_without_bwrap():
    """SVPGSBENCH_SANDBOX=external (explicit) is setpriv-only, so bwrap being absent
    must NOT fail closed (it cannot create namespaces there anyway)."""
    orig_which = sr._which
    identity = _install_fake_identity()
    # bwrap deliberately absent; only setpriv present.
    sr._which = lambda prog: {"setpriv": FAKE_SETPRIV}.get(prog)
    try:
        with mock.patch.dict(os.environ, {"SVPGSBENCH_SANDBOX": "external"}):
            assert resolve_sandbox() == (None, FAKE_SETPRIV)
    finally:
        sr._which = orig_which
        _restore_fake_identity(identity)


def test_external_mode_argv_is_setpriv_uid_drop_no_bwrap():
    """External-mode argv drops to svpgsub via setpriv with no inheritable caps and
    no-new-privs, and invokes NO bwrap / mount namespace (the platform owns it)."""
    run_dir = tempfile.mkdtemp(prefix="svpgs_run_")
    sub_dir = tempfile.mkdtemp(prefix="svpgs_sub_")
    try:
        with mock.patch.dict(os.environ, {"SVPGSBENCH_SANDBOX": "external"}):
            argv = _bwrap_prefix(None, FAKE_SETPRIV, rw_dir=run_dir,
                                 submission=sub_dir)
        joined = " ".join(argv)
        assert argv[0] == FAKE_SETPRIV, "external jail must start at setpriv"
        assert "bwrap" not in joined and "--unshare-net" not in joined, \
            "external mode must not invoke bwrap / a mount namespace"
        assert f"--reuid {sr.SANDBOX_USER}" in joined
        assert f"--regid {sr.SANDBOX_USER}" in joined
        assert "--clear-groups" in argv
        assert "--inh-caps=-all" in argv and "--no-new-privs" in argv
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(sub_dir, ignore_errors=True)


# --- 2. confinement argv never exposes truth/grader; no network ------------

def test_build_and_phase_argv_exclude_truth_and_network():
    corpus_truth = "/some/corpus/svld_strong__s200/truth"
    grader_dir = "/some/repo/grader"
    build_dir = tempfile.mkdtemp(prefix="svpgs_build_")
    run_dir = tempfile.mkdtemp(prefix="svpgs_run_")
    sub_dir = tempfile.mkdtemp(prefix="svpgs_sub_")
    try:
        # native bwrap path (opt-in): the namespace jail is what these bindings test.
        with mock.patch.dict(os.environ, {"SVPGSBENCH_SANDBOX": "bwrap"}):
            # build phase: only the build dir is writable, nothing else bound
            build_argv = _bwrap_prefix(FAKE_BWRAP, FAKE_SETPRIV, rw_dir=build_dir)
            # fit/predict phase: run dir rw + submission ro
            phase_argv = _bwrap_prefix(FAKE_BWRAP, FAKE_SETPRIV, rw_dir=run_dir,
                                       submission=sub_dir)
        for argv, label in ((build_argv, "build"), (phase_argv, "phase")):
            joined = " ".join(argv)
            required_namespaces = {
                "--unshare-pid", "--unshare-net", "--unshare-ipc",
                "--unshare-uts", "--unshare-cgroup",
            }
            assert required_namespaces.issubset(argv), \
                f"{label}: mandatory namespaces missing"
            assert "--unshare-all" not in argv and "--unshare-user" not in argv, \
                f"{label}: user namespace would unmap the private-work owner"
            assert corpus_truth not in joined, f"{label}: truth must NOT be mounted"
            assert grader_dir not in joined, f"{label}: grader must NOT be mounted"
            assert "--ro-bind /etc /etc" not in joined
            assert "--ro-bind /opt /opt" not in joined
            assert f"--reuid {sr.SANDBOX_USER}" in joined
        # build must NOT bind the submission source read-only either (source is
        # copied INTO the build dir), so a build cannot reach anything but its dir
        assert build_dir in " ".join(build_argv)
        assert sub_dir not in " ".join(build_argv)
        # the fit/predict phase binds the (already-built, frozen) submission ro
        assert sub_dir in " ".join(phase_argv)
        assert run_dir in " ".join(phase_argv)
    finally:
        for d in (build_dir, run_dir, sub_dir):
            shutil.rmtree(d, ignore_errors=True)


def test_private_workdir_matches_retained_host_uid_namespace():
    """The 0700 work bind must remain reachable after the fixed-uid drop."""
    with tempfile.TemporaryDirectory(prefix="svpgs_owner_parent_") as parent:
        work = os.path.join(parent, "work")
        os.mkdir(work)
        expected = (os.getuid(), os.getgid())
        with mock.patch.object(sr, "_sandbox_identity", return_value=expected):
            sr._open_private_work(work)
        st = os.stat(work)
        assert (st.st_uid, st.st_gid) == expected
        assert stat.S_IMODE(st.st_mode) == 0o700

        with mock.patch.dict(os.environ, {"SVPGSBENCH_SANDBOX": "bwrap"}):
            argv = _bwrap_prefix(FAKE_BWRAP, FAKE_SETPRIV, rw_dir=work)
        assert "--unshare-user" not in argv and "--unshare-all" not in argv
        assert argv[argv.index("--bind") + 1:argv.index("--bind") + 3] == [work, "/work"]
        assert argv[argv.index("--chdir") + 1] == "/work"
        assert argv[argv.index("--reuid") + 1] == sr.SANDBOX_USER


def test_real_private_workdir_preflight_when_provisioned():
    """Exercise real bwrap/setpriv ownership on a provisioned Linux worker.

    Pinned to the native bwrap path: the default (platform-owned setpriv jail)
    relies on a 0700 seal this bare smoke test does not set up, so it is exercised
    by the packaged bake/grade path, not here."""
    if not sr.sys_platform_linux() or os.geteuid() != 0:
        return
    if sr._which("bwrap") is None or sr._which("setpriv") is None:
        return
    try:
        sr.pwd.getpwnam(sr.SANDBOX_USER)
    except KeyError:
        return
    with mock.patch.dict(os.environ, {"SVPGSBENCH_SANDBOX": "bwrap"}):
        preflight_sandbox(log=lambda *_args: None)


# --- 3. build runs in an isolated, frozen copy with a sanitized env --------

def _build_with_mocked_bwrap(sub):
    """Exercise staging/build logic while mocking only the unavailable kernel wrapper."""
    original = (sr.preflight_sandbox, sr.resolve_sandbox, sr._bwrap_prefix,
                sr._sandbox_identity, sr.os.chown, sr._preexec_limits)
    sr.preflight_sandbox = lambda log=print: None
    sr.resolve_sandbox = lambda: (FAKE_BWRAP, FAKE_SETPRIV)
    sr._bwrap_prefix = lambda *_args, **_kwargs: []
    sr._sandbox_identity = lambda: (os.getuid(), os.getgid())
    sr.os.chown = lambda *_args, **_kwargs: None
    sr._preexec_limits = lambda _seconds: os.setsid
    try:
        return build_submission(sub, log=lambda *_: None)
    finally:
        (sr.preflight_sandbox, sr.resolve_sandbox, sr._bwrap_prefix,
         sr._sandbox_identity, sr.os.chown, sr._preexec_limits) = original

def test_build_runs_in_isolated_frozen_copy_source_untouched():
    """Build must not run in-place or with the inherited grader environment."""
    marker = "planted_by_build"
    build_sh = (
        "#!/bin/sh\n"
        # prove build ran and can write to its own (copied) dir
        f"touch {marker}\n"
        # prove the env is sanitized: importing the grader/datagen must FAIL
        "python3 -c 'import datagen' 2>/dev/null && echo IMPORTED_DATAGEN > leak || true\n"
        # prove PYTHONPATH is not leaking the repo root
        "python3 -c 'import os,sys; sys.exit(0 if not os.environ.get(\"PYTHONPATH\") else 3)' || echo HAD_PYTHONPATH > leak\n"
    )
    sub = _mk_submission(build_sh=build_sh)
    # Seed the *source* with PYTHONPATH pointing at the repo, to prove build's
    # sanitized env drops it regardless of the grader's environment.
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ok, built, note = _build_with_mocked_bwrap(sub)
    try:
        assert ok, f"build should succeed: {note}"
        assert built is not None and built != sub, "build must use an isolated copy"
        # marker exists in the COPY, not in the source
        assert os.path.exists(os.path.join(built, marker)), "build artifact missing in copy"
        assert not os.path.exists(os.path.join(sub, marker)), \
            "source was mutated in place -- build ran unconfined against the source"
        # sanitized env: neither leak sentinel was produced
        assert not os.path.exists(os.path.join(built, "leak")), \
            "build env was not sanitized (imported datagen or saw PYTHONPATH)"
        # frozen read-only: the copy's files are not writable
        st = os.stat(os.path.join(built, "fit")).st_mode
        assert not (st & stat.S_IWUSR), "built artifact must be frozen read-only"
    finally:
        os.environ.pop("PYTHONPATH", None)
        sr._rmtree_ro(built) if built else None
        shutil.rmtree(sub, ignore_errors=True)


def test_build_rejects_symlinked_submission():
    sub = _mk_submission()
    # add a symlink pointing at a sensitive absolute path
    os.symlink("/etc/passwd", os.path.join(sub, "sneaky"))
    try:
        ok, built, note = _build_with_mocked_bwrap(sub)
        assert not ok and built is None, "symlinked submission must be rejected"
        assert "symlink" in note.lower()
    finally:
        shutil.rmtree(sub, ignore_errors=True)


def test_build_rejects_symlinked_submission_root():
    with tempfile.TemporaryDirectory(prefix="svpgs_root_") as root:
        real = _mk_submission()
        link = os.path.join(root, "submission")
        os.symlink(real, link, target_is_directory=True)
        try:
            ok, built, note = _build_with_mocked_bwrap(link)
            assert not ok and built is None
            assert "root" in note.lower() or "follow" in note.lower()
        finally:
            shutil.rmtree(real, ignore_errors=True)


def test_build_rejects_symlinked_submission_root_before_any_open():
    """The root must be rejected by an lstat, not merely by the kernel's
    O_NOFOLLOW: prove the rejection happens even where the open would succeed."""
    with tempfile.TemporaryDirectory(prefix="svpgs_root_") as root:
        real = _mk_submission()
        link = os.path.join(root, "submission")
        os.symlink(real, link, target_is_directory=True)
        try:
            raised = None
            try:
                sr._open_submission_root(link)
            except sr.SubmissionTreeError as e:
                raised = str(e)
            assert raised is not None, "a symlinked submission root was opened"
            assert "symlink" in raised.lower(), raised
        finally:
            shutil.rmtree(real, ignore_errors=True)


def test_snapshot_is_pinned_to_the_parent_directory_it_opened():
    """A symlink ABOVE the root is trusted verifier infrastructure (macOS `/var`
    is one), so it is resolved -- but the resolution must be pinned: replacing a
    parent component AFTER staging begins cannot redirect the walk, because every
    step runs relative to the retained parent/root fds."""
    with tempfile.TemporaryDirectory(prefix="svpgs_parent_") as base:
        real_parent = os.path.join(base, "real-parent")
        os.mkdir(real_parent)
        source = _mk_submission(extra={"payload": "original\n"})
        staged = os.path.join(real_parent, "submission")
        shutil.move(source, staged)

        link_parent = os.path.join(base, "parent")
        os.symlink(real_parent, link_parent, target_is_directory=True)
        through_symlinked_parent = os.path.join(link_parent, "submission")

        decoy_parent = os.path.join(base, "decoy-parent")
        os.mkdir(decoy_parent)
        decoy = _mk_submission(extra={"payload": "redirected\n"})
        shutil.move(decoy, os.path.join(decoy_parent, "submission"))

        real_fwalk = os.fwalk
        swapped = False

        def swap_parent_then_walk(*args, **kwargs):
            nonlocal swapped
            if not swapped:
                os.unlink(link_parent)
                os.symlink(decoy_parent, link_parent, target_is_directory=True)
                swapped = True
            yield from real_fwalk(*args, **kwargs)

        with mock.patch.object(sr.os, "fwalk", side_effect=swap_parent_then_walk):
            ok, built, note = _build_with_mocked_bwrap(through_symlinked_parent)
        try:
            assert ok, note
            with open(os.path.join(built, "payload")) as fh:
                assert fh.read() == "original\n", \
                    "a parent-component swap redirected the trusted snapshot"
        finally:
            sr._rmtree_ro(built) if built else None


def test_root_rename_swap_race_is_rejected():
    """Swap the root for a symlink in the window between the lstat and the open.
    The device+inode recheck (or openat2's RESOLVE_NO_SYMLINKS) must catch it."""
    with tempfile.TemporaryDirectory(prefix="svpgs_race_") as base:
        source = os.path.join(base, "submission")
        os.mkdir(source)
        decoy = os.path.join(base, "decoy")
        os.mkdir(decoy)
        with open(os.path.join(decoy, "payload"), "w") as fh:
            fh.write("redirected\n")

        real_stat = os.stat
        swapped = False

        def stat_then_swap(*args, **kwargs):
            nonlocal swapped
            result = real_stat(*args, **kwargs)
            if not swapped and args and args[0] == "submission":
                swapped = True
                os.rmdir(source)
                os.symlink(decoy, source, target_is_directory=True)
            return result

        with mock.patch.object(sr.os, "stat", side_effect=stat_then_swap):
            raised = None
            try:
                sr._open_submission_root(source)
            except sr.SubmissionTreeError as e:
                raised = str(e)
        assert swapped, "test did not trigger the rename/swap race"
        assert raised is not None, "a swapped submission root was accepted"
        assert "swap" in raised.lower() or "follow" in raised.lower(), raised


def test_root_swapped_for_another_real_directory_is_rejected():
    """The sharper race: the root is replaced between the lstat and the open by a
    real DIRECTORY, so O_NOFOLLOW happily opens it. Only the device+inode recheck
    against what was lstat'd can catch this."""
    with tempfile.TemporaryDirectory(prefix="svpgs_ino_") as base:
        source = os.path.join(base, "submission")
        os.mkdir(source)
        decoy = os.path.join(base, "decoy")
        os.mkdir(decoy)
        with open(os.path.join(decoy, "payload"), "w") as fh:
            fh.write("redirected\n")

        real_stat = os.stat
        swapped = False

        def stat_then_swap(*args, **kwargs):
            nonlocal swapped
            result = real_stat(*args, **kwargs)
            if not swapped and args and args[0] == "submission":
                swapped = True
                os.rename(source, os.path.join(base, "moved"))
                os.rename(decoy, source)
            return result

        with mock.patch.object(sr.os, "stat", side_effect=stat_then_swap):
            raised = None
            try:
                fd = sr._open_submission_root(source)
                os.close(fd)
            except sr.SubmissionTreeError as e:
                raised = str(e)
        assert swapped, "test did not trigger the inode swap"
        assert raised is not None, \
            "the root was swapped for a different real directory and was accepted"
        assert "swap" in raised.lower(), raised


def test_root_open_falls_back_when_openat2_is_unavailable():
    """openat2 is Linux 5.6+ and can be blocked by seccomp; production runs Linux
    while dev runs macOS, so BOTH paths must hold. Force the fallback and require
    the same guarantees: a real root opens, a symlinked root is refused."""
    with tempfile.TemporaryDirectory(prefix="svpgs_fallback_") as base:
        real = os.path.join(base, "submission")
        os.mkdir(real)
        link = os.path.join(base, "linked")
        os.symlink(real, link, target_is_directory=True)

        with mock.patch.object(sr, "_openat2_no_symlinks", return_value=None):
            fd = sr._open_submission_root(real)
            try:
                assert os.fstat(fd).st_ino == os.stat(real).st_ino
            finally:
                os.close(fd)

            raised = None
            try:
                sr._open_submission_root(link)
            except sr.SubmissionTreeError as e:
                raised = str(e)
            assert raised is not None and "symlink" in raised.lower(), raised


def test_openat2_is_requested_with_no_symlink_resolution_on_linux():
    """On Linux the kernel itself must enforce the no-symlink/no-escape resolution;
    assert we ask for exactly that rather than trusting our own bookkeeping."""
    if not sr.sys_platform_linux():
        return
    with tempfile.TemporaryDirectory(prefix="svpgs_openat2_") as base:
        real = os.path.join(base, "submission")
        os.mkdir(real)
        parent_fd = os.open(base, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fd = sr._openat2_no_symlinks(parent_fd, "submission")
            if fd is None:
                return  # kernel/seccomp has no openat2; the fallback test covers it
            try:
                assert os.fstat(fd).st_ino == os.stat(real).st_ino
            finally:
                os.close(fd)
            os.symlink(real, os.path.join(base, "linked"),
                       target_is_directory=True)
            raised = False
            try:
                leaked = sr._openat2_no_symlinks(parent_fd, "linked")
                if leaked is not None:
                    os.close(leaked)
            except sr.SubmissionTreeError:
                raised = True
            assert raised, "openat2 followed a symlinked root"
        finally:
            os.close(parent_fd)
    assert sr.RESOLVE_NO_SYMLINKS == 0x004 and sr.RESOLVE_BENEATH == 0x008


def test_root_swap_cannot_redirect_snapshot():
    """A replaced source pathname cannot redirect a walk rooted at an open fd."""
    with tempfile.TemporaryDirectory(prefix="svpgs_root_swap_") as root:
        source = os.path.join(root, "submission")
        os.mkdir(source)
        for name in ("fit", "predict"):
            path = os.path.join(source, name)
            with open(path, "w") as fh:
                fh.write("#!/bin/sh\nexit 0\n")
            os.chmod(path, 0o755)
        with open(os.path.join(source, "payload"), "w") as fh:
            fh.write("original\n")

        decoy = os.path.join(root, "decoy")
        os.mkdir(decoy)
        with open(os.path.join(decoy, "payload"), "w") as fh:
            fh.write("redirected\n")
        moved = os.path.join(root, "opened-submission")
        real_fwalk = os.fwalk
        swapped = False

        def swap_then_walk(*args, **kwargs):
            nonlocal swapped
            if not swapped:
                os.rename(source, moved)
                os.symlink(decoy, source, target_is_directory=True)
                swapped = True
            yield from real_fwalk(*args, **kwargs)

        with mock.patch.object(sr.os, "fwalk", side_effect=swap_then_walk):
            ok, built, note = _build_with_mocked_bwrap(source)
        try:
            assert ok, note
            with open(os.path.join(built, "payload")) as fh:
                assert fh.read() == "original\n"
        finally:
            sr._rmtree_ro(built) if built else None


def test_in_place_mutation_during_snapshot_is_rejected():
    """Reject mutation even when filesystem timestamps cannot expose it."""
    sub = _mk_submission(extra={"payload": "a" * (2 * 1024 * 1024)})
    payload = os.path.join(sub, "payload")
    payload_identity = (os.stat(payload).st_dev, os.stat(payload).st_ino)
    real_read = os.read
    mutated = False

    def read_then_mutate(fd, size):
        nonlocal mutated
        chunk = real_read(fd, size)
        opened = os.fstat(fd)
        if (chunk and not mutated
                and (opened.st_dev, opened.st_ino) == payload_identity):
            mutated = True
            with open(payload, "r+b", buffering=0) as writer:
                writer.write(b"b")
                os.fsync(writer.fileno())
        return chunk

    try:
        # Simulate a filesystem whose timestamp granularity is too coarse to
        # distinguish the in-place write. The byte-for-byte verification pass,
        # rather than mtime/ctime luck, must still reject the snapshot.
        def coarse_version(st):
            return st.st_dev, st.st_ino, st.st_size

        with mock.patch.object(sr.os, "read", side_effect=read_then_mutate), \
                mock.patch.object(sr, "_source_version", side_effect=coarse_version):
            ok, built, note = _build_with_mocked_bwrap(sub)
        assert mutated, "test did not trigger the in-place mutation"
        assert not ok and built is None, note
        assert "changed" in note.lower(), note
    finally:
        shutil.rmtree(sub, ignore_errors=True)


def test_build_rejects_fifo_without_blocking():
    """A FIFO is not source code and must be rejected before staging can read it."""
    sub = _mk_submission()
    os.mkfifo(os.path.join(sub, "blocking_fifo"))
    try:
        started = time.monotonic()
        ok, built, note = _build_with_mocked_bwrap(sub)
        elapsed = time.monotonic() - started
        assert not ok and built is None
        assert "regular file" in note.lower() or "fifo" in note.lower()
        assert elapsed < 2.0, "FIFO validation must fail before any blocking read"
    finally:
        shutil.rmtree(sub, ignore_errors=True)


def test_build_rejects_generated_symlink():
    sub = _mk_submission(build_sh="#!/bin/sh\nln -s /etc/passwd generated-link\n")
    try:
        ok, built, note = _build_with_mocked_bwrap(sub)
        assert not ok and built is None
        assert "unsafe build output" in note.lower()
    finally:
        shutil.rmtree(sub, ignore_errors=True)


def test_resource_limits_fail_closed():
    apply_limits = sr._preexec_limits(10)
    with mock.patch.object(sr.resource, "setrlimit", side_effect=OSError("denied")):
        try:
            apply_limits()
        except OSError:
            return
    raise AssertionError("resource-limit failure was swallowed")


def test_cpu_guard_converts_wall_time_to_four_vcpu_aggregate_time():
    assert sr.WORKER_VCPUS == 4
    assert sr._cpu_limit_seconds(3600) == 14_401
    assert sr._cpu_limit_seconds(72) == 289
    assert sr._cpu_limit_seconds(48) == 193


def test_submission_numeric_runtime_receives_all_advertised_vcpus():
    env = sr._submission_env(sr.WORKER_VCPUS)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert env[name] == "4"


def test_sandbox_spawn_failure_stays_trusted_error():
    with tempfile.TemporaryDirectory(prefix="svpgs_spawn_") as work, \
            mock.patch.object(sr.subprocess, "Popen", side_effect=OSError("denied")):
        try:
            sr._run(["/mandatory/bwrap"], work, 10, sr._submission_env(1))
        except SandboxUnavailable:
            return
    raise AssertionError("trusted sandbox spawn failure was scored as solver failure")


def test_stderr_flood_is_drained_and_bounded():
    """Large stderr must neither deadlock nor grow an unbounded temp file."""
    work = tempfile.mkdtemp(prefix="svpgs_stderr_")
    try:
        cmd = [sys.executable, "-c",
               "import os; os.write(2, b'x' * (2 * 1024 * 1024)); "
               "os.write(2, b'END_SENTINEL')"]
        original = sr._preexec_limits
        sr._preexec_limits = lambda _seconds: os.setsid
        try:
            rc, _, tail = sr._run(cmd, work, 30, sr._submission_env(1))
        finally:
            sr._preexec_limits = original
        assert rc == 0
        assert len(tail.encode("utf-8")) <= sr.MAX_STDERR_BYTES
        assert tail.endswith("END_SENTINEL")
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --- 4. preflight aborts scoring when the sandbox leaks ---------------------

def _fake_run_factory(stderr):
    """Return a stand-in for sr._run that yields a canned stderr tail, so the
    preflight leak-detection can be tested without a real bwrap."""
    def _fake_run(cmd, cwd, timeout, env):
        return 0, 0.01, stderr
    return _fake_run


def _mock_preflight(stderr):
    original = sr.resolve_sandbox, sr._run, sr._open_private_work
    sr.resolve_sandbox = lambda: (FAKE_BWRAP, FAKE_SETPRIV)
    sr._run = _fake_run_factory(stderr)
    sr._open_private_work = lambda _path: None
    return original


def test_preflight_aborts_on_fs_leak():
    original = _mock_preflight("LEAK_FS\nPREFLIGHT_DONE\n")
    try:
        raised = False
        try:
            preflight_sandbox(log=lambda *_: None)
        except SandboxUnavailable as e:
            raised = "readable" in str(e) or "exfiltrat" in str(e).lower()
        assert raised, "preflight must abort when a secret is readable in-sandbox"
    finally:
        sr.resolve_sandbox, sr._run, sr._open_private_work = original


def test_preflight_aborts_on_network_leak():
    # Native bwrap path only: it owns --unshare-net, so a reachable network proves
    # the jail broke. The default platform jail has no network namespace (the
    # platform owns egress) and deliberately does NOT assert no-network (decision 15).
    original = _mock_preflight("LEAK_NET\nPREFLIGHT_DONE\n")
    try:
        raised = False
        try:
            with mock.patch.dict(os.environ, {"SVPGSBENCH_SANDBOX": "bwrap"}):
                preflight_sandbox(log=lambda *_: None)
        except SandboxUnavailable as e:
            raised = "network" in str(e).lower()
        assert raised, "native bwrap preflight must abort when it reaches the network"
    finally:
        sr.resolve_sandbox, sr._run, sr._open_private_work = original


def test_default_preflight_ignores_network_leak():
    """Default (platform-owned) preflight does NOT fail on a reachable network:
    harbor verifiers legitimately have egress, and read-protection is the seal."""
    original = _mock_preflight("LEAK_NET\nPREFLIGHT_DONE\n")
    saved = os.environ.pop("SVPGSBENCH_SANDBOX", None)
    try:
        preflight_sandbox(log=lambda *_: None)  # nothing set == external == no raise
    finally:
        if saved is not None:
            os.environ["SVPGSBENCH_SANDBOX"] = saved
        sr.resolve_sandbox, sr._run, sr._open_private_work = original


def test_preflight_aborts_when_probe_did_not_run():
    original = _mock_preflight("bwrap: some error\n")
    try:
        raised = False
        try:
            preflight_sandbox(log=lambda *_: None)
        except SandboxUnavailable:
            raised = True
        assert raised, "preflight must abort when its probe never ran under bwrap"
    finally:
        sr.resolve_sandbox, sr._run, sr._open_private_work = original


def test_preflight_passes_when_sandbox_seals():
    original = _mock_preflight("PREFLIGHT_DONE\n")
    try:
        preflight_sandbox(log=lambda *_: None)  # no raise
    finally:
        sr.resolve_sandbox, sr._run, sr._open_private_work = original


# --- 5. there is no sandbox-relaxation API ---------------------------------

def test_no_sandbox_relaxation_api():
    from grader import grade
    assert not hasattr(sr, "SandboxPolicy")
    assert list(inspect.signature(resolve_sandbox).parameters) == []
    assert "sandbox" not in inspect.signature(grade.grade_corpus).parameters


def test_build_submission_aborts_without_bwrap_before_touching_submission():
    """Every build path must fail closed before touching untrusted source."""
    orig_which = sr._which
    sr._which = lambda prog: None  # simulate bwrap absent
    try:
        raised = False
        try:
            sr.build_submission("/nonexistent/sub", log=lambda *_: None)
        except SandboxUnavailable:
            raised = True
        assert raised, "submission build must fail closed when bwrap is missing"
    finally:
        sr._which = orig_which


def _all_tests():
    return [v for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]


def main():
    failures = []
    for t in _all_tests():
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, f"assertion: {e}"))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} FAILED")
        return 1
    print(f"ALL {len(_all_tests())} SANDBOX-SECURITY TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --- the corpus key must be UNREACHABLE from inside the sandbox -------------
#
# This is a RUNTIME test on purpose. The suite already asserts that the grader
# directory and corpus truth do not appear in the bwrap argv, but that is a
# static scan of a string: it proves what we did not ASK for, not what the
# kernel actually hands a submission. Those are two different authorities, and a
# gap between them is invisible to every argv-based test.
#
# The stake is total. This repo is grader-side and the corpus is HMAC-keyed: a
# submission that can read the key can forge anchors and the manifest, i.e.
# mint its own reward scale. "Unreachable by construction" is the claim; this
# executes it.
def _read_attempt_source(paths):
    """Probe: try to read each path, report the ones that opened.

    It writes the verdict to /work (the only writable bind) rather than stdout,
    because _run returns stderr only -- a probe whose result we cannot read would
    make this test pass by construction, which is the failure mode it exists to
    prevent.
    """
    return (
        "import sys\n"
        f"paths = {paths!r}\n"
        "readable = []\n"
        "for p in paths:\n"
        "    try:\n"
        "        with open(p, 'rb') as fh:\n"
        "            fh.read(1)\n"
        "        readable.append(p)\n"
        "    except OSError:\n"
        "        pass\n"
        "open('/work/probe_result.txt', 'w').write(chr(10).join(readable))\n"
    )


def test_corpus_key_is_unreachable_from_inside_the_sandbox():
    """A submission must not be able to read the corpus key. Executed, not argued.

    Measured 2026-07-15 on a provisioned AL2023 box, with a real 0600 key placed
    at the real runtime path. Confinement is DEFENCE IN DEPTH, and it is worth
    knowing that all three layers hold independently:

      1. namespace -- /run is never bound, so the path does not exist inside the
         sandbox at all (FileNotFoundError, not PermissionError);
      2. uid -- the phase runs as `svpgsub` via setpriv, and the key is 0600
         root-owned;
      3. host policy -- force-binding the key AND chmod 644 STILL yielded
         PermissionError for uid 1001.

    HONEST LIMITATION: because of (2) and (3) I could not construct a real leak
    on that host, so this test has been verified to PASS on a sealed sandbox and
    its probe is proven non-blind by the positive control, but it has never been
    observed to FAIL on a genuine escape. Treat it as a tripwire on layer (1) --
    the one a refactor can plausibly break, e.g. someone adding a /run or
    repo-root bind -- not as a proven detector of every escape.
    """
    if not sr.sys_platform_linux() or os.geteuid() != 0:
        return
    if sr._which("bwrap") is None or sr._which("setpriv") is None:
        return
    try:
        sr.pwd.getpwnam(sr.SANDBOX_USER)
    except KeyError:
        return
    if not os.path.exists("/opt/svpgs-venv/bin/python"):
        return

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    targets = [
        "/run/secrets/svpgsbench-corpus.key",                  # owner-only runtime key
        os.path.join(repo_root, "grader", "corpus-key.hex"),   # sealed source credential
        os.path.expanduser("~/.svpgsbench/corpus_shipping.key"),
        os.path.join(repo_root, "grader", "grade.py"),         # grader internals
        os.path.join(repo_root, "gold", "fit"),                # exact score-1 source
        os.path.join(repo_root, "gold", "pgs_core.py"),        # exact score-1 engine
    ]
    # Only assert against paths that actually exist on this host: a target that is
    # simply absent would "pass" for the wrong reason and hide a later regression.
    present = [p for p in targets if os.path.exists(p)]
    if not present:
        return
    # POSITIVE CONTROL. Without it, `readable == []` is not evidence of
    # confinement -- a probe that cannot open ANYTHING (wrong interpreter, broken
    # bind, silently swallowed OSError) produces exactly the same empty list as a
    # perfectly sealed sandbox. The control is a path the sandbox legitimately
    # DOES bind, so it must come back readable; if it does not, the probe is blind
    # and its verdict about the secrets means nothing. Measure detection as a
    # margin between "can see what it should" and "cannot see what it must not".
    control = "/opt/svpgs-venv/bin/python"

    run_dir = tempfile.mkdtemp(prefix="svpgs_keytest_")
    try:
        os.chmod(run_dir, 0o700)
        os.chown(run_dir, *sr._sandbox_identity())
        bwrap, setpriv = sr.resolve_sandbox()
        prefix = sr._bwrap_prefix(bwrap, setpriv, rw_dir=run_dir)
        rc, _elapsed, err = sr._run(
            prefix + ["/opt/svpgs-venv/bin/python", "-c",
                      _read_attempt_source(present + [control])],
            run_dir, 60, os.environ.copy(),
        )
        # A probe that did not run cannot testify. Without this the test would
        # pass whenever the sandbox broke -- the exact shape of false pass this
        # file is meant to catch.
        assert rc == 0, f"probe did not run inside the sandbox (rc={rc}): {err}"
        result_path = os.path.join(run_dir, "probe_result.txt")
        assert os.path.exists(result_path), "probe produced no verdict"
        with open(result_path) as fh:
            readable = [line for line in fh.read().splitlines() if line.strip()]
        assert control in readable, (
            "probe is BLIND: it could not read the positive control "
            f"{control!r}, so an empty secret list would prove nothing"
        )
        readable = [p for p in readable if p != control]
        assert readable == [], (
            "SANDBOX ESCAPE: a submission can read grader-side secrets "
            f"{readable!r}. The corpus is HMAC-keyed, so a readable key lets a "
            "submission forge anchors and mint its own reward scale."
        )
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

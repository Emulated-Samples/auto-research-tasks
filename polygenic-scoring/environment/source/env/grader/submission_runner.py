"""Run an agent submission (fit + predict) on one dataset, with timing + timeouts.

Contract (documented in the task instruction):
  <submission>/build.sh   optional, run once offline before grading (compile step)
  <submission>/fit        reads train files from CWD, writes model.out
  <submission>/predict    reads model.out + test files from CWD, writes pred.csv

Per dataset we create a fresh run dir, copy the dataset's public/ files into it,
run `fit` then `predict` there (CWD = run dir), and read pred.csv. Times fit and
predict separately. On timeout / non-zero exit / missing-or-malformed pred.csv,
returns status != "ok" and the caller scores the dataset at the validity floor.

Every untrusted phase runs in the mandatory minimal-mount Bubblewrap sandbox
with no network and no grader/truth mounts.
"""
from __future__ import annotations
import errno
import hashlib
import math
import os
import pwd
import resource
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time

import numpy as np

from grader.contract import BUILD_TIMEOUT_S, FIT_TIMEOUT_S, PREDICT_TIMEOUT_S
# Cap the stderr we retain per phase so a submission that floods its error stream
# cannot exhaust grader memory; we keep only the tail for diagnostics.
MAX_STDERR_BYTES = 4096

# Bounds for safely reading a submission's pred.csv. The corpus test splits are
# in the low thousands of rows (~4.8k) with a handful of columns (mean, optional
# sd, optional sample_id). These caps are generous multiples of that yet finite,
# so a hostile submission cannot exhaust grader memory with a giant or absurdly
# wide/tall output; the exact row count is still separately required to equal
# n_test_expected downstream.
MAX_PRED_BYTES = 64 * 1024 * 1024  # 64 MiB
MAX_PRED_ROWS = 200_000
MAX_PRED_COLS = 64

# The trusted grader copies the submission before running it. Bound that copy so
# an untrusted workspace cannot hang setup with a FIFO/socket or exhaust the
# verifier disk with an enormous/sparse artifact before the sandbox even starts.
MAX_SUBMISSION_FILES = 20_000
MAX_SUBMISSION_FILE_BYTES = 1024 * 1024 * 1024      # 1 GiB per artifact
MAX_SUBMISSION_TOTAL_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB total
MAX_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_ADDRESS_SPACE_BYTES = 6 * 1024 * 1024 * 1024
MAX_PROCESSES = 256
MAX_OPEN_FILES = 1024
SANDBOX_USER = "svpgsub"
# Hyperfocal and the monolithic task both pin this environment to four vCPUs.
# RLIMIT_CPU is aggregate thread CPU time for one process, while the phase
# deadlines below are wall-clock limits. Give a valid fully parallel process the
# same wall interval before this defense-in-depth guard can fire.
WORKER_VCPUS = 4


class PredReadError(Exception):
    """A submission output could not be safely read: it was not a regular file
    (symlink / FIFO / socket / device) or it exceeded a hard size/shape cap.
    Distinct from a merely malformed CSV (which yields None and therefore the
    validity-floor reward); this is a rejection of an unsafe or abusive output
    path."""


class SubmissionTreeError(Exception):
    """The submission tree is unsafe or too large to stage."""


def _is_regular_file(path):
    """True iff `path` is a regular file, checked WITHOUT following a final
    symlink (os.lstat). Rejects a symlink / FIFO / socket / device planted by the
    submission where the grader expects a real file."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode)


def _open_regular_ro(path, max_bytes):
    """Open `path` read-only for text streaming, but only if it is a regular file
    within `max_bytes`. lstat first (reject non-regular: symlink/FIFO/socket/
    device) and open O_NOFOLLOW so the final component cannot be a symlink that
    redirects the read out to truth/grader files. Returns a text file object, or
    None if the path is absent. Raises PredReadError for an unsafe/oversized
    path."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as e:
        raise PredReadError(f"cannot stat {os.path.basename(path)}: {e}")
    if not stat.S_ISREG(st.st_mode):
        raise PredReadError(f"{os.path.basename(path)} is not a regular file")
    if st.st_size > max_bytes:
        raise PredReadError(
            f"{os.path.basename(path)} too large ({st.st_size} > {max_bytes} bytes)")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    return os.fdopen(fd, "r", encoding="utf-8", errors="replace")


def _safe_create_dest(dst):
    """Open `dst` for writing so it CANNOT follow a symlink the submission may have
    pre-created to redirect the write outside the run dir. Any existing entry at
    `dst` (regular file, or a fit-planted symlink/dir) is removed first, then the
    file is created with O_CREAT|O_EXCL|O_NOFOLLOW. Returns a binary file object.
    The parent is always the grader-created run dir and the name has no path
    separator, so only the final component needs guarding; fit is not running
    while staging happens, so there is no concurrent-creation race."""
    try:
        os.unlink(dst)
    except FileNotFoundError:
        pass
    except IsADirectoryError:
        shutil.rmtree(dst, ignore_errors=True)
    except OSError:
        # some platforms raise PermissionError (not IsADirectoryError) for
        # unlink() of a directory
        if os.path.isdir(dst) and not os.path.islink(dst):
            shutil.rmtree(dst, ignore_errors=True)
        else:
            raise
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    return os.fdopen(fd, "wb")

# Phase contract: `fit` sees only metadata + the training split; the test
# features are staged in only after `fit` exits and before `predict` runs, so a
# solver cannot inspect or tune against the test feature distribution during fit.
FIT_FILES = [
    "family.txt", "formula.txt", "dgp.json", "variant_metadata.tsv",
    "covariates_train.csv", "genotypes_train.tsv",
]
TEST_FILES = [
    "covariates_test.csv", "genotypes_test.tsv",
]
PUBLIC_FILES = FIT_FILES + TEST_FILES


class SandboxUnavailable(RuntimeError):
    """Required isolation is unavailable or failed its active preflight."""


def _external_sandbox():
    """True when the PLATFORM owns the isolation boundary. THIS IS THE DEFAULT.

    This is the harbor-release product branch: it must grade under stock harbor
    with ZERO configuration. Harbor runs this verifier as root inside a per-task
    container that already jails the agent, but that container is NOT granted
    CAP_SYS_ADMIN, so bwrap cannot create the mount/pid/net namespaces it needs and
    every bwrap invocation dies with EPERM ("Creating new namespace failed:
    Operation not permitted"). Because ``preflight_sandbox()`` runs its probe UNDER
    bwrap, that fired at BAKE (the deployment preflight) as well as at grade time,
    so the stock bwrap runner could not even produce an image. Therefore the
    bwrap self-sandbox is SKIPPED BY DEFAULT and the runner uses a setpriv-only uid
    jail instead.

    This is safe because the read-protection that actually matters here does NOT
    come from bwrap's minimal mounts: the corpus / datagen / grader / truth are
    root-sealed 0700 by the environment's sealPrivateState (index.ts, applied
    BEFORE this preflight) and asserted unreadable AS the svpgsub uid before
    grading; this runner additionally root-seals the repo .git (``_seal_repo_git``).
    The submission runs as svpgsub via setpriv (--clear-groups, no inheritable
    caps, no-new-privs), so it still cannot read the held-out labels or reach the
    corpus generator. A reachable network is not evidence of a broken jail here
    (harbor verifiers legitimately have egress), so the preflight asserts
    sealed-state unreadability rather than no-network.

    Native EC2 rollouts, where bwrap CAN create namespaces, opt back into the full
    bwrap jail by setting SVPGSBENCH_SANDBOX=bwrap. Nothing set == the shipped
    default == platform-owned setpriv jail. (Mirrors the manifold external-sandbox
    precedent; decision 15.)
    """
    return os.environ.get("SVPGSBENCH_SANDBOX", "external").strip().lower() != "bwrap"


def _seal_repo_git():
    """External mode only: root-seal the repo's .git (0700) before the probe.

    The setpriv-only jail keeps no mount namespace, so .git (whose full history
    includes the corpus generator and can reconstruct held-out truth) is no longer
    hidden by omission the way bwrap's minimal mounts hid it. Root still reads it;
    the svpgsub probe must not. Idempotent, and a missing .git (pin-only images
    ship no history) is fine. Only meaningful as root."""
    if os.geteuid() != 0:
        return
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    git_dir = os.path.join(repo_root, ".git")
    try:
        st = os.lstat(git_dir)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(st.st_mode) and not stat.S_ISLNK(st.st_mode):
        os.chmod(git_dir, 0o700)


def resolve_sandbox():
    """Return mandatory (Bubblewrap, setpriv) tools and validate the fixed jail uid.

    In external mode the platform owns the namespace jail; only the setpriv uid
    drop is ours, so bwrap is not required (and is deliberately not invoked -- it
    cannot create namespaces there). ``bwrap`` is returned as None in that mode.
    """
    if not sys_platform_linux() or os.geteuid() != 0:
        raise SandboxUnavailable("sandbox requires a Linux root verifier")
    setpriv = _which("setpriv")
    try:
        pwd.getpwnam(SANDBOX_USER)
    except KeyError as e:
        raise SandboxUnavailable(f"required sandbox user {SANDBOX_USER!r} is absent") from e
    if setpriv is None:
        raise SandboxUnavailable("required setpriv sandbox tool is unavailable")
    if _external_sandbox():
        return None, setpriv
    bwrap = _which("bwrap")
    if bwrap is None:
        raise SandboxUnavailable("required bwrap/setpriv sandbox tools are unavailable")
    return bwrap, setpriv


def sys_platform_linux():
    return os.uname().sysname == "Linux" if hasattr(os, "uname") else False


def _sandbox_identity():
    account = pwd.getpwnam(SANDBOX_USER)
    return account.pw_uid, account.pw_gid


def _open_private_work(path):
    uid, gid = _sandbox_identity()
    os.chown(path, uid, gid, follow_symlinks=False)
    os.chmod(path, 0o700, follow_symlinks=False)


def _bwrap_prefix(bwrap, setpriv, rw_dir, submission=None):
    """Argv prefix confining a phase to `rw_dir` (writable), each path in
    `ro_binds` (read-only), and a minimal read-only system, with NO network
    (`--unshare-net`). PID, IPC, UTS, cgroup, and network namespaces are new;
    the host user namespace is deliberately retained so the fixed host
    ``svpgsub`` uid can traverse its owner-only work bind after ``setpriv``.
    The corpus truth, grader, and datagen tree are never bound in.

    HYPERFOCAL PATCH(sandbox-external): when SVPGSBENCH_SANDBOX=external the
    platform owns the namespace jail (see ``_external_sandbox``), so this returns a
    setpriv-only uid jail with NO bwrap and NO mount namespace. The phase then runs
    in the real ``rw_dir`` (via the Popen cwd in ``_run``) and, in
    ``run_on_dataset``, execs the submission from its real path rather than the
    ``/submission`` bind. Read-protection of the corpus/truth comes from the 0700
    root seal (sealPrivateState + ``_seal_repo_git``), not from these minimal
    mounts. Native EC2 rollouts never set the variable and take the bwrap path
    below."""
    if _external_sandbox():
        return [setpriv, "--reuid", SANDBOX_USER, "--regid", SANDBOX_USER,
                "--clear-groups", "--inh-caps=-all", "--no-new-privs"]
    ro = []
    for d in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/opt/svpgs-venv"):
        if os.path.exists(d):
            ro += ["--ro-bind", d, d]
    args = [bwrap, "--unshare-pid", "--unshare-net", "--unshare-ipc",
            "--unshare-uts", "--unshare-cgroup",
            "--die-with-parent", "--new-session",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--dir", "/opt", "--dir", "/etc", *ro]
    for source in ("/etc/passwd", "/etc/group", "/etc/ld.so.cache"):
        if os.path.exists(source):
            args += ["--ro-bind", source, source]
    # /etc/alternatives holds the distro linker/toolchain symlinks: on Amazon
    # Linux /usr/bin/ld -> /etc/alternatives/ld -> /usr/bin/ld.bfd. With /etc
    # freshly --dir'd (empty), omitting it makes g++/gcc/gfortran LINKING fail
    # ("cannot find 'ld'") -> build_failed -> a genuine compiled-language
    # submission (svpgs allows any toolchain) silently scores 0. Bind it read-only
    # (symlinks into the already-mounted /usr, so isolation is unchanged).
    if os.path.isdir("/etc/alternatives"):
        args += ["--ro-bind", "/etc/alternatives", "/etc/alternatives"]
    args += ["--bind", rw_dir, "/work"]
    if submission is not None:
        args += ["--ro-bind", submission, "/submission"]
    args += ["--chdir", "/work", "--", setpriv, "--reuid", SANDBOX_USER,
             "--regid", SANDBOX_USER, "--clear-groups"]
    return args


def preflight_sandbox(log=print):
    """Before scoring, VERIFY the sandbox actually confines: as the sandboxed
    identity, attempt to (a) read a secret sentinel placed OUTSIDE every bound
    directory and (b) open a network connection. Abort scoring unless BOTH fail.

    In external mode (SVPGSBENCH_SANDBOX=external) the probe still runs as svpgsub
    and still fails closed if sealed host state is readable or the uid drop failed
    (LEAK_FS / LEAK_UID) -- so an absent platform seal is caught loudly. It does
    NOT treat a reachable network as a failure there: harbor verifiers legitimately
    have egress (decision 15) and the platform owns the no-network policy, so
    reachability is not evidence of a broken jail. The repo .git is root-sealed
    first so the setpriv-only jail cannot read it."""
    if _external_sandbox():
        _seal_repo_git()
    bwrap, setpriv = resolve_sandbox()
    secret_dir = tempfile.mkdtemp(prefix="svpgs_secret_")
    work = tempfile.mkdtemp(prefix="svpgs_preflight_")
    token = f"SVPGS_SENTINEL_{os.getpid()}_{time.time_ns()}"
    secret_path = os.path.join(secret_dir, "held_out_truth")
    with open(secret_path, "w") as fh:
        fh.write(token)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_root = os.path.join(repo_root, "corpus")
    truth_paths = []
    for dirpath, dirnames, filenames in os.walk(corpus_root):
        if os.path.basename(dirpath) == "truth":
            truth_paths.append(dirpath)
            break
    protected = [secret_path, "/root", os.path.join(repo_root, ".git"), *truth_paths]
    # This is trusted isolation preflight rather than solver execution, so one
    # numeric thread keeps the probe deterministic without narrowing the solver's
    # advertised four-vCPU allowance.
    env = _submission_env(1)
    probe = (
        "import os,pwd,socket,sys\n"
        f"paths={protected!r}\n"
        f"if os.getuid()==0 or pwd.getpwuid(os.getuid()).pw_name!={SANDBOX_USER!r}: sys.stderr.write('LEAK_UID\\n')\n"
        "for path in paths:\n"
        " try:\n"
        "  os.listdir(path) if os.path.isdir(path) else open(path,'rb').read(1)\n"
        "  sys.stderr.write('LEAK_FS:'+path+'\\n')\n"
        " except OSError: pass\n"
        "try:\n    s=socket.socket(); s.settimeout(3); s.connect(('1.1.1.1',53))\n"
        "    sys.stderr.write('LEAK_NET\\n'); s.close()\n"
        "except OSError:\n    pass\n"
        "sys.stderr.write('PREFLIGHT_DONE\\n')\n"
    )
    try:
        _open_private_work(work)
        prefix = _bwrap_prefix(bwrap, setpriv, rw_dir=work)
        rc, _, err = _run(prefix + ["/opt/svpgs-venv/bin/python", "-c", probe],
                          work, 60, env)
    finally:
        shutil.rmtree(secret_dir, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)
    if "PREFLIGHT_DONE" not in err:
        raise SandboxUnavailable(
            f"sandbox preflight probe did not run under bwrap (rc={rc}); refusing "
            f"to grade without a verified sandbox. stderr: {err[-400:]}")
    if "LEAK_FS" in err or "LEAK_UID" in err:
        raise SandboxUnavailable(
            "sandbox preflight FAILED: protected state was readable or uid drop failed; aborting.")
    if "LEAK_NET" in err and not _external_sandbox():
        raise SandboxUnavailable(
            "sandbox preflight FAILED: the sandbox reached the network; aborting.")
    if _external_sandbox():
        log("[preflight] bwrap self-sandbox SKIPPED (default; platform owns "
            "isolation) — setpriv uid jail verified: sealed state unreadable + uid "
            "drop confirmed. Set SVPGSBENCH_SANDBOX=bwrap for the native bwrap jail.")
    else:
        log("[preflight] sandbox verified: secret unreadable + network unreachable")


def _freeze_readonly(root):
    """chmod the built submission tree read-only (dirs r-x, files r--, keeping
    the exec bit) so a later phase cannot mutate the built artifact and the
    build's output is a frozen input to fit/predict."""
    _validate_submission_tree(root)
    for dirpath, dirnames, filenames in os.walk(root, topdown=False,
                                                followlinks=False):
        for name in filenames:
            p = os.path.join(dirpath, name)
            cur = os.lstat(p).st_mode
            os.chown(p, 0, 0, follow_symlinks=False)
            os.chmod(p, 0o555 if cur & 0o111 else 0o444, follow_symlinks=False)
        for name in dirnames:
            p = os.path.join(dirpath, name)
            os.chown(p, 0, 0, follow_symlinks=False)
            os.chmod(p, 0o555, follow_symlinks=False)
    os.chown(root, 0, 0, follow_symlinks=False)
    os.chmod(root, 0o555, follow_symlinks=False)


def _prepare_build_tree(root):
    """Validate, normalize, and make the private build copy writable by svpgsub."""
    _validate_submission_tree(root)
    uid, gid = _sandbox_identity()
    for dirpath, dirnames, filenames in os.walk(root, topdown=False,
                                                followlinks=False):
        for name in filenames:
            p = os.path.join(dirpath, name)
            mode = os.lstat(p).st_mode
            os.chown(p, uid, gid, follow_symlinks=False)
            os.chmod(p, 0o700 if mode & 0o111 else 0o600,
                     follow_symlinks=False)
        for name in dirnames:
            p = os.path.join(dirpath, name)
            os.chown(p, uid, gid, follow_symlinks=False)
            os.chmod(p, 0o700, follow_symlinks=False)
    os.chown(root, uid, gid, follow_symlinks=False)
    os.chmod(root, 0o700, follow_symlinks=False)


def _rmtree_ro(path):
    """rmtree that first restores write bits, so a read-only frozen build tree
    can still be cleaned up."""
    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
            func(p)
        except OSError:
            pass
    shutil.rmtree(path, onerror=_onerror)


RESOLVE_NO_SYMLINKS = 0x004
RESOLVE_BENEATH = 0x008


def _openat2_no_symlinks(dirfd, name):
    """openat2(2) with RESOLVE_NO_SYMLINKS|RESOLVE_BENEATH, or None where the
    syscall is unavailable (pre-5.6 Linux, macOS, a seccomp filter). The kernel
    then refuses the open outright if `name` is a symlink or escapes `dirfd`,
    so the guarantee does not rest on our own lstat/fstat bookkeeping."""
    if not sys_platform_linux():
        return None
    import ctypes

    class _OpenHow(ctypes.Structure):
        _fields_ = [("flags", ctypes.c_uint64),
                    ("mode", ctypes.c_uint64),
                    ("resolve", ctypes.c_uint64)]

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syscall = libc.syscall
    except (OSError, AttributeError):
        return None
    syscall.restype = ctypes.c_long
    how = _OpenHow(
        flags=os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        mode=0,
        resolve=RESOLVE_NO_SYMLINKS | RESOLVE_BENEATH)
    SYS_OPENAT2 = 437
    ctypes.set_errno(0)
    fd = syscall(ctypes.c_long(SYS_OPENAT2), ctypes.c_int(dirfd),
                 ctypes.c_char_p(os.fsencode(name)), ctypes.byref(how),
                 ctypes.c_size_t(ctypes.sizeof(how)))
    if fd >= 0:
        return int(fd)
    err = ctypes.get_errno()
    if err in (errno.ENOSYS, errno.EPERM, errno.EINVAL, errno.E2BIG):
        return None  # syscall absent/blocked: fall back to the openat path
    raise SubmissionTreeError(
        f"cannot open submission root without following links: "
        f"{os.strerror(err)}")


def _open_submission_root(source):
    """Open the submission root directory, never traversing a symlink at the
    final (solver-reachable) component.

    The parent directory is opened FIRST and every subsequent step is relative to
    that retained fd, so a rename/swap of the root -- or of any component above it
    -- after we hold the fd cannot redirect the staging. The basename is then
    lstat'd through the parent fd, opened no-follow (openat2 with
    RESOLVE_NO_SYMLINKS|RESOLVE_BENEATH where the kernel offers it), and the
    opened fd's device+inode must still equal what was lstat'd -- defeating a
    swap in the window between the two.

    Only the final component is required to be a real directory: components
    ABOVE the root are trusted verifier infrastructure (and are legitimately
    symlinked on some hosts, e.g. macOS `/var`), while everything at or below it
    is solver-controlled and is what the no-follow chain guards."""
    source = os.path.abspath(source)
    parent, name = os.path.split(source)
    if not name or name in (".", ".."):
        raise SubmissionTreeError(f"invalid submission root path {source!r}")
    try:
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as e:
        raise SubmissionTreeError(
            f"cannot open the submission's parent directory: {e}") from e
    try:
        try:
            before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as e:
            raise SubmissionTreeError(f"cannot inspect submission root: {e}") from e
        if stat.S_ISLNK(before.st_mode):
            raise SubmissionTreeError(
                "submission root is a symlink; refusing to follow it")
        if not stat.S_ISDIR(before.st_mode):
            raise SubmissionTreeError("submission root is not a regular directory")
        root_fd = _openat2_no_symlinks(parent_fd, name)
        if root_fd is None:
            try:
                root_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent_fd)
            except OSError as e:
                raise SubmissionTreeError(
                    f"cannot open submission root without following links: {e}") from e
    finally:
        os.close(parent_fd)
    try:
        opened = os.fstat(root_fd)
        if not stat.S_ISDIR(opened.st_mode):
            raise SubmissionTreeError("submission root is not a regular directory")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise SubmissionTreeError(
                "submission root was swapped while it was opened")
    except Exception:
        os.close(root_fd)
        raise
    return root_fd


def _source_version(st):
    """Fields that must remain stable while one source file is copied."""
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _source_stat(name, dirfd, rel):
    try:
        return os.stat(name, dir_fd=dirfd, follow_symlinks=False)
    except OSError as e:
        raise SubmissionTreeError(
            f"submission entry {rel!r} changed while it was staged: {e}") from e


def _snapshot_submission(source):
    """Make one bounded private snapshot through retained directory fds.

    The live solver tree is opened once without following its root
    (``_open_submission_root``), walked from that retained fd, and every regular
    file is opened with ``O_NOFOLLOW``. Its identity/version is proved both
    immediately after open and after the final byte is copied. A solver therefore
    cannot redirect trusted staging by swapping the root, a checked pathname, or a
    symlink between validation and copy. Source races are submission rejections;
    destination I/O failures stay trusted grader errors.
    """
    root_fd = _open_submission_root(source)
    try:
        built = tempfile.mkdtemp(prefix="svpgs_build_")
        n_files = 0
        total_bytes = 0
        try:
            walker = os.fwalk(".", topdown=True, follow_symlinks=False,
                              dir_fd=root_fd)
            while True:
                try:
                    dirpath, dirnames, filenames, dirfd = next(walker)
                except StopIteration:
                    break
                except OSError as e:
                    raise SubmissionTreeError(
                        f"submission tree changed while it was walked: {e}") from e

                rel_dir = os.path.relpath(dirpath, ".")
                if rel_dir == ".":
                    rel_dir = ""
                destination_dir = os.path.join(built, rel_dir)
                os.makedirs(destination_dir, mode=0o700, exist_ok=True)

                for name in list(dirnames):
                    rel = os.path.join(rel_dir, name) if rel_dir else name
                    entry = _source_stat(name, dirfd, rel)
                    if not stat.S_ISDIR(entry.st_mode):
                        raise SubmissionTreeError(
                            f"submission entry {rel!r} is not a directory")
                    os.makedirs(os.path.join(destination_dir, name),
                                mode=0o700, exist_ok=True)

                for name in filenames:
                    rel = os.path.join(rel_dir, name) if rel_dir else name
                    before = _source_stat(name, dirfd, rel)
                    if not stat.S_ISREG(before.st_mode):
                        raise SubmissionTreeError(
                            f"submission entry {rel!r} is not a regular file "
                            "(symlinks/FIFOs/sockets forbidden)")
                    if before.st_size > MAX_SUBMISSION_FILE_BYTES:
                        raise SubmissionTreeError(
                            f"submission file {rel!r} is too large "
                            f"({before.st_size} bytes)")
                    n_files += 1
                    total_bytes += before.st_size
                    if n_files > MAX_SUBMISSION_FILES:
                        raise SubmissionTreeError(
                            f"submission has too many files "
                            f"({n_files} > {MAX_SUBMISSION_FILES})")
                    if total_bytes > MAX_SUBMISSION_TOTAL_BYTES:
                        raise SubmissionTreeError(
                            f"submission is too large "
                            f"({total_bytes} > {MAX_SUBMISSION_TOTAL_BYTES} bytes)")

                    try:
                        source_fd = os.open(
                            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=dirfd)
                    except OSError as e:
                        raise SubmissionTreeError(
                            f"submission entry {rel!r} changed while it was opened: {e}") from e
                    try:
                        opened = os.fstat(source_fd)
                        if (not stat.S_ISREG(opened.st_mode)
                                or _source_version(opened) != _source_version(before)):
                            raise SubmissionTreeError(
                                f"submission entry {rel!r} changed while it was staged")
                        destination = os.path.join(destination_dir, name)
                        destination_fd = os.open(
                            destination,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL
                            | os.O_NOFOLLOW | os.O_CLOEXEC,
                            0o600)
                        copied = 0
                        copied_digest = hashlib.sha256()
                        try:
                            while True:
                                try:
                                    chunk = os.read(source_fd, 1024 * 1024)
                                except OSError as e:
                                    raise SubmissionTreeError(
                                        f"submission entry {rel!r} changed while it was read: {e}") from e
                                if not chunk:
                                    break
                                copied += len(chunk)
                                if copied > opened.st_size:
                                    raise SubmissionTreeError(
                                        f"submission entry {rel!r} grew while it was staged")
                                copied_digest.update(chunk)
                                view = memoryview(chunk)
                                while view:
                                    written = os.write(destination_fd, view)
                                    if written <= 0:
                                        raise OSError(
                                            f"short write while staging {rel!r}")
                                    view = view[written:]
                        finally:
                            os.close(destination_fd)

                        # A same-size in-place write can be invisible to an mtime/ctime
                        # comparison on filesystems with coarse or deferred timestamps.
                        # Re-read the retained source fd and require its bytes to match the
                        # exact stream copied above. This makes the stable-snapshot proof
                        # independent of timestamp granularity.
                        try:
                            os.lseek(source_fd, 0, os.SEEK_SET)
                        except OSError as e:
                            raise SubmissionTreeError(
                                f"submission entry {rel!r} changed while it was verified: {e}") from e
                        verified = 0
                        verified_digest = hashlib.sha256()
                        while True:
                            try:
                                chunk = os.read(source_fd, 1024 * 1024)
                            except OSError as e:
                                raise SubmissionTreeError(
                                    f"submission entry {rel!r} changed while it was verified: {e}") from e
                            if not chunk:
                                break
                            verified += len(chunk)
                            if verified > opened.st_size:
                                raise SubmissionTreeError(
                                    f"submission entry {rel!r} grew while it was verified")
                            verified_digest.update(chunk)
                        finished = os.fstat(source_fd)
                        if (copied != opened.st_size
                                or verified != opened.st_size
                                or verified_digest.digest() != copied_digest.digest()
                                or _source_version(finished) != _source_version(opened)):
                            raise SubmissionTreeError(
                                f"submission entry {rel!r} changed while it was copied")
                        os.chmod(destination,
                                 0o700 if before.st_mode & 0o111 else 0o600)
                    finally:
                        os.close(source_fd)
            return built
        except Exception:
            _rmtree_ro(built)
            raise
    finally:
        os.close(root_fd)


def build_submission(submission_dir, log=print):
    """Compile the submission in an ISOLATED, FROZEN copy, under the same-or-
    tighter confinement as fit/predict, and verify the sandbox first.

    Returns (ok, built_dir, status): `built_dir` is a fresh copy of the
    submission (with any build artifacts) to be used as the source for
    fit/predict, frozen read-only; use it in place of `submission_dir`. On a
    rejected submission `built_dir` is None. Callers must `_rmtree_ro(built_dir)`
    when done.

    Self-enforcing: `preflight_sandbox()` runs here, so the secret-read, network
    probe, and fail-closed enforcement hold regardless of the caller.

    Isolation: the source is copied into a private build dir; `build.sh` runs
    with a sanitized env (no repo PYTHONPATH -> no `import grader`/`import
    datagen`) and, under `bwrap`, with NO network and NO corpus/truth/grader
    mounts -- only the build dir is visible -- so build.sh cannot read or
    exfiltrate the held-out labels."""
    # Verify + enforce the sandbox here so protection holds no matter how we are
    # called; on a required-but-unavailable or leaky sandbox this raises
    # SandboxUnavailable and the grade aborts before any submission code runs.
    preflight_sandbox(log=log)
    bwrap, setpriv = resolve_sandbox()
    try:
        built = _snapshot_submission(submission_dir)
    except SubmissionTreeError as e:
        return False, None, str(e)
    keep_built = False
    try:
        try:
            _prepare_build_tree(built)
        except SubmissionTreeError as e:
            return False, None, f"unsafe staged submission: {e}"
        bs = os.path.join(built, "build.sh")
        if os.path.exists(bs):
            env = _submission_env(WORKER_VCPUS)
            prefix = _bwrap_prefix(bwrap, setpriv, rw_dir=built)
            rc, _, err = _run(prefix + ["bash", "build.sh"], built,
                              BUILD_TIMEOUT_S, env)
            if rc != 0:
                log(f"build.sh failed rc={rc}: {err[-500:]}")
                return False, None, f"build.sh failed rc={rc}"
        try:
            _freeze_readonly(built)
        except SubmissionTreeError as e:
            return False, None, f"unsafe build output: {e}"
        keep_built = True
        return True, built, ""
    finally:
        # A rejected, failed, or interrupted build must not leave its mutable
        # staging tree behind. Successful trees are retained for fit/predict and
        # removed by grade_corpus after the final dataset (or on any exception).
        if not keep_built:
            _rmtree_ro(built)


def _kill_group(pgid):
    """SIGKILL a whole process group; only an already-gone group is benign."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _preexec_limits(cpu_seconds):
    """Apply every mandatory limit without swallowing kernel errors."""
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
        resource.setrlimit(resource.RLIMIT_AS,
                           (MAX_ADDRESS_SPACE_BYTES, MAX_ADDRESS_SPACE_BYTES))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
        resource.setrlimit(resource.RLIMIT_NPROC, (MAX_PROCESSES, MAX_PROCESSES))
        resource.setrlimit(resource.RLIMIT_NOFILE, (MAX_OPEN_FILES, MAX_OPEN_FILES))
        os.setsid()
    return apply


def _cpu_limit_seconds(wall_timeout):
    """Aggregate process CPU allowance corresponding to a wall-clock phase cap."""
    return max(1, math.ceil(float(wall_timeout) * WORKER_VCPUS) + 1)


class _TailBuffer:
    """Thread-owned bounded byte ring for an untrusted stderr stream."""

    def __init__(self, cap=MAX_STDERR_BYTES):
        self.cap = cap
        self.data = bytearray()

    def feed(self, chunk):
        self.data.extend(chunk)
        overflow = len(self.data) - self.cap
        if overflow > 0:
            del self.data[:overflow]

    def text(self):
        return bytes(self.data).decode("utf-8", "replace")


def _drain_stderr(stream, tail):
    """Continuously drain stderr while retaining only its bounded tail."""
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                break
            tail.feed(chunk)
    except (OSError, ValueError):
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _run(cmd, cwd, timeout, env):
    """Run one phase in its own process group (created by the fail-closed preexec) so a
    timeout — or normal exit that left daemons behind — kills the entire tree,
    not just the direct child. stdout is discarded; stderr is continuously
    drained through a bounded in-memory ring, so flood output cannot consume the
    verifier's memory or disk.
    Returns (returncode, elapsed_s, stderr_tail). rc 124 == timeout."""
    t0 = time.time()
    if timeout is not None and timeout <= 0:
        return 124, 0.0, "TIMEOUT (no time budget remaining)"
    tail = _TailBuffer()
    drainer = None
    p = None
    try:
        try:
            cpu_seconds = _cpu_limit_seconds(timeout)
            p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.PIPE,
                                 preexec_fn=_preexec_limits(cpu_seconds))
        except OSError as e:
            raise SandboxUnavailable(
                f"mandatory sandbox phase could not start: {e}") from e
        drainer = threading.Thread(target=_drain_stderr, args=(p.stderr, tail),
                                   daemon=True)
        drainer.start()
        pgid = p.pid  # preexec os.setsid() makes the child its process-group leader
        timed_out = False
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            _kill_group(pgid)  # always: clean up any surviving group members
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            if drainer is not None:
                drainer.join(timeout=2)
                if drainer.is_alive() and p.stderr is not None:
                    # A descendant that escaped the process group may still hold
                    # the write end. Closing our read end prevents a leaked
                    # background thread and gives the writer SIGPIPE.
                    try:
                        p.stderr.close()
                    except OSError:
                        pass
                    drainer.join(timeout=1)
        err_tail = tail.text()
        if timed_out:
            return 124, time.time() - t0, "TIMEOUT"
        rc = p.returncode if p.returncode is not None else 137
        return rc, time.time() - t0, err_tail
    finally:
        if p is not None and p.stderr is not None:
            try:
                p.stderr.close()
            except OSError:
                pass


def read_pred_csv(path):
    """Return dict col->array, or None if malformed. Numeric columns become
    float arrays; a non-numeric column (e.g. an optional `sample_id`) is kept as
    a string array so the grader can verify exact input order without breaking
    parsing.

    Safety: the path must be a regular file -- a symlink/FIFO/socket/device
    raises PredReadError (not a silent None) -- it is opened O_NOFOLLOW, and it is
    bounded by MAX_PRED_BYTES / MAX_PRED_ROWS / MAX_PRED_COLS and parsed as a
    stream so a hostile submission cannot exhaust grader memory. Rejects (returns
    None) a missing file, an empty body, duplicate header names, or any row whose
    field count differs from the header -- a ragged CSV is ambiguous and must
    score as an invalid submission, not silently misalign."""
    fh = _open_regular_ro(path, MAX_PRED_BYTES)  # raises PredReadError if unsafe
    if fh is None:
        return None
    try:
        with fh:
            header = fh.readline().rstrip("\n").split(",")
            if not header or header == [""]:
                return None
            if len(set(header)) != len(header):
                return None  # duplicate column names -> ambiguous
            ncol = len(header)
            if ncol > MAX_PRED_COLS:
                raise PredReadError(
                    f"pred.csv has too many columns ({ncol} > {MAX_PRED_COLS})")
            rows = []
            for ln in fh:
                if not ln.strip():
                    continue
                r = ln.rstrip("\n").split(",")
                if len(r) != ncol:
                    return None  # ragged / inconsistent row widths
                rows.append(r)
                if len(rows) > MAX_PRED_ROWS:
                    raise PredReadError(
                        f"pred.csv exceeds row cap ({MAX_PRED_ROWS})")
            if not rows:
                return None
            cols = {}
            for i, h in enumerate(header):
                raw = [r[i] for r in rows]
                try:
                    cols[h] = np.array([float(v) for v in raw], dtype=np.float64)
                except (ValueError, IndexError, OverflowError):
                    # keep as strings so an optional id column doesn't break
                    # parsing; required prediction columns are coerced-or-rejected
                    # downstream.
                    cols[h] = np.array(raw, dtype=object)
            return cols
    except PredReadError:
        raise
    except Exception:
        return None


def _read_regular_bytes(path, max_bytes):
    """Read one bounded regular file without following its final component."""
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise PredReadError(f"cannot stat {os.path.basename(path)}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise PredReadError(f"{os.path.basename(path)} is not a regular file")
    if st.st_size > max_bytes:
        raise PredReadError(
            f"{os.path.basename(path)} too large ({st.st_size} > {max_bytes} bytes)")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        chunks = []
        remaining = st.st_size
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                raise PredReadError(f"short read from {os.path.basename(path)}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise PredReadError(f"{os.path.basename(path)} grew while it was read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def run_prefit_predict(
    submission_dir,
    dataset_public,
    model_bytes,
    *,
    n_test_expected=None,
    family=None,
):
    """Execute an already-fitted model through the exact submission predictor.

    This is the causal reference-reproduction path used after every trusted fit has
    completed.  It stages only ``model.out`` and the public test inputs, runs the
    exact locked ``predict`` executable inside the same mandatory, networkless
    sandbox and 30-second cap as a submission, then returns both the parsed values
    and the exact ``pred.csv`` bytes that caused them.  There is no in-process
    prediction implementation or permissive fallback.
    """
    if family != "binomial-logit":
        raise ValueError(f"unsupported family {family!r}; expected 'binomial-logit'")
    if not isinstance(model_bytes, bytes) or not model_bytes:
        raise ValueError("prefit model bytes must be nonempty bytes")
    if len(model_bytes) > MAX_OUTPUT_BYTES:
        raise ValueError("prefit model exceeds the output cap")
    _validate_submission_tree(submission_dir)
    preflight_sandbox(log=lambda _message: None)
    bwrap, setpriv = resolve_sandbox()
    run_dir = tempfile.mkdtemp(prefix="svpgs_prefit_predict_")
    _open_private_work(run_dir)
    sandbox = _bwrap_prefix(
        bwrap, setpriv, rw_dir=run_dir, submission=submission_dir)

    def stage(name):
        source = os.path.join(dataset_public, name)
        destination = os.path.join(run_dir, name)
        if os.path.exists(source):
            with _safe_create_dest(destination) as output, open(source, "rb") as input_:
                shutil.copyfileobj(input_, output)
        elif os.path.exists(source + ".gz"):
            import gzip
            with _safe_create_dest(destination) as output, \
                    gzip.open(source + ".gz", "rb") as input_:
                shutil.copyfileobj(input_, output)
        else:
            raise FileNotFoundError(source)

    try:
        with _safe_create_dest(os.path.join(run_dir, "model.out")) as output:
            output.write(model_bytes)
        for name in TEST_FILES:
            stage(name)
        predict = os.path.join(submission_dir, "predict")
        if not os.access(predict, os.X_OK):
            return {
                "status": "no_executable", "pred": None, "pred_bytes": None,
                "t_predict": 0.0, "predict_rc": None,
                "detail": "predict not executable",
            }
        rc, t_predict, error = _run(
            sandbox + ["/submission/predict"], run_dir,
            PREDICT_TIMEOUT_S, _submission_env(WORKER_VCPUS))
        if rc != 0:
            return {
                "status": "predict_failed", "pred": None, "pred_bytes": None,
                "t_predict": t_predict, "predict_rc": rc,
                "detail": f"rc={rc} {error}",
            }
        pred_path = os.path.join(run_dir, "pred.csv")
        try:
            pred = read_pred_csv(pred_path)
            pred_bytes = _read_regular_bytes(pred_path, MAX_PRED_BYTES)
        except PredReadError as exc:
            return {
                "status": "unsafe_pred", "pred": None, "pred_bytes": None,
                "t_predict": t_predict, "predict_rc": rc,
                "detail": f"unsafe pred.csv: {exc}",
            }
        if pred is None or "mean" not in pred:
            return {
                "status": "bad_pred", "pred": None, "pred_bytes": None,
                "t_predict": t_predict, "predict_rc": rc,
                "detail": "missing/malformed pred.csv",
            }
        mean = pred["mean"]
        if (
            mean.dtype.kind not in "fiu"
            or (n_test_expected is not None and len(mean) != n_test_expected)
            or not np.all(np.isfinite(mean))
            or mean.min() < 0.0
            or mean.max() > 1.0
        ):
            return {
                "status": "bad_pred", "pred": None, "pred_bytes": None,
                "t_predict": t_predict, "predict_rc": rc,
                "detail": "predictor output violates the binary prediction contract",
            }
        return {
            "status": "ok", "pred": pred, "pred_bytes": pred_bytes,
            "t_predict": t_predict, "predict_rc": rc, "detail": "",
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "pred_sha256": hashlib.sha256(pred_bytes).hexdigest(),
        }
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# ---- isolation ------------------------------------------------------------
# The submission's fit/predict run untrusted code. We do NOT hand them the
# grader's environment (which points PYTHONPATH at the repo root, exposing the
# hidden truth/anchors and the generator) and, where the kernel supports it,
# we run them behind a network-less mount+uid sandbox so they cannot read the
# corpus truth, the grader, or reach the network.

def _submission_env(numeric_threads):
    """A minimal, sanitized environment for the submission process. Repo- and
    venv-pointing variables (PYTHONPATH, VIRTUAL_ENV, PYTHONHOME, ...) are
    dropped so `import`-ing the grader package or the datagen truth is not one
    env var away; HOME/TMPDIR are pinned inside the run dir."""
    keep = {}
    for k in ("PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT"):
        if k in os.environ:
            keep[k] = os.environ[k]
    keep.setdefault("PATH", "/usr/bin:/bin:/usr/local/bin")
    keep["HOME"] = "/work"
    keep["TMPDIR"] = "/tmp"
    keep["PYTHONDONTWRITEBYTECODE"] = "1"
    if type(numeric_threads) is not int or not 1 <= numeric_threads <= WORKER_VCPUS:
        raise ValueError(
            f"numeric_threads must be an integer in [1, {WORKER_VCPUS}]")
    threads = str(numeric_threads)
    keep.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                MKL_NUM_THREADS=threads, VECLIB_MAXIMUM_THREADS=threads,
                NUMEXPR_NUM_THREADS=threads)
    return keep


def _validate_submission_tree(root):
    """Require a bounded tree containing directories and regular files only."""
    try:
        root_st = os.lstat(root)
    except OSError as e:
        raise SubmissionTreeError(f"cannot inspect submission: {e}") from e
    if not stat.S_ISDIR(root_st.st_mode):
        raise SubmissionTreeError("submission root is not a regular directory")

    n_files = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
            except OSError as e:
                raise SubmissionTreeError(f"cannot inspect submission entry {name!r}: {e}") from e
            if not stat.S_ISDIR(st.st_mode):
                raise SubmissionTreeError(
                    f"submission entry {os.path.relpath(p, root)!r} is not a directory")
        for name in filenames:
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
            except OSError as e:
                raise SubmissionTreeError(f"cannot inspect submission entry {name!r}: {e}") from e
            rel = os.path.relpath(p, root)
            if not stat.S_ISREG(st.st_mode):
                raise SubmissionTreeError(
                    f"submission entry {rel!r} is not a regular file (symlinks/FIFOs/sockets forbidden)")
            if st.st_size > MAX_SUBMISSION_FILE_BYTES:
                raise SubmissionTreeError(
                    f"submission file {rel!r} is too large ({st.st_size} bytes)")
            n_files += 1
            total += st.st_size
            if n_files > MAX_SUBMISSION_FILES:
                raise SubmissionTreeError(
                    f"submission has too many files ({n_files} > {MAX_SUBMISSION_FILES})")
            if total > MAX_SUBMISSION_TOTAL_BYTES:
                raise SubmissionTreeError(
                    f"submission is too large ({total} > {MAX_SUBMISSION_TOTAL_BYTES} bytes)")


def _which(prog):
    for d in os.environ.get("PATH", "/usr/bin:/bin").split(os.pathsep):
        p = os.path.join(d, prog)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def run_on_dataset(submission_dir, dataset_public, *, n_test_expected=None,
                   family=None):
    """Returns dict: status, pred (col->array) | None, t_fit, t_predict, detail.

    Every dataset receives the contract's non-transferable fit and predict caps.
    A phase that finishes early does not donate time to
    the other phase or to another dataset. A missing sandbox raises
    SandboxUnavailable rather than running unconfined."""
    if family != "binomial-logit":
        raise ValueError(f"unsupported family {family!r}; expected 'binomial-logit'")
    run_dir = tempfile.mkdtemp(prefix="svpgs_run_")
    _open_private_work(run_dir)
    # Untrusted code: sanitized env (no repo PYTHONPATH leaking truth/grader) and,
    # where available, a network-less mount+uid sandbox around each phase.
    # The solver contract advertises four vCPUs. Numeric runtimes must see that
    # exact allowance too; forcing BLAS/OpenMP to one thread made an honest
    # NumPy/SciPy implementation slower than the execution contract it was given.
    env = _submission_env(WORKER_VCPUS)
    try:
        _validate_submission_tree(submission_dir)
    except SubmissionTreeError as e:
        shutil.rmtree(run_dir, ignore_errors=True)
        return {"status": "submission_reject", "pred": None, "t_fit": 0,
                "t_predict": 0, "detail": str(e)}
    bwrap, setpriv = resolve_sandbox()
    sandbox = _bwrap_prefix(bwrap, setpriv, rw_dir=run_dir,
                            submission=submission_dir)
    # In bwrap mode the submission is bind-mounted at /submission; in external mode
    # there is no mount namespace, so exec it from its real path instead.
    submission_root = submission_dir if _external_sandbox() else "/submission"
    def stage(files):
        """Copy each staged file into the run dir WITHOUT following a symlink the
        submission's fit phase may have pre-created at the destination (which
        shutil.copy would follow, redirecting the write outside the run dir).
        _safe_create_dest unlinks any existing entry then creates the dest with
        O_CREAT|O_EXCL|O_NOFOLLOW -- fit cannot redirect the write, and cannot
        pre-seed a TEST_FILES path to leak the held-out test features early."""
        for f in files:
            src = os.path.join(dataset_public, f)
            dst = os.path.join(run_dir, f)
            if os.path.exists(src):
                with _safe_create_dest(dst) as fout, open(src, "rb") as fin:
                    shutil.copyfileobj(fin, fout)
            elif os.path.exists(src + ".gz"):
                # corpus ships large genotype tables gzipped to keep the repo small;
                # decompress into the run dir so the submission sees plain .tsv
                # (the agent contract is unchanged).
                import gzip
                with _safe_create_dest(dst) as fout, \
                        gzip.open(src + ".gz", "rb") as fin:
                    shutil.copyfileobj(fin, fout)
    try:
        # Stage metadata + training files only; test features are added after fit.
        stage(FIT_FILES)
        fit = os.path.join(submission_dir, "fit")
        predict = os.path.join(submission_dir, "predict")
        if not (os.access(fit, os.X_OK) and os.access(predict, os.X_OK)):
            return {"status": "no_executable", "pred": None, "t_fit": 0,
                    "t_predict": 0, "detail": "fit/predict not executable"}
        rc, t_fit, err = _run(sandbox + [os.path.join(submission_root, "fit")],
                              run_dir, FIT_TIMEOUT_S, env)
        if rc != 0:
            return {"status": "fit_failed", "pred": None, "t_fit": t_fit,
                    "t_predict": 0, "detail": f"rc={rc} {err}"}
        if not _is_regular_file(os.path.join(run_dir, "model.out")):
            return {"status": "no_model", "pred": None, "t_fit": t_fit,
                    "t_predict": 0, "detail": "fit wrote no regular model.out"}
        # Only now, after fit has finished, expose the held-out test features.
        stage(TEST_FILES)
        rc, t_pred, err = _run(sandbox + [os.path.join(submission_root, "predict")],
                               run_dir, PREDICT_TIMEOUT_S, env)
        if rc != 0:
            return {"status": "predict_failed", "pred": None, "t_fit": t_fit,
                    "t_predict": t_pred, "detail": f"rc={rc} {err}"}
        try:
            pred = read_pred_csv(os.path.join(run_dir, "pred.csv"))
        except PredReadError as e:
            return {"status": "unsafe_pred", "pred": None, "t_fit": t_fit,
                    "t_predict": t_pred, "detail": f"unsafe pred.csv: {e}"}
        if pred is None:
            return {"status": "bad_pred", "pred": None, "t_fit": t_fit,
                    "t_predict": t_pred, "detail": "missing/malformed pred.csv"}
        # validity: expected columns + finite + row count + probability range
        need = ["mean"]
        for c in need:
            if c not in pred:
                return {"status": "missing_col", "pred": None, "t_fit": t_fit,
                        "t_predict": t_pred, "detail": f"pred.csv missing '{c}'"}
            # A required column that failed numeric parsing is kept as an object
            # (string) array by read_pred_csv; treat that as malformed output at
            # the validity floor instead of letting np.isfinite raise TypeError.
            if pred[c].dtype.kind not in "fiu":
                return {"status": "bad_pred", "pred": None, "t_fit": t_fit,
                        "t_predict": t_pred,
                        "detail": f"non-numeric value in required column '{c}'"}
        arr = pred["mean"]
        if n_test_expected is not None and len(arr) != n_test_expected:
            return {"status": "row_count", "pred": None, "t_fit": t_fit,
                    "t_predict": t_pred,
                    "detail": f"pred rows {len(arr)} != {n_test_expected}"}
        for c in need:
            if not np.all(np.isfinite(pred[c])):
                return {"status": "nonfinite", "pred": None, "t_fit": t_fit,
                        "t_predict": t_pred, "detail": f"non-finite '{c}'"}
        if arr.min() < 0.0 or arr.max() > 1.0:
            return {"status": "prob_range", "pred": None, "t_fit": t_fit,
                    "t_predict": t_pred, "detail": "binary 'mean' outside [0,1]"}
        try:
            model_bytes = _read_regular_bytes(
                os.path.join(run_dir, "model.out"), MAX_OUTPUT_BYTES)
            pred_bytes = _read_regular_bytes(
                os.path.join(run_dir, "pred.csv"), MAX_PRED_BYTES)
        except PredReadError as e:
            return {"status": "unsafe_output", "pred": None, "t_fit": t_fit,
                    "t_predict": t_pred, "detail": str(e)}
        return {
            "status": "ok", "pred": pred, "pred_bytes": pred_bytes,
            "t_fit": t_fit, "t_predict": t_pred,
            "fit_rc": 0, "predict_rc": 0,
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "pred_sha256": hashlib.sha256(pred_bytes).hexdigest(),
            "detail": "",
        }
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

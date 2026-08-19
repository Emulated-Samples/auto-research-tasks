"""Key handling and domain-separated authentication for corpus schema v6."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from typing import Any


KEY_BYTES = 32
PURPOSES = frozenset({"development", "shipping"})
STREAMS = frozenset({"dgp", "materialize", "bootstrap", "dataset-id"})
ANCHOR_HMAC_DOMAIN = b"svpgsbench|schema-v6|anchor\x00"
MANIFEST_HMAC_DOMAIN = b"svpgsbench|schema-v6|manifest\x00"
DEVELOPMENT_RESULT_HMAC_DOMAIN = b"svpgsbench|schema-v6|development-result\x00"
DEVELOPMENT_REPORT_HMAC_DOMAIN = b"svpgsbench|schema-v6|development-report\x00"


class CorpusKeyError(ValueError):
    """The required private corpus key is absent, unsafe, or malformed."""


def load_corpus_key(key_file: str, *, repository_root: str) -> bytes:
    """Load an exact 32-byte, owner-only regular file outside the repository.

    The descriptor, rather than a pathname stat, is authoritative. ``O_NOFOLLOW``
    rejects a symlink at the final component and ``fstat`` closes the usual
    check/open race. An absolute path is required so invocation location cannot
    silently select a different secret.
    """
    if not isinstance(key_file, str) or not key_file or not os.path.isabs(key_file):
        raise CorpusKeyError("corpus key path must be a non-empty absolute path")
    repository_real = os.path.realpath(repository_root)
    key_real = os.path.realpath(key_file)
    try:
        inside_repository = (
            os.path.commonpath((repository_real, key_real)) == repository_real
        )
    except ValueError as exc:
        raise CorpusKeyError(
            "corpus key path cannot be compared to repository root"
        ) from exc
    if inside_repository:
        raise CorpusKeyError("corpus key must be stored outside the repository")

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(key_file, flags)
    except OSError as exc:
        raise CorpusKeyError(f"cannot securely open corpus key: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CorpusKeyError("corpus key must be a regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise CorpusKeyError("corpus key mode must be exactly 0600")
        if metadata.st_uid != os.geteuid():
            raise CorpusKeyError("corpus key must be owned by the current user")
        if metadata.st_nlink != 1:
            raise CorpusKeyError("corpus key must have exactly one hard link")
        key = b""
        while len(key) <= KEY_BYTES:
            chunk = os.read(descriptor, KEY_BYTES + 1 - len(key))
            if not chunk:
                break
            key += chunk
    finally:
        os.close(descriptor)
    if len(key) != KEY_BYTES:
        raise CorpusKeyError(f"corpus key must contain exactly {KEY_BYTES} bytes")
    return key


def corpus_key_id(key: bytes) -> str:
    """Return the non-secret identifier recorded in a schema-v6 manifest."""
    _require_key(key)
    return hashlib.sha256(key).hexdigest()


def _require_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) != KEY_BYTES:
        raise CorpusKeyError(f"corpus key must contain exactly {KEY_BYTES} bytes")


def _require_pipeline_digest(pipeline_sha256: str) -> None:
    if (
        not isinstance(pipeline_sha256, str)
        or len(pipeline_sha256) != 64
        or any(character not in "0123456789abcdef" for character in pipeline_sha256)
    ):
        raise ValueError("pipeline_sha256 must be a lowercase SHA-256 digest")


def stream_context(
    *,
    purpose: str,
    pipeline_sha256: str,
    category: str,
    replicate: int,
    stream: str,
) -> bytes:
    """Return the exact schema-v6 HMAC context for one independent RNG stream."""
    if purpose not in PURPOSES:
        raise ValueError(f"invalid corpus purpose {purpose!r}")
    _require_pipeline_digest(pipeline_sha256)
    if (
        not isinstance(category, str)
        or not category
        or "|" in category
        or "\x00" in category
    ):
        raise ValueError(f"invalid category for stream derivation: {category!r}")
    if type(replicate) is not int or replicate < 0:
        raise ValueError(f"replicate must be a non-negative integer, got {replicate!r}")
    if stream not in STREAMS:
        raise ValueError(f"invalid corpus stream {stream!r}")
    return (
        "schema-v6"
        f"|purpose={purpose}"
        f"|pipeline_sha256={pipeline_sha256}"
        f"|category={category}"
        f"|replicate={replicate}"
        f"|stream={stream}"
    ).encode("utf-8")


def derive_stream_digest(
    key: bytes,
    *,
    purpose: str,
    pipeline_sha256: str,
    category: str,
    replicate: int,
    stream: str,
) -> bytes:
    """Derive one full HMAC-SHA256 stream root from the private corpus key."""
    _require_key(key)
    context = stream_context(
        purpose=purpose,
        pipeline_sha256=pipeline_sha256,
        category=category,
        replicate=replicate,
        stream=stream,
    )
    return hmac.new(key, context, hashlib.sha256).digest()


def derive_stream_seed(
    key: bytes,
    *,
    purpose: str,
    pipeline_sha256: str,
    category: str,
    replicate: int,
    stream: str,
) -> int:
    """Derive an unsigned 128-bit seed; the seed itself is never serialized."""
    digest = derive_stream_digest(
        key,
        purpose=purpose,
        pipeline_sha256=pipeline_sha256,
        category=category,
        replicate=replicate,
        stream=stream,
    )
    return int.from_bytes(digest[:16], "big", signed=False)


def opaque_dataset_id(
    key: bytes,
    *,
    purpose: str,
    pipeline_sha256: str,
    category: str,
    replicate: int,
) -> str:
    """Derive the opaque, stable-on-this-pipeline schema-v6 dataset identifier."""
    digest = derive_stream_digest(
        key,
        purpose=purpose,
        pipeline_sha256=pipeline_sha256,
        category=category,
        replicate=replicate,
        stream="dataset-id",
    )
    return "d_" + digest[:16].hex()


def canonical_json_payload(value: dict[str, Any], *, exclude_field: str) -> bytes:
    """Canonicalize a signed JSON object while omitting its signature field."""
    if not isinstance(value, dict):
        raise TypeError("authenticated JSON value must be an object")
    payload = {key: item for key, item in value.items() if key != exclude_field}
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def json_hmac_sha256(
    key: bytes,
    value: dict[str, Any],
    *,
    exclude_field: str,
    domain: bytes,
) -> str:
    """Authenticate one canonical JSON payload under an explicit domain."""
    _require_key(key)
    if not isinstance(domain, bytes) or not domain.endswith(b"\x00"):
        raise ValueError("HMAC domain must be NUL-terminated bytes")
    payload = canonical_json_payload(value, exclude_field=exclude_field)
    return hmac.new(key, domain + payload, hashlib.sha256).hexdigest()


def verify_json_hmac(
    key: bytes,
    value: dict[str, Any],
    *,
    field: str,
    domain: bytes,
) -> bool:
    """Verify a lowercase HMAC-SHA256 using timing-safe comparison."""
    supplied = value.get(field) if isinstance(value, dict) else None
    if (
        not isinstance(supplied, str)
        or len(supplied) != 64
        or any(character not in "0123456789abcdef" for character in supplied)
    ):
        return False
    expected = json_hmac_sha256(
        key,
        value,
        exclude_field=field,
        domain=domain,
    )
    return hmac.compare_digest(supplied, expected)

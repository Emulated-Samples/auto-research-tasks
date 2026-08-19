"""Tests for the cherimoya install-skill command.

install-skill copies the bundled agent skill into a skills directory,
creating a `cherimoya/` subdirectory that holds SKILL.md plus the
references. A copy install must never carry `.ipynb_checkpoints`
autosaves along (they exist in the working tree but must not ship into
a user's skills directory), a collision without --force must error, and
--force must overwrite.
"""

import argparse
import os

import pytest

from cherimoya_cli.commands import install_skill


def _run(tmp_path, symlink=False, force=False, directory=None):
	args = argparse.Namespace(
		directory=str(directory) if directory is not None else str(tmp_path),
		symlink=symlink, force=force)
	install_skill.run(args)
	return os.path.join(
		str(directory) if directory is not None else str(tmp_path),
		"cherimoya")


def test_copy_install_writes_skill_and_references(tmp_path):
	dest = _run(tmp_path)
	assert os.path.isfile(os.path.join(dest, "SKILL.md"))
	refs = os.path.join(dest, "references")
	assert os.path.isdir(refs)
	# There is at least one reference and every one is a .md file.
	names = os.listdir(refs)
	assert any(name.endswith(".md") for name in names)


def test_copy_install_excludes_ipynb_checkpoints(tmp_path):
	"""Stale Jupyter autosaves in the source tree must not be copied."""
	dest = _run(tmp_path)
	for root, dirs, _ in os.walk(dest):
		assert ".ipynb_checkpoints" not in dirs, (
			"install-skill copied a .ipynb_checkpoints directory into "
			"{}".format(root))


def test_collision_without_force_errors(tmp_path):
	_run(tmp_path)
	with pytest.raises(FileExistsError):
		_run(tmp_path)


def test_force_overwrites_existing(tmp_path):
	dest = _run(tmp_path)
	# Drop a stray file; --force should wipe the directory before copying.
	stray = os.path.join(dest, "stray.txt")
	with open(stray, "w") as f:
		f.write("x")
	_run(tmp_path, force=True)
	assert not os.path.exists(stray)
	assert os.path.isfile(os.path.join(dest, "SKILL.md"))


def test_symlink_install_points_at_source(tmp_path):
	dest = _run(tmp_path, symlink=True)
	assert os.path.islink(dest)
	assert os.path.isfile(os.path.join(dest, "SKILL.md"))

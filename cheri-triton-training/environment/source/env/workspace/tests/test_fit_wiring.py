"""Wiring tests for `cherimoya fit` — confirms parameters flow into the
right downstream calls without actually training."""

import argparse
import json
from unittest import mock

import pytest


@pytest.fixture
def fit_json(tmp_path):
	"""Write a minimal JSON that satisfies merge_parameters' required keys."""
	from cherimoya_cli.defaults import default_fit_parameters

	# Include every key from the defaults so merge_parameters' "missing
	# required" check passes (it errors when a key is absent and its
	# default is None outside a small whitelist). Then override the
	# values we care about for the test.
	cfg = dict(default_fit_parameters)
	cfg['sequences'] = 'fake.fa'
	cfg['loci'] = 'fake.bed'
	cfg['negatives'] = 'fake_negatives.bed'
	cfg['signals'] = ['fake.bw']
	cfg['name'] = 'fit_wiring_test'
	cfg['device'] = 'cpu'
	cfg['num_workers'] = 3   # the value we want to verify is forwarded
	cfg['batch_size'] = 16

	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))
	return str(path)


def test_fit_forwards_num_workers_to_peak_generator(fit_json):
	"""fit.run must pass parameters['num_workers'] to PeakGenerator. We
	stop execution immediately after the call by raising from a fake
	PeakGenerator and then inspect the kwargs."""

	from cherimoya_cli.commands import fit as fit_cmd

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_peak_generator(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	# Block import-time side effects of the heavy modules `fit.run`
	# pulls in. We only need to exercise the wiring up to PeakGenerator.
	with mock.patch("cherimoya.io.PeakGenerator", side_effect=fake_peak_generator):
		try:
			fit_cmd.run(argparse.Namespace(parameters=fit_json))
		except _StopFit:
			pass
		except Exception as e:
			# Any error AFTER PeakGenerator was called is fine — the
			# point of the test is whether it received the right kwargs.
			if not captured:
				raise

	assert captured, "PeakGenerator was never called"
	assert captured.get('num_workers') == 3, (
		"fit.run did not forward num_workers; got {!r}"
		.format(captured.get('num_workers'))
	)


def test_default_fit_parameters_default_num_workers_is_one():
	from cherimoya_cli.defaults import (
		default_fit_parameters,
		default_pipeline_parameters,
	)
	assert default_fit_parameters['num_workers'] == 1
	assert default_pipeline_parameters['fit_parameters']['num_workers'] == 1


def test_fit_forwards_grouped_signals_to_peak_generator(tmp_path):
	"""When the JSON gives a grouped signals spec, fit.run must
	forward the inferred signal_groups (and control_groups) to
	PeakGenerator. Without this the structured form would be flattened
	to "all unstranded" downstream."""

	from cherimoya_cli.commands import fit as fit_cmd
	from cherimoya_cli.defaults import default_fit_parameters

	cfg = dict(default_fit_parameters)
	cfg['sequences'] = 'fake.fa'
	cfg['loci'] = 'fake.bed'
	cfg['negatives'] = 'fake_negatives.bed'
	# One unstranded ATAC group + one stranded TF group; per-group
	# counts mean signal_groups=[1, 2].
	cfg['signals'] = ['atac.bw', ['ctcf.+.bw', 'ctcf.-.bw']]
	cfg['controls'] = [['ctl.+.bw', 'ctl.-.bw']]
	cfg['name'] = 'fit_wiring_groups_test'
	cfg['device'] = 'cpu'

	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_peak_generator(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	with mock.patch(
			"cherimoya.io.PeakGenerator", side_effect=fake_peak_generator):
		try:
			fit_cmd.run(argparse.Namespace(parameters=str(path)))
		except _StopFit:
			pass
		except Exception:
			if not captured:
				raise

	assert captured.get('signal_groups') == [1, 2], (
		"signal_groups not forwarded; got {!r}".format(
			captured.get('signal_groups')))
	assert captured.get('control_groups') == [2], (
		"control_groups not forwarded; got {!r}".format(
			captured.get('control_groups')))


def test_fit_does_not_flatten_signals_for_downstream_evaluate(tmp_path):
	"""fit.run hands ``parameters['signals']`` to PeakGenerator and
	(at the end of training) deepcopies the same dict into the
	evaluate JSON. If fit silently re-writes ``signals`` to its flat
	form, a stranded pair ``[[+, -]]`` becomes ``[+, -]`` in the
	evaluate JSON, which then re-parses as two *unstranded* channels
	— the same bug the grouping API was added to prevent. Pin the
	contract by snapshotting what PeakGenerator actually receives."""

	from cherimoya_cli.commands import fit as fit_cmd
	from cherimoya_cli.defaults import default_fit_parameters

	original_signals = [['ctcf.+.bw', 'ctcf.-.bw']]
	cfg = dict(default_fit_parameters)
	cfg['sequences'] = 'fake.fa'
	cfg['loci'] = 'fake.bed'
	cfg['negatives'] = 'fake_negatives.bed'
	cfg['signals'] = original_signals
	cfg['name'] = str(tmp_path / 'fit_eval_roundtrip')
	cfg['device'] = 'cpu'

	path = tmp_path / "fit.json"
	path.write_text(json.dumps(cfg))

	captured = {}

	class _StopFit(Exception):
		pass

	def fake_peak_generator(**kwargs):
		captured.update(kwargs)
		raise _StopFit()

	with mock.patch(
			"cherimoya.io.PeakGenerator", side_effect=fake_peak_generator):
		try:
			fit_cmd.run(argparse.Namespace(parameters=str(path)))
		except _StopFit:
			pass
		except Exception:
			if not captured:
				raise

	# PeakGenerator must see the *structured* signals form. If fit had
	# pre-flattened it the assertion below would fail with
	# `signals == ['ctcf.+.bw', 'ctcf.-.bw']` (two unstranded tracks).
	assert captured.get('signals') == original_signals, (
		"fit flattened the structured signals form before PeakGenerator: "
		"got {!r}".format(captured.get('signals')))

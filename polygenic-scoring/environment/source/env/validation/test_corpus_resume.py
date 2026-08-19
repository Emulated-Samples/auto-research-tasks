from datagen import build_corpus


def test_build_all_resumes_from_authenticated_partial_datasets(monkeypatch, tmp_path):
    expected = (("complete", 0), ("missing", 0))
    checks = []
    built = []
    finalized = []

    monkeypatch.setattr(build_corpus, "CORPUS", str(tmp_path))
    monkeypatch.setattr(build_corpus, "_expected_datasets", lambda: expected)
    monkeypatch.setattr(
        build_corpus,
        "_generation_pipeline_sha256",
        lambda: "pipeline",
    )
    monkeypatch.setattr(
        build_corpus,
        "_dataset_id",
        lambda category, replicate, auth_key, pipeline: f"{category}-{replicate}",
    )

    def provenance_matches(
        out_dir,
        category,
        replicate,
        auth_key,
        pipeline,
        *,
        require_manifest=False,
    ):
        checks.append((category, replicate, require_manifest))
        return (category == "complete", "match" if category == "complete" else "missing")

    monkeypatch.setattr(build_corpus, "_provenance_matches", provenance_matches)
    monkeypatch.setattr(
        build_corpus,
        "build_one",
        lambda category, replicate, auth_key, pipeline: built.append(
            (category, replicate, pipeline)
        ),
    )
    monkeypatch.setattr(
        build_corpus,
        "finalize",
        lambda auth_key, pipeline: finalized.append(pipeline),
    )

    build_corpus.build_all(b"key")

    assert checks == [
        ("complete", 0, False),
        ("missing", 0, False),
    ]
    assert built == [("missing", 0, "pipeline")]
    assert finalized == ["pipeline"]

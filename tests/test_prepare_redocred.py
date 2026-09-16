import hashlib

import pytest

from knowledge_agents.prepare_redocred import download_verified


def test_download_verified_uses_hash_and_reuses_valid_file(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b"official-data")
    expected = hashlib.sha256(b"official-data").hexdigest()
    destination = tmp_path / "nested" / "download.json"
    result = download_verified(source.as_uri(), destination, expected)
    assert result.read_bytes() == b"official-data"
    assert download_verified("not-used", destination, expected) == destination


def test_download_verified_rejects_wrong_hash(tmp_path):
    source = tmp_path / "source.json"
    source.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="hash mismatch"):
        download_verified(source.as_uri(), tmp_path / "download.json", "0" * 64)

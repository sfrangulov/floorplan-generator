# tests/test_cli_generator.py
import json
import os
import sys
import tempfile
import pytest
from click.testing import CliRunner

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from floorplan_generator import cli


def test_cli_generates_json_files():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            "--count", "3", "--seed", "42",
            "--num-rooms", "3-5",
            "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0, result.output
        files = sorted(os.listdir(tmpdir))
        assert len(files) == 3
        assert all(f.endswith(".json") for f in files)
        with open(os.path.join(tmpdir, files[0])) as f:
            data = json.load(f)
        assert "meta" in data
        assert "spaces" in data


def test_cli_reproducible():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d1, \
         tempfile.TemporaryDirectory() as d2:
        runner.invoke(cli, ["--count", "2", "--seed", "42", "--output-dir", d1])
        runner.invoke(cli, ["--count", "2", "--seed", "42", "--output-dir", d2])
        for fname in os.listdir(d1):
            with open(os.path.join(d1, fname)) as f1, \
                 open(os.path.join(d2, fname)) as f2:
                assert json.load(f1) == json.load(f2)

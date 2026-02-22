# tests/test_cli_synthetic.py
import json
import os
import sys
import tempfile
import pytest
from click.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from floorplan_generator import cli as gen_cli
from generate_synthetic import cli as synth_cli


@pytest.fixture
def json_dir():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(gen_cli, [
            "--count", "3", "--seed", "42",
            "--num-rooms", "3-5", "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0
        yield tmpdir


def test_render_produces_images(json_dir):
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as outdir:
        result = runner.invoke(synth_cli, [
            "--input-dir", json_dir,
            "--output-dir", outdir,
            "--image-size", "256",
            "--workers", "1",
        ])
        assert result.exit_code == 0, result.output
        images = [f for f in os.listdir(os.path.join(outdir, "images"))
                  if f.endswith(".png")]
        assert len(images) == 3


def test_render_with_masks(json_dir):
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as outdir:
        result = runner.invoke(synth_cli, [
            "--input-dir", json_dir,
            "--output-dir", outdir,
            "--image-size", "256",
            "--workers", "1",
            "--with-masks",
        ])
        assert result.exit_code == 0
        masks = os.listdir(os.path.join(outdir, "masks"))
        assert len(masks) == 3


def test_render_with_coco(json_dir):
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as outdir:
        result = runner.invoke(synth_cli, [
            "--input-dir", json_dir,
            "--output-dir", outdir,
            "--image-size", "256",
            "--workers", "1",
            "--with-coco",
        ])
        assert result.exit_code == 0
        coco_path = os.path.join(outdir, "annotations", "coco.json")
        assert os.path.exists(coco_path)
        with open(coco_path) as f:
            coco = json.load(f)
        assert len(coco["images"]) == 3
        assert len(coco["annotations"]) > 0
        assert len(coco["categories"]) == 15

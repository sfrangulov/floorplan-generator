# tests/test_integration.py
"""End-to-end pipeline test: generate JSON -> render images -> verify annotations."""
import json
import os
import sys
import tempfile
import pytest
from click.testing import CliRunner
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from floorplan_generator import cli as gen_cli
from generate_synthetic import cli as synth_cli


def test_full_pipeline():
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as workdir:
        json_dir = os.path.join(workdir, "json")
        output_dir = os.path.join(workdir, "output")

        # Step 1: Generate JSONs
        result = runner.invoke(gen_cli, [
            "--count", "5", "--seed", "42",
            "--num-rooms", "3-6",
            "--output-dir", json_dir,
        ])
        assert result.exit_code == 0, f"Generator failed: {result.output}"
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        assert len(json_files) == 5

        # Step 2: Render with all outputs
        result = runner.invoke(synth_cli, [
            "--input-dir", json_dir,
            "--output-dir", output_dir,
            "--image-size", "512",
            "--workers", "1",
            "--with-masks",
            "--with-coco",
        ])
        assert result.exit_code == 0, f"Renderer failed: {result.output}"

        # Verify images
        img_dir = os.path.join(output_dir, "images")
        images = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        assert len(images) == 5
        img = Image.open(os.path.join(img_dir, images[0]))
        assert img.size == (512, 512)

        # Verify masks
        mask_dir = os.path.join(output_dir, "masks")
        masks = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
        assert len(masks) == 5
        mask = Image.open(os.path.join(mask_dir, masks[0]))
        assert mask.mode == "L"

        # Verify COCO
        coco_path = os.path.join(output_dir, "annotations", "coco.json")
        assert os.path.exists(coco_path)
        with open(coco_path) as f:
            coco = json.load(f)
        assert len(coco["images"]) == 5
        assert len(coco["categories"]) == 7
        assert len(coco["annotations"]) >= 5  # at least 1 per image

        # Verify annotation references valid images
        image_ids = {im["id"] for im in coco["images"]}
        for ann in coco["annotations"]:
            assert ann["image_id"] in image_ids
            assert ann["bbox"][2] > 0  # width > 0
            assert ann["bbox"][3] > 0  # height > 0

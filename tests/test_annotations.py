# tests/test_annotations.py
import json
import numpy as np
import pytest
from PIL import Image

from floorplan.generator import FloorplanGenerator, GeneratorConfig
from floorplan.annotations import COCOAnnotator


@pytest.fixture
def sample_floorplan():
    cfg = GeneratorConfig(num_rooms=4, seed=42)
    return FloorplanGenerator(cfg).generate()


def test_semantic_mask_shape(sample_floorplan):
    ann = COCOAnnotator(image_size=512)
    mask = ann.generate_semantic_mask(sample_floorplan)
    assert isinstance(mask, Image.Image)
    assert mask.size == (512, 512)
    assert mask.mode == "L"


def test_semantic_mask_has_classes(sample_floorplan):
    ann = COCOAnnotator(image_size=512)
    mask = ann.generate_semantic_mask(sample_floorplan)
    pixels = set(mask.getdata())
    assert 0 in pixels  # background
    assert len(pixels) > 1  # at least one non-bg class


def test_instance_mask_shape(sample_floorplan):
    ann = COCOAnnotator(image_size=512)
    mask = ann.generate_instance_mask(sample_floorplan)
    assert mask.size == (512, 512)
    assert mask.mode == "I"  # 32-bit integer


def test_coco_json_structure(sample_floorplan):
    ann = COCOAnnotator(image_size=512)
    coco = ann.generate_coco_annotation(sample_floorplan, image_id=1,
                                         file_name="test.png")
    assert "image" in coco
    assert "annotations" in coco
    assert coco["image"]["id"] == 1
    assert coco["image"]["width"] == 512
    assert len(coco["annotations"]) > 0
    for a in coco["annotations"]:
        assert "bbox" in a
        assert "segmentation" in a
        assert "category_id" in a
        assert len(a["bbox"]) == 4


def test_coco_categories():
    ann = COCOAnnotator(image_size=512)
    cats = ann.get_categories()
    assert len(cats) == 7  # wall, room, corridor, bathroom, utility, door, window
    names = {c["name"] for c in cats}
    assert "wall" in names
    assert "living_room" in names

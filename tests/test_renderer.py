# tests/test_renderer.py
import pytest
import numpy as np
from PIL import Image

from floorplan.generator import FloorplanGenerator, GeneratorConfig
from floorplan.renderer import FloorplanRenderer, RenderConfig
from floorplan.styles import generate_style


@pytest.fixture
def sample_floorplan():
    cfg = GeneratorConfig(num_rooms=4, seed=42)
    return FloorplanGenerator(cfg).generate()


def test_renderer_produces_image(sample_floorplan):
    style = generate_style(np.random.default_rng(42))
    renderer = FloorplanRenderer(RenderConfig(image_size=512))
    img = renderer.render(sample_floorplan, style)
    assert isinstance(img, Image.Image)
    assert img.size == (512, 512)
    assert img.mode == "RGB"


def test_renderer_different_sizes(sample_floorplan):
    style = generate_style(np.random.default_rng(42))
    for size in [256, 512, 1024]:
        renderer = FloorplanRenderer(RenderConfig(image_size=size))
        img = renderer.render(sample_floorplan, style)
        assert img.size == (size, size)


def test_renderer_deterministic(sample_floorplan):
    style = generate_style(np.random.default_rng(42))
    r1 = FloorplanRenderer(RenderConfig(image_size=256))
    r2 = FloorplanRenderer(RenderConfig(image_size=256))
    img1 = r1.render(sample_floorplan, style)
    img2 = r2.render(sample_floorplan, style)
    assert list(img1.getdata()) == list(img2.getdata())


def test_renderer_not_all_background(sample_floorplan):
    style = generate_style(np.random.default_rng(42))
    renderer = FloorplanRenderer(RenderConfig(image_size=512))
    img = renderer.render(sample_floorplan, style)
    pixels = list(img.getdata())
    bg = style.bg_color
    non_bg = [p for p in pixels if p != bg]
    assert len(non_bg) > 100  # walls drawn

import numpy as np
import pytest
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from floorplan.furniture import FurniturePlacer


def test_programmatic_furniture_draws():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    poly = Polygon([(20, 20), (180, 20), (180, 180), (20, 180)])
    placer = FurniturePlacer(furniture_dir=None)
    rng = np.random.default_rng(42)
    placer.place_furniture(draw, poly, "living_room", rng, density=0.8,
                           transform={"scale": 1.0, "offset_x": 0, "offset_y": 0})
    pixels = list(img.getdata())
    white_count = sum(1 for p in pixels if p == (255, 255, 255))
    assert white_count < 200 * 200  # something was drawn


def test_no_furniture_at_zero_density():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    poly = Polygon([(20, 20), (180, 20), (180, 180), (20, 180)])
    placer = FurniturePlacer(furniture_dir=None)
    rng = np.random.default_rng(42)
    placer.place_furniture(draw, poly, "living_room", rng, density=0.0,
                           transform={"scale": 1.0, "offset_x": 0, "offset_y": 0})
    pixels = list(img.getdata())
    white_count = sum(1 for p in pixels if p == (255, 255, 255))
    assert white_count == 200 * 200

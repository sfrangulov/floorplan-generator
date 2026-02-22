# tests/test_geometry.py
import pytest
import numpy as np
from shapely.geometry import Polygon, box

from floorplan.geometry import (
    create_rectangle,
    create_indented_room,
    attach_room_to_wall,
    find_shared_walls,
    classify_external_walls,
    snap_to_grid,
    polygons_overlap,
)


def test_create_rectangle():
    poly = create_rectangle(0, 0, 3000, 4000)
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    assert abs(poly.area - 3000 * 4000) < 1e-6


def test_create_indented_room():
    rng = np.random.default_rng(42)
    poly = create_indented_room(
        width=4000, height=3000,
        indent_depth=500, indent_width=1000,
        rng=rng
    )
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    assert poly.area < 4000 * 3000  # indentation reduces area
    assert len(poly.exterior.coords) > 5  # more than rectangle


def test_snap_to_grid():
    coords = [(100.3, 200.7), (300.1, 400.9)]
    snapped = snap_to_grid(coords, grid_size=1.0)
    assert snapped == [(100.0, 201.0), (300.0, 401.0)]


def test_polygons_overlap():
    p1 = box(0, 0, 100, 100)
    p2 = box(50, 50, 150, 150)
    p3 = box(200, 200, 300, 300)
    assert polygons_overlap(p1, p2) is True
    assert polygons_overlap(p1, p3) is False


def test_polygons_touching_not_overlap():
    p1 = box(0, 0, 100, 100)
    p2 = box(100, 0, 200, 100)  # shares edge
    assert polygons_overlap(p1, p2) is False


def test_attach_room_to_wall():
    rng = np.random.default_rng(42)
    existing = box(0, 0, 3000, 4000)
    wall_start = np.array([3000.0, 0.0])
    wall_end = np.array([3000.0, 4000.0])
    new_width, new_height = 2500, 3000
    result = attach_room_to_wall(
        wall_start, wall_end, new_width, new_height, rng=rng
    )
    assert isinstance(result, Polygon)
    assert result.is_valid
    # New room should be to the right of x=3000
    minx, _, _, _ = result.bounds
    assert minx >= 3000 - 1  # tolerance for snapping


def test_find_shared_walls():
    p1 = box(0, 0, 3000, 4000)
    p2 = box(3000, 0, 6000, 4000)  # shares right wall of p1
    shared = find_shared_walls(p1, p2)
    assert len(shared) > 0
    # shared wall should be along x=3000
    for seg in shared:
        assert abs(seg[0][0] - 3000) < 1 or abs(seg[1][0] - 3000) < 1


def test_classify_external_walls():
    polys = [box(0, 0, 100, 100), box(100, 0, 200, 100)]
    walls_0 = classify_external_walls(0, polys)
    # The wall at x=100 (shared with poly 1) should be internal
    external_count = sum(1 for _, is_ext in walls_0 if is_ext)
    internal_count = sum(1 for _, is_ext in walls_0 if not is_ext)
    assert external_count == 3  # top, bottom, left
    assert internal_count == 1  # right (shared)

# src/floorplan/geometry.py
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, LineString, box as shapely_box


def create_rectangle(x: float, y: float, width: float, height: float) -> Polygon:
    """Create a rectangle polygon at (x, y) with given dimensions."""
    return shapely_box(x, y, x + width, y + height)


def create_indented_room(
    width: float,
    height: float,
    indent_depth: float,
    indent_width: float,
    rng: np.random.Generator,
    origin_x: float = 0,
    origin_y: float = 0,
) -> Polygon:
    """Create a room with a rectangular indentation cut from one wall."""
    base = create_rectangle(origin_x, origin_y, width, height)
    wall_idx = rng.integers(0, 4)

    if wall_idx == 0:  # bottom wall
        offset = rng.uniform(0, width - indent_width)
        cutout = shapely_box(
            origin_x + offset,
            origin_y,
            origin_x + offset + indent_width,
            origin_y + indent_depth,
        )
    elif wall_idx == 1:  # right wall
        offset = rng.uniform(0, height - indent_width)
        cutout = shapely_box(
            origin_x + width - indent_depth,
            origin_y + offset,
            origin_x + width,
            origin_y + offset + indent_width,
        )
    elif wall_idx == 2:  # top wall
        offset = rng.uniform(0, width - indent_width)
        cutout = shapely_box(
            origin_x + offset,
            origin_y + height - indent_depth,
            origin_x + offset + indent_width,
            origin_y + height,
        )
    else:  # left wall
        offset = rng.uniform(0, height - indent_width)
        cutout = shapely_box(
            origin_x,
            origin_y + offset,
            origin_x + indent_depth,
            origin_y + offset + indent_width,
        )

    result = base.difference(cutout)
    if not result.is_valid or result.is_empty:
        return base
    return result


def snap_to_grid(
    coords: list[tuple[float, float]], grid_size: float
) -> list[tuple[float, float]]:
    """Snap coordinates to nearest grid point."""
    return [
        (round(x / grid_size) * grid_size, round(y / grid_size) * grid_size)
        for x, y in coords
    ]


def polygons_overlap(p1: Polygon, p2: Polygon) -> bool:
    """Check if two polygons overlap (sharing an edge is NOT overlap)."""
    if not p1.intersects(p2):
        return False
    inter = p1.intersection(p2)
    return inter.area > 1e-6


def attach_room_to_wall(
    wall_start: np.ndarray,
    wall_end: np.ndarray,
    new_width: float,
    new_height: float,
    rng: np.random.Generator,
) -> Polygon:
    """Create a new room polygon attached to the given wall segment.

    wall_start, wall_end define the wall segment.
    new_width is measured along the wall direction, new_height is perpendicular outward.
    """
    wall_vec = wall_end - wall_start
    wall_length = np.linalg.norm(wall_vec)
    wall_dir = wall_vec / wall_length
    normal = np.array([wall_dir[1], -wall_dir[0]])

    max_offset = max(0, wall_length - new_width)
    offset = rng.uniform(0, max_offset) if max_offset > 0 else 0

    p0 = wall_start + wall_dir * offset
    p1 = p0 + wall_dir * min(new_width, wall_length)
    p2 = p1 + normal * new_height
    p3 = p0 + normal * new_height

    coords = [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]
    poly = Polygon(coords)

    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def find_shared_walls(
    p1: Polygon, p2: Polygon, tolerance: float = 5.0
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Find shared wall segments between two polygons."""
    shared = []
    boundary_inter = p1.boundary.intersection(p2.boundary)

    if boundary_inter.is_empty:
        return shared

    if boundary_inter.geom_type == "MultiLineString":
        lines = list(boundary_inter.geoms)
    elif boundary_inter.geom_type == "LineString":
        lines = [boundary_inter]
    elif boundary_inter.geom_type == "GeometryCollection":
        lines = [g for g in boundary_inter.geoms if g.geom_type == "LineString"]
    else:
        return shared

    for line in lines:
        if line.length > tolerance:
            coords = list(line.coords)
            shared.append((coords[0], coords[-1]))

    return shared


def classify_external_walls(
    space_idx: int, all_polygons: list[Polygon], tolerance: float = 5.0
) -> list[tuple[tuple[tuple[float, float], tuple[float, float]], bool]]:
    """Classify each wall segment of a polygon as external or internal."""
    poly = all_polygons[space_idx]
    coords = list(poly.exterior.coords)
    result = []

    for i in range(len(coords) - 1):
        seg_start = coords[i]
        seg_end = coords[i + 1]
        seg_line = LineString([seg_start, seg_end])

        is_external = True
        for j, other in enumerate(all_polygons):
            if j == space_idx:
                continue
            boundary_inter = seg_line.intersection(other.boundary)
            if not boundary_inter.is_empty and boundary_inter.length > tolerance:
                is_external = False
                break

        result.append(((seg_start, seg_end), is_external))

    return result

# src/floorplan/annotations.py
"""COCO-format annotation generator for floorplans.

Produces semantic masks, instance masks, and COCO JSON annotations
using the same coordinate transform as the renderer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from floorplan.models import Floorplan, Space, SpaceType, OpeningType

# ------------------------------------------------------------------ #
# Semantic class IDs
# ------------------------------------------------------------------ #
# 0 = background
# 1 = wall
# 2 = room
# 3 = corridor
# 4 = bathroom
# 5 = utility
# 6 = door
# 7 = window

SPACE_TYPE_TO_CLASS: dict[SpaceType, int] = {
    SpaceType.ROOM: 2,
    SpaceType.CORRIDOR: 3,
    SpaceType.BATHROOM: 4,
    SpaceType.UTILITY: 5,
}

CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "wall", "supercategory": "structure"},
    {"id": 2, "name": "room", "supercategory": "space"},
    {"id": 3, "name": "corridor", "supercategory": "space"},
    {"id": 4, "name": "bathroom", "supercategory": "space"},
    {"id": 5, "name": "utility", "supercategory": "space"},
    {"id": 6, "name": "door", "supercategory": "opening"},
    {"id": 7, "name": "window", "supercategory": "opening"},
]


class COCOAnnotator:
    """Generates COCO-format annotations for floorplans.

    Uses the same coordinate transform logic as the renderer to ensure
    pixel-perfect alignment between rendered images and annotation masks.
    """

    def __init__(self, image_size: int = 512, margin: float = 0.1) -> None:
        self.image_size = image_size
        self.margin = margin

    # ------------------------------------------------------------------ #
    # Coordinate transform (same logic as FloorplanRenderer)
    # ------------------------------------------------------------------ #

    def _compute_transform(self, floorplan: Floorplan) -> dict[str, float]:
        """Compute scale and offset to fit floorplan in image with margin."""
        size = self.image_size
        margin = self.margin

        all_x: list[float] = []
        all_y: list[float] = []
        for space in floorplan.spaces:
            for pt in space.polygon:
                all_x.append(pt[0])
                all_y.append(pt[1])

        if not all_x:
            return {"scale": 1.0, "offset_x": 0.0, "offset_y": 0.0}

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        extent_x = max_x - min_x
        extent_y = max_y - min_y

        if extent_x < 1e-6:
            extent_x = 1.0
        if extent_y < 1e-6:
            extent_y = 1.0

        usable = size * (1 - 2 * margin)
        scale = min(usable / extent_x, usable / extent_y)

        offset_x = (size - extent_x * scale) / 2 - min_x * scale
        offset_y = (size - extent_y * scale) / 2 - min_y * scale

        return {"scale": scale, "offset_x": offset_x, "offset_y": offset_y}

    def _to_px(self, x: float, y: float, transform: dict[str, float]) -> tuple[float, float]:
        """Convert mm coordinates to pixel coordinates."""
        px = x * transform["scale"] + transform["offset_x"]
        py = y * transform["scale"] + transform["offset_y"]
        return px, py

    # ------------------------------------------------------------------ #
    # Semantic mask
    # ------------------------------------------------------------------ #

    def generate_semantic_mask(self, floorplan: Floorplan) -> Image.Image:
        """Generate a semantic segmentation mask.

        Returns:
            PIL Image in mode "L" (8-bit grayscale) where each pixel value
            is a semantic class ID (0=background, 1=wall, 2=room, etc.).
        """
        size = self.image_size
        mask = Image.new("L", (size, size), 0)  # background = 0
        draw = ImageDraw.Draw(mask)
        transform = self._compute_transform(floorplan)

        # Layer 1: Room polygons (filled with their class ID)
        for space in floorplan.spaces:
            class_id = SPACE_TYPE_TO_CLASS.get(space.type, 2)
            px_points = [self._to_px(pt[0], pt[1], transform) for pt in space.polygon]
            if len(px_points) >= 3:
                draw.polygon(px_points, fill=class_id)

        # Layer 2: Walls on top of rooms
        for space in floorplan.spaces:
            for wall in space.walls:
                p1 = self._to_px(wall.p1[0], wall.p1[1], transform)
                p2 = self._to_px(wall.p2[0], wall.p2[1], transform)
                # Use a reasonable width for wall lines
                wall_width = max(2, int(wall.thickness * transform["scale"]))
                draw.line([p1, p2], fill=1, width=wall_width)

        # Layer 3: Openings on top of walls
        for space in floorplan.spaces:
            for wall in space.walls:
                p1 = np.array(wall.p1)
                p2 = np.array(wall.p2)
                wall_vec = p2 - p1
                wall_len_mm = float(np.linalg.norm(wall_vec))
                if wall_len_mm < 1e-6:
                    continue
                wall_dir = wall_vec / wall_len_mm

                for opening in wall.openings:
                    if opening.type == OpeningType.DOOR:
                        class_id = 6
                    else:
                        class_id = 7

                    center_mm = p1 + wall_vec * opening.offset
                    half_width_mm = opening.width / 2.0
                    start_mm = center_mm - wall_dir * half_width_mm
                    end_mm = center_mm + wall_dir * half_width_mm

                    start_px = self._to_px(float(start_mm[0]), float(start_mm[1]), transform)
                    end_px = self._to_px(float(end_mm[0]), float(end_mm[1]), transform)

                    wall_width = max(2, int(wall.thickness * transform["scale"]))
                    draw.line([start_px, end_px], fill=class_id, width=wall_width)

        return mask

    # ------------------------------------------------------------------ #
    # Instance mask
    # ------------------------------------------------------------------ #

    def generate_instance_mask(self, floorplan: Floorplan) -> Image.Image:
        """Generate an instance segmentation mask.

        Returns:
            PIL Image in mode "I" (32-bit integer) where each pixel value
            is a unique instance ID. 0 = background.
        """
        size = self.image_size
        mask = Image.new("I", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        transform = self._compute_transform(floorplan)

        for idx, space in enumerate(floorplan.spaces, start=1):
            px_points = [self._to_px(pt[0], pt[1], transform) for pt in space.polygon]
            if len(px_points) >= 3:
                draw.polygon(px_points, fill=idx)

        return mask

    # ------------------------------------------------------------------ #
    # COCO JSON annotation
    # ------------------------------------------------------------------ #

    def generate_coco_annotation(
        self,
        floorplan: Floorplan,
        image_id: int,
        file_name: str,
    ) -> dict[str, Any]:
        """Generate COCO-format annotation for a single floorplan image.

        Returns:
            A dict with "image" and "annotations" keys.
        """
        size = self.image_size
        transform = self._compute_transform(floorplan)

        image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": size,
            "height": size,
        }

        annotations: list[dict[str, Any]] = []
        ann_id = 1

        # Annotations for each space (room polygon)
        for space in floorplan.spaces:
            category_id = SPACE_TYPE_TO_CLASS.get(space.type, 2)
            px_points = [self._to_px(pt[0], pt[1], transform) for pt in space.polygon]

            if len(px_points) < 3:
                continue

            # Segmentation: list of [x1, y1, x2, y2, ...] polygons
            segmentation = []
            flat_coords: list[float] = []
            for px, py in px_points:
                flat_coords.append(round(px, 2))
                flat_coords.append(round(py, 2))
            segmentation.append(flat_coords)

            # Bounding box: [x, y, width, height]
            xs = [p[0] for p in px_points]
            ys = [p[1] for p in px_points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox = [
                round(x_min, 2),
                round(y_min, 2),
                round(x_max - x_min, 2),
                round(y_max - y_min, 2),
            ]

            # Area
            try:
                poly = Polygon(px_points)
                area = round(poly.area, 2)
            except Exception:
                area = round(bbox[2] * bbox[3], 2)

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

        # Annotations for walls
        for space in floorplan.spaces:
            for wall in space.walls:
                p1_px = self._to_px(wall.p1[0], wall.p1[1], transform)
                p2_px = self._to_px(wall.p2[0], wall.p2[1], transform)

                # Create a wall polygon from the line with thickness
                p1 = np.array(wall.p1)
                p2 = np.array(wall.p2)
                wall_vec = p2 - p1
                wall_len = float(np.linalg.norm(wall_vec))
                if wall_len < 1e-6:
                    continue

                wall_dir = wall_vec / wall_len
                normal = np.array([-wall_dir[1], wall_dir[0]])

                half_t = wall.thickness / 2.0
                corners_mm = [
                    p1 + normal * half_t,
                    p2 + normal * half_t,
                    p2 - normal * half_t,
                    p1 - normal * half_t,
                ]
                corners_px = [
                    self._to_px(float(c[0]), float(c[1]), transform)
                    for c in corners_mm
                ]

                # Segmentation
                flat_coords = []
                for px, py in corners_px:
                    flat_coords.append(round(px, 2))
                    flat_coords.append(round(py, 2))

                xs = [c[0] for c in corners_px]
                ys = [c[1] for c in corners_px]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = [
                    round(x_min, 2),
                    round(y_min, 2),
                    round(x_max - x_min, 2),
                    round(y_max - y_min, 2),
                ]

                try:
                    poly = Polygon(corners_px)
                    area = round(poly.area, 2)
                except Exception:
                    area = round(bbox[2] * bbox[3], 2)

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,  # wall
                    "segmentation": [flat_coords],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
                ann_id += 1

                # Annotations for openings on this wall
                for opening in wall.openings:
                    if opening.type == OpeningType.DOOR:
                        cat_id = 6
                    else:
                        cat_id = 7

                    center_mm = p1 + wall_vec * opening.offset
                    half_w = opening.width / 2.0
                    start_mm = center_mm - wall_dir * half_w
                    end_mm = center_mm + wall_dir * half_w

                    opening_corners_mm = [
                        start_mm + normal * half_t,
                        end_mm + normal * half_t,
                        end_mm - normal * half_t,
                        start_mm - normal * half_t,
                    ]
                    opening_corners_px = [
                        self._to_px(float(c[0]), float(c[1]), transform)
                        for c in opening_corners_mm
                    ]

                    flat_coords_op = []
                    for px, py in opening_corners_px:
                        flat_coords_op.append(round(px, 2))
                        flat_coords_op.append(round(py, 2))

                    xs_op = [c[0] for c in opening_corners_px]
                    ys_op = [c[1] for c in opening_corners_px]
                    x_min_op, x_max_op = min(xs_op), max(xs_op)
                    y_min_op, y_max_op = min(ys_op), max(ys_op)
                    bbox_op = [
                        round(x_min_op, 2),
                        round(y_min_op, 2),
                        round(x_max_op - x_min_op, 2),
                        round(y_max_op - y_min_op, 2),
                    ]

                    try:
                        poly_op = Polygon(opening_corners_px)
                        area_op = round(poly_op.area, 2)
                    except Exception:
                        area_op = round(bbox_op[2] * bbox_op[3], 2)

                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "segmentation": [flat_coords_op],
                        "bbox": bbox_op,
                        "area": area_op,
                        "iscrowd": 0,
                    })
                    ann_id += 1

        return {
            "image": image_info,
            "annotations": annotations,
        }

    # ------------------------------------------------------------------ #
    # Categories
    # ------------------------------------------------------------------ #

    def get_categories(self) -> list[dict[str, Any]]:
        """Return the list of 7 COCO category dicts."""
        return list(CATEGORIES)

    # ------------------------------------------------------------------ #
    # Full dataset builder
    # ------------------------------------------------------------------ #

    def build_coco_dataset(
        self, image_annotations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Merge per-image annotations into a full COCO dataset JSON.

        Args:
            image_annotations: A list of dicts, each returned by
                ``generate_coco_annotation()``.

        Returns:
            A dict with "images", "annotations", and "categories" keys,
            following the standard COCO dataset format.
        """
        images: list[dict[str, Any]] = []
        all_annotations: list[dict[str, Any]] = []
        ann_id = 1

        for img_ann in image_annotations:
            images.append(img_ann["image"])
            for ann in img_ann["annotations"]:
                # Re-number annotation IDs to be globally unique
                ann_copy = dict(ann)
                ann_copy["id"] = ann_id
                all_annotations.append(ann_copy)
                ann_id += 1

        return {
            "images": images,
            "annotations": all_annotations,
            "categories": list(CATEGORIES),
        }

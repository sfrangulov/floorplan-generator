# src/floorplan/renderer.py
"""Floorplan renderer: converts Floorplan + StyleConfig into a PIL Image."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Point, Polygon

from floorplan.models import Floorplan, Space, Wall, Opening, OpeningType


@dataclass
class RenderConfig:
    """Configuration for rendering a floorplan."""

    image_size: int = 512
    margin: float = 0.1


class FloorplanRenderer:
    """Renders a Floorplan model to a PIL Image using a StyleConfig."""

    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()

    # ------------------------------------------------------------------ #
    # Main render method
    # ------------------------------------------------------------------ #

    def render(self, floorplan: Floorplan, style) -> Image.Image:
        """Render a floorplan with the given style to a PIL Image.

        Args:
            floorplan: The Floorplan model to render.
            style: A StyleConfig instance controlling visual appearance.

        Returns:
            A PIL Image (RGB) of the rendered floorplan.
        """
        size = self.config.image_size
        img = Image.new("RGB", (size, size), style.bg_color)
        draw = ImageDraw.Draw(img)

        transform = self._compute_transform(floorplan)

        # 1. Room fills
        if style.fill_rooms:
            for space in floorplan.spaces:
                self._draw_room_fill(draw, space, style, transform)

        # 2. Walls
        for space in floorplan.spaces:
            for wall in space.walls:
                self._draw_wall(draw, wall, style, transform)
            self._draw_wall_junctions(draw, space, style, transform)

        # 3. Doors and windows (drawn on top of walls)
        # Deduplicate shared walls: two rooms sharing a wall both have their
        # own Wall object with independent openings.  Use a canonical key
        # (sorted endpoints) so each physical wall is only drawn once.
        drawn_walls: set[tuple] = set()
        for space in floorplan.spaces:
            for wall in space.walls:
                if not wall.openings:
                    continue
                key = self._wall_key(wall)
                if key in drawn_walls:
                    continue
                drawn_walls.add(key)
                for opening in wall.openings:
                    if opening.type == OpeningType.DOOR:
                        self._draw_door(draw, wall, opening, style, transform)
                    elif opening.type == OpeningType.WINDOW:
                        self._draw_window(draw, wall, opening, style, transform)

        # 4+5. Shared collision tracking for dimensions and labels
        occupied_rects: list[tuple[float, float, float, float]] = []

        if style.show_dimensions:
            drawn_dim_walls: set[tuple] = set()
            for space in floorplan.spaces:
                drawn_lengths: set[int] = set()
                for wall in space.walls:
                    key = self._wall_key(wall)
                    if key in drawn_dim_walls:
                        continue
                    drawn_dim_walls.add(key)
                    wall_len = int(round(float(np.linalg.norm(
                        np.array(wall.p2) - np.array(wall.p1)
                    ))))
                    if wall_len in drawn_lengths:
                        continue
                    drawn_lengths.add(wall_len)
                    self._draw_dimension(img, draw, wall, transform, style, occupied_rects)

        if style.show_labels:
            for space in floorplan.spaces:
                self._draw_label(draw, space, style, transform, occupied_rects)

        return img

    # ------------------------------------------------------------------ #
    # Coordinate transform
    # ------------------------------------------------------------------ #

    def _compute_transform(self, floorplan: Floorplan) -> dict:
        """Compute scale and offset to fit floorplan in image with margin.

        Returns:
            A dict with keys "scale", "offset_x", "offset_y".
        """
        size = self.config.image_size
        margin = self.config.margin

        # Find bounding box of all spaces
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

        # Prevent division by zero
        if extent_x < 1e-6:
            extent_x = 1.0
        if extent_y < 1e-6:
            extent_y = 1.0

        usable = size * (1 - 2 * margin)
        scale = min(usable / extent_x, usable / extent_y)

        # Center the floorplan in the image
        offset_x = (size - extent_x * scale) / 2 - min_x * scale
        offset_y = (size - extent_y * scale) / 2 - min_y * scale

        return {"scale": scale, "offset_x": offset_x, "offset_y": offset_y}

    # ------------------------------------------------------------------ #
    # Wall deduplication
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wall_key(wall: Wall) -> tuple:
        """Canonical key for a physical wall, independent of endpoint order.

        Rounds coordinates to 1 mm so that reversed copies of a shared wall
        (Room A's p1→p2 vs Room B's p2→p1) produce the same key.
        """
        a = (round(wall.p1[0], 0), round(wall.p1[1], 0))
        b = (round(wall.p2[0], 0), round(wall.p2[1], 0))
        return (min(a, b), max(a, b))

    # ------------------------------------------------------------------ #
    # Coordinate conversion
    # ------------------------------------------------------------------ #

    def _to_px(self, x: float, y: float, transform: dict) -> tuple[float, float]:
        """Convert mm coordinates to pixel coordinates."""
        px = x * transform["scale"] + transform["offset_x"]
        py = y * transform["scale"] + transform["offset_y"]
        return px, py

    # ------------------------------------------------------------------ #
    # Room fill
    # ------------------------------------------------------------------ #

    def _draw_room_fill(self, draw: ImageDraw.ImageDraw, space: Space, style, transform: dict) -> None:
        """Fill a room polygon with its type-based color."""
        fill_color = style.room_fill_colors.get(space.type.value, style.bg_color)
        px_points = [self._to_px(pt[0], pt[1], transform) for pt in space.polygon]
        if len(px_points) >= 3:
            draw.polygon(px_points, fill=fill_color)

    # ------------------------------------------------------------------ #
    # Wall drawing
    # ------------------------------------------------------------------ #

    def _draw_wall(self, draw: ImageDraw.ImageDraw, wall: Wall, style, transform: dict) -> None:
        """Draw a wall as a filled rectangle using wall.thickness."""
        p1 = np.array(self._to_px(wall.p1[0], wall.p1[1], transform))
        p2 = np.array(self._to_px(wall.p2[0], wall.p2[1], transform))

        wall_vec = p2 - p1
        wall_len = float(np.linalg.norm(wall_vec))
        if wall_len < 1e-6:
            return

        # Compute half-thickness in pixels (at least style.line_width / 2)
        half_thick_px = max(wall.thickness * transform["scale"] / 2.0, style.line_width / 2.0)

        # Perpendicular unit vector
        perp = np.array([-wall_vec[1], wall_vec[0]]) / wall_len

        # Four corners of the wall rectangle
        offset = perp * half_thick_px
        corners = [
            tuple(p1 + offset),
            tuple(p2 + offset),
            tuple(p2 - offset),
            tuple(p1 - offset),
        ]
        draw.polygon(corners, fill=style.wall_color)

    def _draw_wall_junctions(self, draw: ImageDraw.ImageDraw, space: Space, style, transform: dict) -> None:
        """Fill corner junctions between consecutive walls to eliminate gaps."""
        walls = space.walls
        if len(walls) < 2:
            return

        for i in range(len(walls)):
            w1 = walls[i]
            w2 = walls[(i + 1) % len(walls)]

            # Find the shared endpoint (w1.p2 should match w2.p1 for consecutive walls)
            # Check both orderings for robustness
            p1_end = np.array(w1.p2)
            p2_start = np.array(w2.p1)
            dist = float(np.linalg.norm(p1_end - p2_start))

            if dist < 1e-3:
                junction_mm = p1_end
            else:
                # Try reversed match
                p2_end = np.array(w2.p2)
                dist2 = float(np.linalg.norm(p1_end - p2_end))
                if dist2 < 1e-3:
                    junction_mm = p1_end
                else:
                    continue

            junction_px = np.array(self._to_px(junction_mm[0], junction_mm[1], transform))
            half_size = max(
                w1.thickness * transform["scale"] / 2.0,
                w2.thickness * transform["scale"] / 2.0,
                style.line_width / 2.0,
            )

            corners = [
                (junction_px[0] - half_size, junction_px[1] - half_size),
                (junction_px[0] + half_size, junction_px[1] - half_size),
                (junction_px[0] + half_size, junction_px[1] + half_size),
                (junction_px[0] - half_size, junction_px[1] + half_size),
            ]
            draw.polygon(corners, fill=style.wall_color)

    # ------------------------------------------------------------------ #
    # Door drawing
    # ------------------------------------------------------------------ #

    def _draw_door(self, draw: ImageDraw.ImageDraw, wall: Wall, opening: Opening, style, transform: dict) -> None:
        """Draw a door opening on a wall.

        Supports three styles: 'arc', 'gap', 'arc_line'.
        """
        p1 = np.array(wall.p1)
        p2 = np.array(wall.p2)
        wall_vec = p2 - p1
        wall_len_mm = float(np.linalg.norm(wall_vec))

        if wall_len_mm < 1e-6:
            return

        # Pixel-space wall length check
        p1_px = np.array(self._to_px(p1[0], p1[1], transform))
        p2_px = np.array(self._to_px(p2[0], p2[1], transform))
        wall_len_px = float(np.linalg.norm(p2_px - p1_px))
        if wall_len_px < 1:
            return

        wall_dir = wall_vec / wall_len_mm
        normal = np.array([-wall_dir[1], wall_dir[0]])

        # Door center in mm coordinates
        center_mm = p1 + wall_vec * opening.offset
        half_width_mm = opening.width / 2.0

        # Door endpoints in mm
        door_start_mm = center_mm - wall_dir * half_width_mm
        door_end_mm = center_mm + wall_dir * half_width_mm

        # Convert to pixel coordinates
        door_start_px = np.array(self._to_px(door_start_mm[0], door_start_mm[1], transform))
        door_end_px = np.array(self._to_px(door_end_mm[0], door_end_mm[1], transform))

        # Clear wall segment behind door
        clear_width = max(int(wall.thickness * transform["scale"]) + 2, style.line_width + 2, 3)
        draw.line(
            [tuple(door_start_px), tuple(door_end_px)],
            fill=style.bg_color,
            width=clear_width,
        )

        if style.door_style == "gap":
            # Gap style: just the cleared wall segment
            return

        # Determine hinge point and swing direction
        if opening.swing == "left":
            hinge_px = door_start_px
            swing_end_px = door_end_px
        else:
            hinge_px = door_end_px
            swing_end_px = door_start_px

        door_width_px = float(np.linalg.norm(door_end_px - door_start_px))
        if door_width_px < 1:
            return

        # Calculate arc bounding box
        # The arc swings from the wall line into the room
        normal_px = np.array(self._to_px(normal[0] + p1[0], normal[1] + p1[1], transform)) - p1_px
        norm_len = float(np.linalg.norm(normal_px))
        if norm_len > 0:
            normal_px = normal_px / norm_len

        # Arc bounding box centered on hinge
        max_radius = max(8.0, self.config.image_size * 0.03)
        radius = min(door_width_px * 0.5, max_radius)
        if radius < 4:
            return
        bbox = [
            hinge_px[0] - radius,
            hinge_px[1] - radius,
            hinge_px[0] + radius,
            hinge_px[1] + radius,
        ]

        # Compute start angle for the arc
        # The arc goes from the wall direction to perpendicular (90 degrees)
        swing_vec = swing_end_px - hinge_px
        swing_angle = math.degrees(math.atan2(-swing_vec[1], swing_vec[0]))

        # Normal direction for the arc end
        perp_vec = np.array([-swing_vec[1], swing_vec[0]])
        perp_angle = math.degrees(math.atan2(-perp_vec[1], perp_vec[0]))

        # Ensure arc goes the correct way (choose the shorter 90-degree arc)
        start_a = swing_angle
        end_a = perp_angle

        # Normalize angles
        while end_a - start_a > 180:
            end_a -= 360
        while end_a - start_a < -180:
            end_a += 360

        if end_a < start_a:
            start_a, end_a = end_a, start_a

        if style.door_style in ("arc", "arc_line"):
            draw.arc(bbox, start=start_a, end=end_a, fill=style.wall_color, width=1)

        if style.door_style == "arc_line":
            # Draw a line from hinge to the perpendicular end of the arc
            arc_end_pt = hinge_px + perp_vec
            # Normalize perp_vec to radius length
            perp_len = float(np.linalg.norm(perp_vec))
            if perp_len > 0:
                arc_end_pt = hinge_px + perp_vec / perp_len * radius
                draw.line(
                    [tuple(hinge_px), tuple(arc_end_pt)],
                    fill=style.wall_color,
                    width=1,
                )

    # ------------------------------------------------------------------ #
    # Window drawing
    # ------------------------------------------------------------------ #

    def _draw_window(self, draw: ImageDraw.ImageDraw, wall: Wall, opening: Opening, style, transform: dict) -> None:
        """Draw a window as 3 parallel lines perpendicular to the wall."""
        p1 = np.array(wall.p1)
        p2 = np.array(wall.p2)
        wall_vec = p2 - p1
        wall_len_mm = float(np.linalg.norm(wall_vec))

        if wall_len_mm < 1e-6:
            return

        wall_dir = wall_vec / wall_len_mm
        normal = np.array([-wall_dir[1], wall_dir[0]])

        # Window center in mm
        center_mm = p1 + wall_vec * opening.offset
        half_width_mm = opening.width / 2.0

        # Window endpoints along wall in mm
        win_start_mm = center_mm - wall_dir * half_width_mm
        win_end_mm = center_mm + wall_dir * half_width_mm

        # Clear the wall behind the window
        win_start_px = np.array(self._to_px(win_start_mm[0], win_start_mm[1], transform))
        win_end_px = np.array(self._to_px(win_end_mm[0], win_end_mm[1], transform))

        clear_width = max(int(wall.thickness * transform["scale"]) + 2, style.line_width + 2, 3)
        draw.line(
            [tuple(win_start_px), tuple(win_end_px)],
            fill=style.bg_color,
            width=clear_width,
        )

        # Perpendicular offset in pixels for the 3 lines
        normal_px_vec = np.array(self._to_px(
            p1[0] + normal[0], p1[1] + normal[1], transform
        )) - np.array(self._to_px(p1[0], p1[1], transform))
        normal_px_len = float(np.linalg.norm(normal_px_vec))
        if normal_px_len > 0:
            normal_px_unit = normal_px_vec / normal_px_len
        else:
            normal_px_unit = np.array([0.0, 1.0])

        offsets = [-2.0, 0.0, 2.0]  # pixels perpendicular to wall
        for off in offsets:
            shift = normal_px_unit * off
            sp = win_start_px + shift
            ep = win_end_px + shift
            draw.line([tuple(sp), tuple(ep)], fill=style.wall_color, width=1)

    # ------------------------------------------------------------------ #
    # Dimension lines
    # ------------------------------------------------------------------ #

    def _draw_dimension(
        self,
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        wall: Wall,
        transform: dict,
        style,
        occupied_rects: list[tuple[float, float, float, float]],
    ) -> None:
        """Draw dimension line with arrows and mm text offset from wall."""
        p1 = np.array(wall.p1)
        p2 = np.array(wall.p2)
        wall_vec = p2 - p1
        wall_len_mm = float(np.linalg.norm(wall_vec))

        if wall_len_mm < 500:
            return  # Skip short walls (too small to label usefully)

        wall_dir = wall_vec / wall_len_mm
        normal = np.array([-wall_dir[1], wall_dir[0]])

        # Offset dimension line from wall by ~12 pixels in mm space
        scale = transform["scale"]
        offset_mm = 12.0 / scale if scale > 0 else 12.0

        dim_p1_mm = p1 + normal * offset_mm
        dim_p2_mm = p2 + normal * offset_mm

        dim_p1_px = self._to_px(dim_p1_mm[0], dim_p1_mm[1], transform)
        dim_p2_px = self._to_px(dim_p2_mm[0], dim_p2_mm[1], transform)

        dim_color = style.wall_color

        # Draw dimension line
        draw.line([dim_p1_px, dim_p2_px], fill=dim_color, width=1)

        # Tick marks at ends (perpendicular to dimension line, ~4px)
        tick_len = 4.0
        tick_mm = tick_len / scale if scale > 0 else tick_len

        for pt_mm, pt_px in [(dim_p1_mm, dim_p1_px), (dim_p2_mm, dim_p2_px)]:
            tick_start_mm = pt_mm - normal * tick_mm
            tick_end_mm = pt_mm + normal * tick_mm
            tick_start_px = self._to_px(tick_start_mm[0], tick_start_mm[1], transform)
            tick_end_px = self._to_px(tick_end_mm[0], tick_end_mm[1], transform)
            draw.line([tick_start_px, tick_end_px], fill=dim_color, width=1)

        # Text at midpoint
        mid_px = (
            (dim_p1_px[0] + dim_p2_px[0]) / 2,
            (dim_p1_px[1] + dim_p2_px[1]) / 2,
        )
        text = f"{int(round(wall_len_mm))}"

        try:
            font = ImageFont.truetype("Arial", 10)
        except (OSError, IOError):
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

        if font is not None:
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = len(text) * 6, 10

            # Compute wall angle for text rotation
            angle_rad = math.atan2(
                -(dim_p2_px[1] - dim_p1_px[1]),
                dim_p2_px[0] - dim_p1_px[0],
            )
            angle_deg = math.degrees(angle_rad)
            # Keep text readable (never upside-down)
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180

            # Render text on temporary RGBA image, rotate, paste
            pad = 4
            tmp_w = tw + 2 * pad
            tmp_h = th + 2 * pad
            txt_img = Image.new("RGBA", (tmp_w, tmp_h), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((pad, pad), text, fill=dim_color, font=font)
            rotated = txt_img.rotate(angle_deg, expand=True, resample=Image.BICUBIC)

            rw, rh = rotated.size
            paste_x = int(mid_px[0] - rw / 2)
            paste_y = int(mid_px[1] - rh / 2)

            text_bbox = (paste_x - 3, paste_y - 3, paste_x + rw + 3, paste_y + rh + 3)
            if self._bbox_overlaps(text_bbox, occupied_rects):
                return
            occupied_rects.append(text_bbox)

            # Paste onto main image (convert to RGB paste if needed)
            if img.mode == "RGB":
                # Create RGB version + use alpha as mask
                img.paste(rotated.convert("RGB"), (paste_x, paste_y), rotated.split()[3])
            else:
                img.paste(rotated, (paste_x, paste_y), rotated)

    # ------------------------------------------------------------------ #
    # Room labels
    # ------------------------------------------------------------------ #

    def _draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        space: Space,
        style,
        transform: dict,
        occupied_rects: list[tuple[float, float, float, float]],
    ) -> None:
        """Draw room type label inside the polygon with adaptive sizing and overlap avoidance."""
        # Build polygon in mm space
        try:
            poly_mm = Polygon([(pt[0], pt[1]) for pt in space.polygon])
            if poly_mm.is_empty or not poly_mm.is_valid:
                return
            rep = poly_mm.representative_point()
            cx, cy = rep.x, rep.y
        except Exception:
            xs = [pt[0] for pt in space.polygon]
            ys = [pt[1] for pt in space.polygon]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            poly_mm = None

        # Build pixel-space polygon for containment checks
        px_pts = [self._to_px(pt[0], pt[1], transform) for pt in space.polygon]
        try:
            poly_px = Polygon(px_pts)
            if poly_px.is_empty or not poly_px.is_valid:
                poly_px = None
        except Exception:
            poly_px = None

        px, py = self._to_px(cx, cy, transform)
        label = space.type.value.replace("_", " ").upper()

        # Compute available space from polygon bounds
        if poly_px is not None:
            minx, miny, maxx, maxy = poly_px.bounds
            avail_w = maxx - minx - 8
            avail_h = maxy - miny - 8
        else:
            avail_w, avail_h = 200, 200

        if avail_w < 20 or avail_h < 8:
            return  # room too small for any label

        # Determine starting font size from room area
        try:
            area_mm2 = poly_mm.area if poly_mm else 1e6
        except Exception:
            area_mm2 = 1e6
        area_px2 = area_mm2 * (transform["scale"] ** 2)
        font_size = int(math.sqrt(area_px2) * 0.12)
        font_size = max(6, min(font_size, 16))

        # Shrink font until text fits available space
        font = None
        tw, th = 0, 0
        while font_size >= 6:
            try:
                font = ImageFont.truetype("Arial", font_size)
            except (OSError, IOError):
                try:
                    font = ImageFont.load_default()
                except Exception:
                    return
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = len(label) * (font_size // 2), font_size + 2
            if tw <= avail_w and th <= avail_h:
                break
            font_size -= 1
        else:
            return  # can't fit even at min size

        if font is None:
            return

        text_x = px - tw / 2
        text_y = py - th / 2

        # Try placement at center, then 8 nudge directions
        offsets = [
            (0, 0),
            (0, th + 4), (0, -(th + 4)),
            (tw + 4, 0), (-(tw + 4), 0),
            (tw, th + 2), (-tw, th + 2),
            (tw, -(th + 2)), (-tw, -(th + 2)),
        ]
        placed = False
        for dx, dy in offsets:
            cx_new = text_x + dx
            cy_new = text_y + dy
            candidate = (cx_new - 3, cy_new - 3, cx_new + tw + 3, cy_new + th + 3)
            inside = poly_px is None or all(
                poly_px.contains(Point(c))
                for c in [(cx_new, cy_new), (cx_new + tw, cy_new),
                           (cx_new, cy_new + th), (cx_new + tw, cy_new + th)]
            )
            if inside and not self._bbox_overlaps(candidate, occupied_rects):
                text_x = cx_new
                text_y = cy_new
                placed = True
                break

        if not placed:
            return  # skip entirely rather than clip or overlap

        label_bbox = (text_x - 3, text_y - 3, text_x + tw + 3, text_y + th + 3)
        occupied_rects.append(label_bbox)
        draw.text((text_x, text_y), label, fill=style.wall_color, font=font)

    @staticmethod
    def _bbox_overlaps(
        bbox: tuple[float, float, float, float],
        others: list[tuple[float, float, float, float]],
    ) -> bool:
        """Check if bbox overlaps with any bbox in the list."""
        x1, y1, x2, y2 = bbox
        for ox1, oy1, ox2, oy2 in others:
            if x1 < ox2 and x2 > ox1 and y1 < oy2 and y2 > oy1:
                return True
        return False

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, Point


# Maps filename prefixes to the room types where they can appear.
# A prefix matches if the icon filename starts with it.
_ICON_ROOM_TYPES: dict[str, list[str]] = {
    "bed_double": ["room"],
    "sofa": ["room"],
    "dining_table": ["room"],
    "armchair": ["room"],
    "desk": ["room", "utility"],
    "rug": ["room"],
    "curtain": ["room"],
    "ottoman": ["room", "bathroom"],
    "plant": ["room", "corridor", "bathroom", "utility"],
    "sideboard": ["room", "corridor"],
    "sink": ["bathroom", "utility"],
    "cabinet": ["utility", "bathroom"],
}


class FurniturePlacer:
    def __init__(self, furniture_dir: str | Path | None = None):
        self.icons: list[Image.Image] = []
        self._icon_names: list[str] = []
        # Icons grouped by room type for fast lookup
        self._icons_by_room: dict[str, list[int]] = {}

        if furniture_dir is not None:
            p = Path(furniture_dir)
            if p.exists():
                for f in sorted(p.glob("*.png")):
                    try:
                        img = Image.open(f).convert("RGBA")
                    except Exception:
                        continue
                    idx = len(self.icons)
                    self.icons.append(img)
                    self._icon_names.append(f.stem)

                    # Categorize by room type using prefix mapping
                    matched = False
                    for prefix, room_types in _ICON_ROOM_TYPES.items():
                        if f.stem.startswith(prefix):
                            for rt in room_types:
                                self._icons_by_room.setdefault(rt, []).append(idx)
                            matched = True
                            break
                    if not matched:
                        # Unknown prefix → available in all room types
                        for rt in ("room", "corridor", "bathroom", "utility"):
                            self._icons_by_room.setdefault(rt, []).append(idx)

    def place_furniture(
        self, draw: ImageDraw.Draw, room_poly: Polygon,
        room_type: str, rng: np.random.Generator,
        density: float, transform: dict,
    ) -> None:
        if density <= 0:
            return

        max_items = max(1, int(density * 5))
        n_items = rng.integers(1, max_items + 1)

        for _ in range(n_items):
            candidates = self._icons_by_room.get(room_type, [])
            if candidates and rng.random() > 0.5:
                self._place_icon(draw, room_poly, room_type, rng, transform)
            else:
                self._place_programmatic(draw, room_poly, room_type, rng, transform)

    def _place_programmatic(
        self, draw: ImageDraw.Draw, room_poly: Polygon,
        room_type: str, rng: np.random.Generator, transform: dict,
    ) -> None:
        minx, miny, maxx, maxy = room_poly.bounds
        pad = (maxx - minx) * 0.15
        inner_minx = minx + pad
        inner_maxx = maxx - pad
        inner_miny = miny + pad
        inner_maxy = maxy - pad
        if inner_maxx <= inner_minx or inner_maxy <= inner_miny:
            return

        for _ in range(10):
            cx = rng.uniform(inner_minx, inner_maxx)
            cy = rng.uniform(inner_miny, inner_maxy)

            if not room_poly.contains(Point(cx, cy)):
                continue

            scale = transform["scale"]
            ox, oy = transform["offset_x"], transform["offset_y"]
            px_cx = cx * scale + ox
            px_cy = cy * scale + oy

            shape = rng.choice(["rect", "circle", "l_shape"])
            color = (180, 180, 180)
            size = rng.uniform(8, 20)

            if shape == "rect":
                w, h = size, size * rng.uniform(0.5, 1.5)
                draw.rectangle(
                    [px_cx - w, px_cy - h, px_cx + w, px_cy + h],
                    outline=color, width=1
                )
            elif shape == "circle":
                r = size * 0.6
                draw.ellipse(
                    [px_cx - r, px_cy - r, px_cx + r, px_cy + r],
                    outline=color, width=1
                )
            else:  # l_shape
                w, h = size, size
                draw.rectangle(
                    [px_cx - w, px_cy - h, px_cx + w, px_cy],
                    outline=color, width=1
                )
                draw.rectangle(
                    [px_cx - w, px_cy, px_cx, px_cy + h],
                    outline=color, width=1
                )
            break

    def _place_icon(
        self, draw: ImageDraw.Draw, room_poly: Polygon,
        room_type: str, rng: np.random.Generator, transform: dict,
    ) -> None:
        candidates = self._icons_by_room.get(room_type, [])
        if not candidates:
            return

        icon_idx = candidates[int(rng.integers(0, len(candidates)))]
        icon = self.icons[icon_idx]

        minx, miny, maxx, maxy = room_poly.bounds
        pad_x = (maxx - minx) * 0.2
        pad_y = (maxy - miny) * 0.2
        if maxx - minx - 2 * pad_x <= 0 or maxy - miny - 2 * pad_y <= 0:
            return

        # Try a few placements to find one inside the polygon
        for _ in range(10):
            cx = rng.uniform(minx + pad_x, maxx - pad_x)
            cy = rng.uniform(miny + pad_y, maxy - pad_y)
            if room_poly.contains(Point(cx, cy)):
                break
        else:
            return

        scale = transform["scale"]

        # Scale icon proportionally to room size (longer side of room bounds)
        room_px_w = (maxx - minx) * scale
        room_px_h = (maxy - miny) * scale
        room_size = max(room_px_w, room_px_h)
        # Target: icon covers 20-40% of the shorter room dimension
        target_size = int(min(room_px_w, room_px_h) * rng.uniform(0.20, 0.40))
        target_size = max(12, min(target_size, 80))

        # Resize preserving aspect ratio
        orig_w, orig_h = icon.size
        ratio = target_size / max(orig_w, orig_h)
        new_w = max(1, int(orig_w * ratio))
        new_h = max(1, int(orig_h * ratio))
        resized = icon.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Orthogonal rotation only (0, 90, 180, 270)
        angle = int(rng.choice([0, 90, 180, 270]))
        if angle != 0:
            resized = resized.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

        px_cx = int(cx * scale + transform["offset_x"])
        px_cy = int(cy * scale + transform["offset_y"])

        img = draw._image
        paste_x = px_cx - resized.width // 2
        paste_y = px_cy - resized.height // 2
        try:
            img.paste(resized, (paste_x, paste_y), resized)
        except Exception:
            pass

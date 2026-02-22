from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, Point


class FurniturePlacer:
    def __init__(self, furniture_dir: str | Path | None = None):
        self.icons: list[Image.Image] = []
        if furniture_dir is not None:
            p = Path(furniture_dir)
            if p.exists():
                for f in sorted(p.glob("*.png")):
                    try:
                        self.icons.append(Image.open(f).convert("RGBA"))
                    except Exception:
                        pass

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
            if self.icons and rng.random() > 0.5:
                self._place_icon(draw, room_poly, rng, transform)
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
        rng: np.random.Generator, transform: dict,
    ) -> None:
        if not self.icons:
            return
        icon = self.icons[int(rng.integers(0, len(self.icons)))]

        minx, miny, maxx, maxy = room_poly.bounds
        pad = (maxx - minx) * 0.2
        cx = rng.uniform(minx + pad, maxx - pad)
        cy = rng.uniform(miny + pad, maxy - pad)

        if not room_poly.contains(Point(cx, cy)):
            return

        scale = transform["scale"]
        target_size = int(rng.uniform(15, 30))
        resized = icon.resize((target_size, target_size), Image.Resampling.LANCZOS)
        angle = float(rng.uniform(0, 360))
        rotated = resized.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

        px_cx = int(cx * scale + transform["offset_x"])
        px_cy = int(cy * scale + transform["offset_y"])

        img = draw._image
        paste_x = px_cx - rotated.width // 2
        paste_y = px_cy - rotated.height // 2
        try:
            img.paste(rotated, (paste_x, paste_y), rotated)
        except Exception:
            pass

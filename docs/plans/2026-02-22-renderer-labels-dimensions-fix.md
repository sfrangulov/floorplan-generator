# Renderer Labels & Dimensions Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix label clipping, dimension text overlap, and add rotated dimension text in the floorplan renderer.

**Architecture:** All changes in `src/floorplan/renderer.py`. Labels get polygon-aware placement (Shapely containment check). Dimension text gets rotated to match wall orientation via temporary RGBA image + paste. Background rectangles behind dimension text removed.

**Tech Stack:** PIL/Pillow (Image, ImageDraw, ImageFont), Shapely (Polygon, Point), numpy

---

### Task 1: Remove dimension background rectangle & increase padding

**Files:**
- Modify: `src/floorplan/renderer.py:505-510`

**Step 1: Edit `_draw_dimension` — remove background rect, increase padding**

In `src/floorplan/renderer.py`, replace lines 505-510:

```python
            text_bbox = (text_x - 1, text_y - 1, text_x + tw + 1, text_y + th + 1)
            if self._bbox_overlaps(text_bbox, occupied_rects):
                return  # skip text, don't clutter
            occupied_rects.append(text_bbox)
            draw.rectangle(text_bbox, fill=style.bg_color)
            draw.text((text_x, text_y), text, fill=dim_color, font=font)
```

with:

```python
            text_bbox = (text_x - 3, text_y - 3, text_x + tw + 3, text_y + th + 3)
            if self._bbox_overlaps(text_bbox, occupied_rects):
                return  # skip text, don't clutter
            occupied_rects.append(text_bbox)
            draw.text((text_x, text_y), text, fill=dim_color, font=font)
```

Changes: padding 1→3, removed `draw.rectangle` line.

**Step 2: Run tests**

Run: `uv run pytest tests/test_renderer.py -v`
Expected: All 4 tests PASS

**Step 3: Generate images and visually verify**

Run:
```bash
rm -rf output/ && uv run python scripts/floorplan_generator.py --count 3 --seed 200 --layout-type random --output-dir output/json && uv run python scripts/generate_synthetic.py --input-dir output/json --output-dir output --image-size 512 --furniture-dir furniture_icons
```
Expected: Dimension text has no white/cream rectangle behind it. More spacing between adjacent dimension texts.

---

### Task 2: Rotate dimension text along wall orientation

**Files:**
- Modify: `src/floorplan/renderer.py:435-510` (the `_draw_dimension` method)

**Step 1: Replace the text-drawing section of `_draw_dimension`**

Replace everything from `# Text at midpoint` (line 481) to end of method with rotated text logic:

```python
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

            # Compute wall angle in degrees
            angle_rad = math.atan2(
                -(dim_p2_px[1] - dim_p1_px[1]),
                dim_p2_px[0] - dim_p1_px[0],
            )
            angle_deg = math.degrees(angle_rad)
            # Normalize so text is never upside-down (keep between -90 and +90)
            if angle_deg > 90:
                angle_deg -= 180
            elif angle_deg < -90:
                angle_deg += 180

            # Render text onto a small temporary RGBA image, then rotate
            pad = 4
            tmp_w = tw + 2 * pad
            tmp_h = th + 2 * pad
            txt_img = Image.new("RGBA", (tmp_w, tmp_h), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((pad, pad), text, fill=dim_color, font=font)
            rotated = txt_img.rotate(angle_deg, expand=True, resample=Image.BICUBIC)

            # Compute paste position (centered on midpoint)
            rw, rh = rotated.size
            paste_x = int(mid_px[0] - rw / 2)
            paste_y = int(mid_px[1] - rh / 2)

            # Collision check using rotated bounding box
            text_bbox = (
                paste_x - 3, paste_y - 3,
                paste_x + rw + 3, paste_y + rh + 3,
            )
            if self._bbox_overlaps(text_bbox, occupied_rects):
                return
            occupied_rects.append(text_bbox)

            # Paste rotated text onto main image
            img = draw.im  # underlying PIL image — NOT reliable
            # Instead, we need access to the image. Pass `img` to the method.
```

**IMPORTANT:** The method needs access to the PIL `Image` object to paste the rotated text. `ImageDraw` doesn't expose it reliably. We must pass `img` as a parameter.

Update the method signature:

```python
def _draw_dimension(
    self,
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    wall: Wall,
    transform: dict,
    style,
    occupied_rects: list[tuple[float, float, float, float]],
) -> None:
```

And the call site in `render()` (line 97):

```python
self._draw_dimension(img, draw, wall, transform, style, occupied_rects)
```

The paste at end of method becomes:

```python
            img.paste(rotated, (paste_x, paste_y), rotated)
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_renderer.py -v`
Expected: All 4 tests PASS

**Step 3: Generate images and visually verify**

Run:
```bash
rm -rf output/ && uv run python scripts/floorplan_generator.py --count 5 --seed 200 --layout-type random --output-dir output/json && uv run python scripts/generate_synthetic.py --input-dir output/json --output-dir output --image-size 512 --furniture-dir furniture_icons
```
Expected: Vertical wall dimensions read vertically. Horizontal wall dimensions remain horizontal. No upside-down text.

---

### Task 3: Labels strictly inside room polygon

**Files:**
- Modify: `src/floorplan/renderer.py:516-609` (the `_draw_label` method)

**Step 1: Add `from shapely.geometry import Point` import**

At top of file, line 10, change:

```python
from shapely.geometry import Polygon
```

to:

```python
from shapely.geometry import Point, Polygon
```

**Step 2: Rewrite `_draw_label` with polygon containment**

Replace the entire `_draw_label` method with:

```python
def _draw_label(
    self,
    draw: ImageDraw.ImageDraw,
    space: Space,
    style,
    transform: dict,
    occupied_rects: list[tuple[float, float, float, float]],
) -> None:
    """Draw room type label inside the polygon with adaptive sizing and overlap avoidance."""
    # Build polygon in mm and px space
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
        avail_w = maxx - minx - 8  # 4px padding each side
        avail_h = maxy - miny - 8
    else:
        avail_w, avail_h = 200, 200  # fallback

    # Skip very small rooms
    if avail_w < 20 or avail_h < 8:
        return

    # Find font size that fits
    try:
        area_mm2 = poly_mm.area if poly_mm else 1e6
    except Exception:
        area_mm2 = 1e6
    area_px2 = area_mm2 * (transform["scale"] ** 2)
    font_size = int(math.sqrt(area_px2) * 0.12)
    font_size = max(6, min(font_size, 16))

    font = None
    tw, th = 0, 0
    # Shrink font until text fits available space
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
        return  # can't fit even at size 6

    if font is None:
        return

    text_x = px - tw / 2
    text_y = py - th / 2

    def _inside_polygon(tx, ty):
        """Check all 4 corners of text bbox are inside room polygon."""
        if poly_px is None:
            return True
        for corner in [(tx, ty), (tx + tw, ty), (tx, ty + th), (tx + tw, ty + th)]:
            if not poly_px.contains(Point(corner)):
                return False
        return True

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
        if not self._bbox_overlaps(candidate, occupied_rects) and _inside_polygon(cx_new, cy_new):
            text_x = cx_new
            text_y = cy_new
            placed = True
            break

    if not placed:
        return  # skip entirely rather than clip or overlap

    label_bbox = (text_x - 3, text_y - 3, text_x + tw + 3, text_y + th + 3)
    occupied_rects.append(label_bbox)
    draw.text((text_x, text_y), label, fill=style.wall_color, font=font)
```

**Step 3: Run tests**

Run: `uv run pytest tests/ -v`
Expected: All 76 tests PASS

**Step 4: Generate images and visually verify**

Run:
```bash
rm -rf output/ && uv run python scripts/floorplan_generator.py --count 10 --seed 200 --layout-type random --output-dir output/json && uv run python scripts/generate_synthetic.py --input-dir output/json --output-dir output --image-size 512 --furniture-dir furniture_icons
```
Expected: No labels clipped by walls. Small rooms have smaller text or no label. Labels don't overlap dimensions or each other.

---

### Task 4: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All 76 tests PASS

**Step 2: Generate 10 images with different seed and verify**

Run:
```bash
rm -rf output/ && uv run python scripts/floorplan_generator.py --count 10 --seed 42 --layout-type random --output-dir output/json && uv run python scripts/generate_synthetic.py --input-dir output/json --output-dir output --image-size 512 --furniture-dir furniture_icons
```

Check all 10 images for:
- [ ] No labels clipped by walls
- [ ] No dimension text background rectangles
- [ ] Vertical dimensions read vertically
- [ ] No overlapping text (labels vs labels, labels vs dimensions)
- [ ] Small rooms: label shrunk or skipped, not clipped

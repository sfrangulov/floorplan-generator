# Fix Renderer Labels & Dimensions — Design

All changes in `src/floorplan/renderer.py`.

## 1. Labels strictly inside room polygon

In `_draw_label`:
- Convert room polygon to pixel coordinates → Shapely `Polygon` in px space
- Compute available width/height from polygon bounds minus padding
- If text with current font exceeds available space → shrink font stepwise (min 6px)
- Before drawing, verify all 4 corners of text bbox lie inside `poly_px`
- If no font size fits → skip label entirely
- Add +3px padding to `occupied_rects` entries for spacing between elements

## 2. Increase padding in dimension collision tracking

In `_draw_dimension`:
- Increase text_bbox margin from 1px to 3px for better spacing between dimensions and labels

## 3. Remove background rectangle behind dimension text

In `_draw_dimension`:
- Remove `draw.rectangle(text_bbox, fill=style.bg_color)` — no opaque background behind numbers

## 4. Rotate dimension text along wall orientation

In `_draw_dimension`:
- Compute wall angle from wall vector
- Render text onto a small temporary RGBA image
- Rotate by wall angle using `Image.rotate()`
- Paste rotated text onto main image at midpoint position
- This makes vertical wall dimensions read vertically, matching architectural convention

## 5. Draw order (unchanged)

fills → walls → doors/windows → dimensions → labels

Dimensions are offset 12px from wall, so window lines don't intersect. Collision tracking handles edge cases.

# Floorplan Generator — Design Document

System for generating synthetic floorplans and rendering them as images for CV dataset creation (segmentation / detection / parsing).

## Architecture

Pipeline approach with two CLI entry points sharing a common library package.

### Project Structure

```
floorplan-generator/
├── pyproject.toml                # UV, Python 3.11+
├── src/
│   └── floorplan/
│       ├── __init__.py
│       ├── generator.py          # FloorplanGenerator — JSON generation
│       ├── renderer.py           # FloorplanRenderer — image rendering
│       ├── annotations.py        # COCOAnnotator — masks + bbox + COCO JSON
│       ├── geometry.py           # Utilities: intersections, snapping, external walls
│       ├── furniture.py          # FurniturePlacer — PNG + programmatic fallback
│       ├── models.py             # Pydantic models (Space, Wall, Opening, Floorplan)
│       └── styles.py             # Palettes, line styles, visual randomization
├── scripts/
│   ├── floorplan_generator.py    # CLI: batch JSON generation
│   └── generate_synthetic.py     # CLI: batch image rendering + annotations
├── furniture_icons/              # PNG furniture icons (optional)
├── output/
│   ├── json/
│   ├── images/
│   ├── masks/
│   └── annotations/
└── tests/
```

### Dependencies

- numpy, Pillow, shapely — core computation and rendering
- pydantic — JSON schema validation between generator and renderer
- click — CLI interface
- tqdm — progress bars for batch generation

## Floorplan Generation Algorithm

### Units

All dimensions in millimeters. Conversion to pixels happens at render time based on target image size.

### Configurable Parameters

| Parameter | Description |
|-----------|-------------|
| `num_rooms` | Number of main rooms (N) |
| `min_room_size`, `max_room_size` | Room dimension ranges (mm) |
| `indent_prob` | Probability of wall indentation |
| `indent_depth_range`, `indent_width_range` | Indentation parameters |
| `global_wall_thickness` | Base wall thickness (mm) |
| `wall_thickness_variation_prob` | Probability of thicker wall |
| `wall_thickness_multiplier_range` | Thickness multiplier range |
| `door_prob` | Door probability between adjacent rooms |
| `window_prob` | Window probability on external walls |
| `max_extent` | Maximum building bounding box size |
| Room type probabilities | room, corridor, bathroom, utility |

### Growing Structure Algorithm

1. Generate first room as rectangle (possibly with indentation) at origin.
2. Iteratively attach new rooms:
   - Select existing room (weighted: prefer rooms with more free external walls; corridors get higher weight to become hubs)
   - Select random external wall of that room
   - Determine new room type by probability weights (bathroom/utility are smaller)
   - Generate dimensions (corridors: higher aspect ratio)
   - Attach to selected wall, aligned at random position along wall length
   - Validate with Shapely: `new_polygon.intersects(existing)` — reject and retry (up to 50 attempts)
   - Snap vertices to grid to prevent micro-gaps between rooms
   - Enforce `max_extent` bounding box constraint
3. Post-processing:
   - Classify walls as external/internal (external = no adjacent room on the other side)
   - Place doors on shared walls (probability `door_prob`, minimum 1 door per room)
   - **Connectivity guarantee**: BFS over adjacency graph; if disconnected, add doors between connected components
   - Place windows on external walls (probability `window_prob`)
   - Apply wall thickness variations

### Room Geometry

- Base shape: rectangle
- With probability `indent_prob`: cut rectangular niche from one wall (L-shaped or notched polygon, 6-8 vertices)
- Corridors: aspect ratio >= 3:1

### Placement Constraints

- Bathrooms attach only to corridors or rooms (not to other bathrooms)
- Corridors act as hubs (higher weight for attachment selection)
- Snap tolerance for wall alignment using Shapely `snap()`

### JSON Output Format

```json
{
  "meta": {
    "seed": 123,
    "global_wall_thickness": 120,
    "units": "mm"
  },
  "spaces": [
    {
      "id": "space_001",
      "type": "room",
      "polygon": [[x,y], [x,y], ...],
      "walls": [
        {
          "p1": [x,y],
          "p2": [x,y],
          "thickness": 120,
          "is_external": true,
          "openings": [
            {"type": "door", "width": 900, "offset": 0.3, "swing": "left"},
            {"type": "window", "width": 1200, "offset": 0.6}
          ]
        }
      ]
    }
  ]
}
```

## Rendering

### Style Randomization

Each image gets a randomly generated `StyleConfig`:
- Line thickness: 1-4px
- Wall color: palette (black, dark gray, blue, brown)
- Background: white, cream, light gray
- Room fill: on/off + pastel colors by room type
- Door style: arc, gap, or arc+line (randomly selected)
- Window style: 3 parallel lines across wall
- Dimension lines: optionally added to external walls (arrows + mm values)
- Room labels: "Room", "Corridor", "WC", "Utility" — optional, random font

### Furniture Placement

`FurniturePlacer` checks for `furniture_icons/` directory:
- If PNGs available: load, scale, rotate, validate placement inside polygon via Shapely
- Fallback: programmatic shapes (rectangle=table/bed, circle=chair, L-shape=sofa, rectangle+circle=sink for bathroom)
- 0-5 items per room, based on area and type

## Annotations (COCO Format)

Generated alongside rendering:

### Segmentation Masks
- PNG, single channel, pixel value = class_id
- Classes: 0=background, 1=wall, 2=room, 3=corridor, 4=bathroom, 5=utility, 6=door, 7=window

### Instance Masks
- Each room instance gets unique id

### COCO JSON
- Standard format: `images`, `annotations` (polygons + bbox), `categories`
- Single COCO JSON per dataset

## CLI Interface

### Step 1: Generate JSON descriptions
```bash
python scripts/floorplan_generator.py \
  --count 5000 --seed 42 \
  --num-rooms 3-8 \
  --output-dir output/json/
```

### Step 2: Render images + annotations
```bash
python scripts/generate_synthetic.py \
  --input-dir output/json/ \
  --output-dir output/ \
  --image-size 512 \
  --workers 4 \
  --with-masks --with-coco
```

### Parallelization
- `multiprocessing.Pool(workers)` for batch generation
- Each worker gets deterministic seed: `base_seed + index`
- Progress tracking via tqdm

# Floorplan Generator

Synthetic floorplan generation system for ML datasets (COCO format) and visual output. Produces realistic apartment and house layouts based on Russian building standards (SNiP), with two-stage pipeline: JSON structure generation followed by image rendering with annotations.

## Features

- **8 layout types**: studio, 1-4 room apartments, small/medium/large houses
- **12 room types**: hallway, corridor, living room, bedroom, kitchen, bathroom, toilet, balcony, storage, utility, garage, terrace
- **Architectural rules**: room adjacency constraints, SNiP-based dimensions, door/window placement rules
- **COCO annotations**: semantic masks, instance masks, bounding boxes (15 categories)
- **Furniture placement**: PNG icons with room-type-aware mapping + programmatic fallback
- **Reproducible**: seeded random generation for deterministic output
- **Batch processing**: CLI tools for generating thousands of samples

## Quick Start

```bash
# Install dependencies
uv sync

# Generate 10 floorplan JSONs with random layout types
uv run python scripts/floorplan_generator.py \
    --count 10 --seed 42 --layout-type random \
    --output-dir output/json

# Render images + masks + COCO annotations
uv run python scripts/generate_synthetic.py \
    --input-dir output/json \
    --output-dir output \
    --image-size 512 \
    --with-masks --with-coco \
    --furniture-dir furniture_icons
```

Output structure:
```
output/
  images/           # PNG floorplan images (512x512)
  masks/            # Semantic segmentation masks (L mode)
  annotations/
    coco.json       # COCO format dataset
```

## Layout Types

| Type | Rooms | Required | Optional |
|------|-------|----------|----------|
| studio | 3 | hallway, bathroom, living_room | balcony |
| 1room | 4 | hallway, bathroom, kitchen, living_room | balcony, storage |
| 2room | 6 | hallway, bathroom, toilet, kitchen, living_room, bedroom | balcony, storage, corridor |
| 3room | 7 | + 2 bedrooms | balcony, storage, corridor |
| 4room | 8 | + 3 bedrooms | 2x balcony, storage, corridor |
| house_small | 7 | hallway, bathroom, toilet, kitchen, living_room, 2x bedroom | terrace, storage, utility |
| house_medium | 9 | + corridor, 3x bedroom | terrace, garage, storage, utility |
| house_large | 10 | + 2x bathroom | terrace, garage, 2x storage, utility |

## CLI Reference

### floorplan_generator.py

Generates floorplan JSON descriptions.

```
--count INTEGER        Number of floorplans (required)
--seed INTEGER         Base random seed (default: 42)
--num-rooms TEXT       Room count range, e.g. "3-8" (default: "3-8")
--layout-type TEXT     Layout type: studio|1room|2room|3room|4room|
                       house_small|house_medium|house_large|random
                       (overrides --num-rooms)
--output-dir PATH      Output directory (required)
--indent-prob FLOAT    L-shaped room probability (default: 0.2)
--door-prob FLOAT      Door placement probability (default: 0.7)
--window-prob FLOAT    Window placement probability (default: 0.5)
--wall-thickness FLOAT Wall thickness in mm (default: 120)
```

### generate_synthetic.py

Renders floorplan images from JSON + optional annotations.

```
--input-dir PATH       Directory with floorplan JSONs (required)
--output-dir PATH      Output directory (required)
--image-size INTEGER   Image resolution in pixels (default: 512)
--workers INTEGER      Parallel workers (default: 1)
--with-masks           Generate semantic segmentation masks
--with-coco            Generate COCO format annotations
--furniture-dir PATH   Path to furniture PNG icons
```

## Architecture

```
src/floorplan/
  generator.py        # Growing-structure algorithm + layout-aware generation
  renderer.py         # PIL-based image rendering (walls, doors, windows, labels)
  annotations.py      # COCO annotation generator (15 categories)
  models.py           # Pydantic models (SpaceType, Wall, Opening, Floorplan)
  geometry.py         # Shapely-based geometric operations
  layout_templates.py # Room specs (SNiP dimensions) + 8 layout templates
  adjacency.py        # Room connection rules (forbidden pairs, allowed parents)
  furniture.py        # Room-type-aware furniture icon placement
  styles.py           # Visual style randomization (colors, line widths)
```

### Pipeline

```
GeneratorConfig → FloorplanGenerator.generate() → Floorplan (JSON)
                                                       ↓
                              FloorplanRenderer.render() → PIL Image
                              COCOAnnotator.generate_*() → Masks / COCO JSON
                              FurniturePlacer.place_*()  → Furniture overlay
```

### Generation Algorithm

1. Place hallway at origin (always first room)
2. Attach rooms in architectural order using `ALLOWED_PARENTS` rules
3. Use `ROOM_SPECS` for SNiP-based dimensions per room type
4. Validate with `NEVER_CONNECT` forbidden pairs (e.g. kitchen-bedroom)
5. Classify walls as internal/external
6. Place doors on shared walls, windows on external walls
7. Ensure full connectivity via BFS + door insertion

### COCO Categories

| ID | Name | Type |
|----|------|------|
| 1 | wall | structure |
| 2-13 | hallway, corridor, living_room, bedroom, kitchen, bathroom, toilet, balcony, storage, utility, garage, terrace | space |
| 14 | door | opening |
| 15 | window | opening |

## JSON Format

All dimensions in millimeters. Example structure:

```json
{
  "meta": {
    "seed": 42,
    "global_wall_thickness": 120,
    "units": "mm"
  },
  "spaces": [
    {
      "id": "space_001",
      "type": "hallway",
      "polygon": [[0, 0], [2500, 0], [2500, 3000], [0, 3000]],
      "walls": [
        {
          "p1": [0, 0], "p2": [2500, 0],
          "thickness": 120,
          "is_external": true,
          "openings": [
            {"type": "window", "width": 1200, "offset": 0.5}
          ]
        }
      ]
    }
  ]
}
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=floorplan
```

## Requirements

- Python >= 3.11
- numpy, Pillow, shapely, pydantic, click, tqdm

# Apartment Architecture: Rule-Based Realistic Floorplan Generation

**Date:** 2026-02-22
**Approach:** Rule-Based Generation (current growing-structure algorithm + architectural rules)
**Goal:** Realistic floorplans for both ML dataset (COCO) and client-facing product

---

## 1. New Room Types (SpaceType)

Replace 4 types with 12:

```python
class SpaceType(str, Enum):
    HALLWAY = "hallway"           # Entry point, always first room
    CORRIDOR = "corridor"         # Circulation between zones
    LIVING_ROOM = "living_room"   # Main living area
    BEDROOM = "bedroom"           # Sleeping room
    KITCHEN = "kitchen"           # Kitchen
    BATHROOM = "bathroom"         # Combined bath+toilet
    TOILET = "toilet"             # Separate toilet
    BALCONY = "balcony"           # Balcony/loggia
    STORAGE = "storage"           # Closet/pantry/wardrobe
    UTILITY = "utility"           # Boiler room/laundry
    GARAGE = "garage"             # Garage (houses only)
    TERRACE = "terrace"           # Terrace (houses only)
```

Old `ROOM` type removed. Backward compat: existing JSONs with `ROOM` map to `LIVING_ROOM`.

---

## 2. Layout Templates (Apartment/House Composition)

### Apartments

| Type | Total m2 | Required Rooms | Optional |
|---|---|---|---|
| studio | 25-35 | hallway, bathroom, kitchen+living_room (combined) | balcony |
| 1room | 35-45 | hallway, bathroom, kitchen, living_room | balcony, storage |
| 2room | 50-65 | hallway, bathroom, toilet, kitchen, living_room, bedroom | balcony, storage, corridor |
| 3room | 70-90 | hallway, bathroom, toilet, kitchen, living_room, bedroom x2 | balcony, storage, corridor |
| 4room | 90-120 | hallway, bathroom, toilet, kitchen, living_room, bedroom x3 | balcony x2, storage, corridor |

### Houses

| Type | Total m2 | Required Rooms | Optional |
|---|---|---|---|
| house_small | 80-120 | hallway, bathroom, toilet, kitchen, living_room, bedroom x2 | terrace, storage, utility |
| house_medium | 120-180 | hallway, bathroom, toilet, kitchen, living_room, bedroom x3, corridor | terrace, garage, storage, utility |
| house_large | 180-250 | hallway, bathroom x2, toilet, kitchen, living_room, bedroom x3, corridor | terrace, garage, storage x2, utility |

CLI: `--layout-type studio|1room|2room|3room|4room|house_small|house_medium|house_large|random`

---

## 3. Adjacency Graph and Generation Order

### Generation order (replaces random attachment):

1. Place `hallway` (always first, entry point)
2. Attach `kitchen`, `bathroom`/`toilet` to hallway
3. Attach `corridor` to hallway (if needed for larger layouts)
4. Attach `living_room` to hallway or corridor
5. Attach `bedroom`(s) to corridor/hallway/living_room
6. Attach `balcony` to living_room/bedroom/kitchen (external wall only)
7. Attach `storage` to hallway/corridor
8. Attach `garage`/`terrace` to hallway (houses only, external)

### Forbidden connections (NEVER):

- kitchen <-> bedroom
- kitchen <-> bathroom/toilet
- living_room <-> bathroom/toilet
- bathroom <-> bathroom
- bedroom <-> bedroom (no direct door)

### External wall requirements:

- MUST have external wall: living_room, bedroom, kitchen, balcony, terrace, garage
- MAY be internal: hallway, corridor, bathroom, toilet, storage, utility

---

## 4. Room Dimensions (mm, from SNiP)

| Type | Width min | Width max | Height min | Height max | Aspect ratio |
|---|---|---|---|---|---|
| hallway | 1400 | 3500 | 1400 | 5000 | 1:1 - 1:2.5 |
| corridor | 850 | 1500 | 3000 | 10000 | 1:3 - 1:10 |
| living_room | 3200 | 6500 | 3200 | 7000 | 1:1 - 1:1.5 |
| bedroom | 2400 | 5000 | 2400 | 5500 | 1:1 - 1:1.5 |
| kitchen | 1700 | 4500 | 2200 | 5000 | 1:1 - 1:1.6 |
| bathroom | 1500 | 3000 | 1500 | 3500 | 1:1 - 1:1.5 |
| toilet | 800 | 1200 | 1200 | 1800 | 1:1.4 - 1:2 |
| balcony | 800 | 2000 | 2000 | 6000 | 1:2 - 1:5 |
| storage | 800 | 2000 | 800 | 2500 | 1:1 - 1:2 |
| utility | 1800 | 3500 | 2000 | 3500 | 1:1 - 1:1.5 |
| garage | 3500 | 7000 | 5500 | 7500 | ~1:1.7 |
| terrace | 3000 | 5000 | 3000 | 9000 | 1:1 - 1:3 |

---

## 5. Door and Window Rules

### Doors by room type:

| Context | Width (mm) |
|---|---|
| Entry (hallway -> exterior) | 900-1000 |
| Standard interior | 700-800 |
| Bathroom / toilet | 600-700 |
| Balcony | 700-800 |

### Windows:

| Category | Rooms | Window |
|---|---|---|
| MUST have window | living_room, bedroom, kitchen | 1000-1800 mm |
| MAY have window | bathroom, toilet, hallway | 400-700 mm or none |
| No window | corridor, storage | never |

### Wall thickness:

| Type | Thickness (mm) |
|---|---|
| External | 200 |
| Internal bearing | 150 |
| Partition | 120 |
| Bathroom partition | 100 |

---

## 6. Renderer and Furniture Updates

### Room fill colors (new types):

Each new SpaceType gets a distinct fill color in StyleConfig.room_fill_colors.

### Furniture icon mapping:

| Icon prefix | Rooms |
|---|---|
| bed_double_* | bedroom |
| sofa_*, sofa_set_*, sofa_corner_*, armchair_* | living_room |
| dining_table_* | kitchen, living_room |
| desk_* | bedroom, living_room |
| sink_* | bathroom, kitchen |
| cabinet_* | utility, storage |
| plant_* | living_room, hallway, corridor, balcony |
| rug_* | living_room, bedroom |
| sideboard_* | living_room, corridor |
| ottoman_* | living_room, bedroom |
| curtain_* | living_room, bedroom |

### COCO annotations:

Update category list to 12 room types. Class IDs auto-assigned from SpaceType enum order.

### Backward compatibility:

- `--num-rooms` kept as override; default composition from `--layout-type`
- Old JSON with `ROOM` type -> `LIVING_ROOM` on load

---

## 7. Files to Modify

| File | Changes |
|---|---|
| `src/floorplan/models.py` | New SpaceType enum (12 types), backward compat validator |
| `src/floorplan/generator.py` | Layout templates, ordered generation, adjacency rules, room dimensions, door/window rules |
| `src/floorplan/styles.py` | Fill colors for new room types |
| `src/floorplan/renderer.py` | No structural changes (reads SpaceType dynamically) |
| `src/floorplan/furniture.py` | Update _ICON_ROOM_TYPES mapping for new types |
| `src/floorplan/annotations.py` | Update COCO categories for 12 types |
| `scripts/floorplan_generator.py` | Add --layout-type CLI param |
| `scripts/generate_synthetic.py` | Pass layout_type through |
| `tests/` | Update all tests for new types, add layout template tests |

# Apartment Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace random room generation with realistic apartment/house layouts using architectural rules from SNiP, real room types, adjacency constraints, and proper dimensions.

**Architecture:** Extend the existing growing-structure algorithm with: (1) new SpaceType enum (12 types), (2) layout templates defining room composition per dwelling type, (3) ordered generation with adjacency rules, (4) SNiP-based room dimensions and door/window rules. The generator, styles, furniture, and annotations modules all update to support new types.

**Tech Stack:** Python, Pydantic, Shapely, NumPy, PIL, Click

---

### Task 1: Expand SpaceType enum with backward compatibility

**Files:**
- Modify: `src/floorplan/models.py:10-14`
- Modify: `tests/test_models.py`

**Step 1: Update the SpaceType enum**

Replace lines 10-14 in `src/floorplan/models.py`:

```python
class SpaceType(str, Enum):
    HALLWAY = "hallway"
    CORRIDOR = "corridor"
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    TOILET = "toilet"
    BALCONY = "balcony"
    STORAGE = "storage"
    UTILITY = "utility"
    GARAGE = "garage"
    TERRACE = "terrace"
```

**Step 2: Add backward-compat validator on Space.type**

In the `Space` class, add a validator that maps old `"room"` value to `LIVING_ROOM`:

```python
class Space(BaseModel):
    id: str
    type: SpaceType
    polygon: list[list[float]] = Field(min_length=3)
    walls: list[Wall] = Field(default_factory=list)

    @field_validator("type", mode="before")
    @classmethod
    def migrate_legacy_type(cls, v):
        if v == "room":
            return "living_room"
        return v
```

**Step 3: Update tests**

In `tests/test_models.py`, update `test_space_creation` to use `SpaceType.LIVING_ROOM` instead of `SpaceType.ROOM`. Add a test:

```python
def test_legacy_room_type_migration():
    space = Space(id="s1", type="room", polygon=[[0,0],[1,0],[1,1]])
    assert space.type == SpaceType.LIVING_ROOM
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_models.py -v`
Expected: All pass

**Step 5: Fix any remaining `SpaceType.ROOM` references across codebase**

Search for `SpaceType.ROOM` and `"room"` in all source files and update to appropriate new types. Key places:
- `src/floorplan/generator.py:148` — room type list
- `src/floorplan/styles.py:24` — ROOM_FILLS dict key
- `src/floorplan/annotations.py:30` — SPACE_TYPE_TO_CLASS
- `tests/` — various test files

**Step 6: Run all tests, fix breakages**

Run: `uv run pytest tests/ -v`

**Step 7: Commit**

```
feat: expand SpaceType enum to 12 room types with backward compat
```

---

### Task 2: Add layout templates and room dimension specs

**Files:**
- Create: `src/floorplan/layout_templates.py`
- Test: `tests/test_layout_templates.py`

**Step 1: Write the test**

```python
# tests/test_layout_templates.py
from floorplan.layout_templates import LAYOUT_TEMPLATES, ROOM_SPECS, get_room_list

def test_all_template_types_exist():
    expected = {"studio", "1room", "2room", "3room", "4room",
                "house_small", "house_medium", "house_large"}
    assert set(LAYOUT_TEMPLATES.keys()) == expected

def test_get_room_list_deterministic():
    import numpy as np
    rng = np.random.default_rng(42)
    rooms = get_room_list("2room", rng)
    # Must contain all required rooms
    types = [r for r in rooms]
    assert "hallway" in types
    assert "bathroom" in types
    assert "kitchen" in types
    assert "living_room" in types
    assert "bedroom" in types

def test_room_specs_have_all_types():
    from floorplan.models import SpaceType
    for st in SpaceType:
        assert st.value in ROOM_SPECS, f"Missing spec for {st.value}"

def test_studio_has_no_separate_bedroom():
    import numpy as np
    rng = np.random.default_rng(42)
    rooms = get_room_list("studio", rng)
    assert "bedroom" not in rooms
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_layout_templates.py -v`

**Step 3: Implement layout_templates.py**

```python
# src/floorplan/layout_templates.py
"""Layout templates and room dimension specs based on SNiP standards."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RoomSpec:
    """Dimension constraints for a room type (all in mm)."""
    width_min: float
    width_max: float
    height_min: float
    height_max: float
    door_width_min: float
    door_width_max: float
    must_have_window: bool
    may_have_window: bool  # if True and must_have_window is False, 50% chance


ROOM_SPECS: dict[str, RoomSpec] = {
    "hallway":     RoomSpec(1400, 3500, 1400, 5000, 900, 1000, False, False),
    "corridor":    RoomSpec(850,  1500, 3000, 10000, 700, 800,  False, False),
    "living_room": RoomSpec(3200, 6500, 3200, 7000, 800, 900,  True,  False),
    "bedroom":     RoomSpec(2400, 5000, 2400, 5500, 700, 800,  True,  False),
    "kitchen":     RoomSpec(1700, 4500, 2200, 5000, 700, 800,  True,  False),
    "bathroom":    RoomSpec(1500, 3000, 1500, 3500, 600, 700,  False, True),
    "toilet":      RoomSpec(800,  1200, 1200, 1800, 600, 700,  False, True),
    "balcony":     RoomSpec(800,  2000, 2000, 6000, 700, 800,  False, False),
    "storage":     RoomSpec(800,  2000, 800,  2500, 600, 700,  False, False),
    "utility":     RoomSpec(1800, 3500, 2000, 3500, 700, 800,  False, True),
    "garage":      RoomSpec(3500, 7000, 5500, 7500, 800, 900,  False, False),
    "terrace":     RoomSpec(3000, 5000, 3000, 9000, 800, 900,  False, False),
}


@dataclass
class LayoutTemplate:
    """Defines the room composition for a dwelling type."""
    required: list[str]          # Room types that must be present
    optional: list[str]          # Room types that may be added (each has 50% chance)
    optional_prob: float = 0.5   # Probability for each optional room


LAYOUT_TEMPLATES: dict[str, LayoutTemplate] = {
    "studio": LayoutTemplate(
        required=["hallway", "bathroom", "living_room"],
        optional=["balcony"],
    ),
    "1room": LayoutTemplate(
        required=["hallway", "bathroom", "kitchen", "living_room"],
        optional=["balcony", "storage"],
    ),
    "2room": LayoutTemplate(
        required=["hallway", "bathroom", "toilet", "kitchen", "living_room", "bedroom"],
        optional=["balcony", "storage", "corridor"],
    ),
    "3room": LayoutTemplate(
        required=["hallway", "bathroom", "toilet", "kitchen", "living_room",
                  "bedroom", "bedroom"],
        optional=["balcony", "storage", "corridor"],
    ),
    "4room": LayoutTemplate(
        required=["hallway", "bathroom", "toilet", "kitchen", "living_room",
                  "bedroom", "bedroom", "bedroom"],
        optional=["balcony", "balcony", "storage", "corridor"],
    ),
    "house_small": LayoutTemplate(
        required=["hallway", "bathroom", "toilet", "kitchen", "living_room",
                  "bedroom", "bedroom"],
        optional=["terrace", "storage", "utility"],
    ),
    "house_medium": LayoutTemplate(
        required=["hallway", "bathroom", "toilet", "kitchen", "living_room",
                  "bedroom", "bedroom", "bedroom", "corridor"],
        optional=["terrace", "garage", "storage", "utility"],
    ),
    "house_large": LayoutTemplate(
        required=["hallway", "bathroom", "bathroom", "toilet", "kitchen",
                  "living_room", "bedroom", "bedroom", "bedroom", "corridor"],
        optional=["terrace", "garage", "storage", "storage", "utility"],
    ),
}


def get_room_list(layout_type: str, rng: np.random.Generator) -> list[str]:
    """Return ordered list of room types for the given layout.

    Order follows architectural generation sequence:
    hallway first, then service rooms, then living spaces, then bedrooms, then extras.
    """
    template = LAYOUT_TEMPLATES[layout_type]
    rooms = list(template.required)

    for opt in template.optional:
        if rng.random() < template.optional_prob:
            rooms.append(opt)

    # Sort into generation order
    order = {
        "hallway": 0, "kitchen": 1, "bathroom": 2, "toilet": 3,
        "corridor": 4, "living_room": 5, "bedroom": 6, "storage": 7,
        "utility": 8, "balcony": 9, "terrace": 10, "garage": 11,
    }
    rooms.sort(key=lambda r: order.get(r, 99))
    return rooms
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_layout_templates.py -v`

**Step 5: Commit**

```
feat: add layout templates and room dimension specs
```

---

### Task 3: Add adjacency rules module

**Files:**
- Create: `src/floorplan/adjacency.py`
- Test: `tests/test_adjacency.py`

**Step 1: Write the test**

```python
# tests/test_adjacency.py
from floorplan.adjacency import can_connect, get_allowed_parents

def test_kitchen_never_connects_to_bedroom():
    assert not can_connect("kitchen", "bedroom")
    assert not can_connect("bedroom", "kitchen")

def test_hallway_connects_to_kitchen():
    assert can_connect("hallway", "kitchen")

def test_bathroom_never_connects_to_living_room():
    assert not can_connect("bathroom", "living_room")
    assert not can_connect("living_room", "bathroom")

def test_get_allowed_parents_for_bedroom():
    parents = get_allowed_parents("bedroom")
    assert "hallway" in parents
    assert "corridor" in parents
    assert "living_room" in parents
    assert "kitchen" not in parents

def test_balcony_allowed_parents():
    parents = get_allowed_parents("balcony")
    assert "living_room" in parents
    assert "bedroom" in parents
    assert "kitchen" in parents
    assert "hallway" not in parents
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_adjacency.py -v`

**Step 3: Implement adjacency.py**

```python
# src/floorplan/adjacency.py
"""Adjacency rules for room connections based on architectural standards."""
from __future__ import annotations

# Forbidden connections (bidirectional)
NEVER_CONNECT: set[frozenset[str]] = {
    frozenset({"kitchen", "bedroom"}),
    frozenset({"kitchen", "bathroom"}),
    frozenset({"kitchen", "toilet"}),
    frozenset({"living_room", "bathroom"}),
    frozenset({"living_room", "toilet"}),
    frozenset({"bathroom", "bathroom"}),
    frozenset({"bedroom", "bedroom"}),
}

# Allowed parent rooms for each type (where it can be attached to)
ALLOWED_PARENTS: dict[str, list[str]] = {
    "hallway":     [],  # First room, no parent
    "kitchen":     ["hallway", "corridor", "living_room"],
    "bathroom":    ["hallway", "corridor", "bedroom"],
    "toilet":      ["hallway", "corridor", "bedroom"],
    "corridor":    ["hallway"],
    "living_room": ["hallway", "corridor"],
    "bedroom":     ["hallway", "corridor", "living_room"],
    "balcony":     ["living_room", "bedroom", "kitchen"],
    "storage":     ["hallway", "corridor", "bedroom"],
    "utility":     ["hallway", "corridor"],
    "garage":      ["hallway", "utility"],
    "terrace":     ["hallway", "living_room"],
}

# Rooms that MUST have at least one external wall (for windows / outdoor access)
MUST_HAVE_EXTERNAL_WALL: set[str] = {
    "living_room", "bedroom", "kitchen", "balcony", "terrace", "garage",
}


def can_connect(type_a: str, type_b: str) -> bool:
    """Check if two room types are allowed to have a door between them."""
    return frozenset({type_a, type_b}) not in NEVER_CONNECT


def get_allowed_parents(room_type: str) -> list[str]:
    """Return list of room types this room can be attached to."""
    return ALLOWED_PARENTS.get(room_type, [])
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_adjacency.py -v`

**Step 5: Commit**

```
feat: add adjacency rules module
```

---

### Task 4: Rewrite generator with layout-aware generation

**Files:**
- Modify: `src/floorplan/generator.py` (major rewrite)
- Modify: `tests/test_generator.py`

This is the largest task. The generator must:

1. Accept `layout_type` parameter (or fall back to `num_rooms` for legacy mode)
2. Use `get_room_list()` to determine room composition
3. Generate rooms in template order (hallway first)
4. Pick parent rooms from `ALLOWED_PARENTS` instead of random
5. Use `ROOM_SPECS` for dimensions instead of generic min/max
6. Apply `can_connect()` validation
7. Use room-type-specific door widths and window rules

**Step 1: Update GeneratorConfig**

Add `layout_type: str | None = None` to `GeneratorConfig`. When set, it overrides `num_rooms` and type probabilities.

**Step 2: Rewrite `_room_dimensions()`**

Use `ROOM_SPECS[stype.value]` to get width/height ranges per type instead of the current hardcoded corridors/bathrooms logic.

**Step 3: Replace `_pick_room_type()` + random attachment with ordered generation**

In `generate()`, when `layout_type` is set:
- Call `get_room_list(layout_type, rng)` to get ordered room list
- For each room in the list, find a valid parent using `ALLOWED_PARENTS`
- Attach to parent's external wall, respecting `MUST_HAVE_EXTERNAL_WALL`

When `layout_type` is None, keep the existing random behavior (backward compat) but use the new SpaceType values.

**Step 4: Update `_place_openings()`**

Use `ROOM_SPECS[space.type.value].door_width_min/max` for door widths. Apply window rules: `must_have_window` rooms always get windows on external walls; `may_have_window` rooms get windows with 50% probability; others never get windows.

**Step 5: Update all generator tests**

Update `tests/test_generator.py`:
- Replace `SpaceType.ROOM` references with new types
- Add tests for layout-type generation:

```python
def test_layout_type_2room():
    cfg = GeneratorConfig(seed=42, layout_type="2room")
    fp = FloorplanGenerator(cfg).generate()
    types = [s.type.value for s in fp.spaces]
    assert "hallway" in types
    assert "kitchen" in types
    assert "living_room" in types
    assert "bedroom" in types
    assert "bathroom" in types

def test_layout_hallway_always_first():
    cfg = GeneratorConfig(seed=42, layout_type="3room")
    fp = FloorplanGenerator(cfg).generate()
    assert fp.spaces[0].type.value == "hallway"

def test_forbidden_adjacency_respected():
    cfg = GeneratorConfig(seed=42, layout_type="3room")
    fp = FloorplanGenerator(cfg).generate()
    # Check no kitchen-bedroom doors exist
    # (verify through wall adjacency analysis)
```

**Step 6: Run all tests**

Run: `uv run pytest tests/ -v`

**Step 7: Commit**

```
feat: rewrite generator with layout-aware ordered generation
```

---

### Task 5: Update styles with new room type colors

**Files:**
- Modify: `src/floorplan/styles.py:23-28`
- Modify: `tests/test_styles.py`

**Step 1: Update ROOM_FILLS dict**

Replace lines 23-28:

```python
ROOM_FILLS = {
    "hallway":     [(245, 240, 230), (240, 235, 225)],
    "corridor":    [(245, 245, 245), (235, 235, 235)],
    "living_room": [(230, 240, 255), (240, 248, 255)],
    "bedroom":     [(240, 235, 250), (235, 230, 245)],
    "kitchen":     [(255, 245, 230), (255, 240, 220)],
    "bathroom":    [(220, 240, 255), (200, 230, 255)],
    "toilet":      [(210, 235, 250), (200, 225, 245)],
    "balcony":     [(235, 250, 235), (225, 245, 225)],
    "storage":     [(245, 240, 235), (240, 235, 230)],
    "utility":     [(255, 240, 220), (245, 235, 225)],
    "garage":      [(230, 230, 230), (225, 225, 225)],
    "terrace":     [(230, 248, 230), (220, 242, 220)],
}
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_styles.py -v`

**Step 3: Commit**

```
feat: add fill colors for all 12 room types
```

---

### Task 6: Update furniture icon mapping for new room types

**Files:**
- Modify: `src/floorplan/furniture.py` (the `_ICON_ROOM_TYPES` dict)
- Modify: `tests/test_furniture.py`

**Step 1: Update _ICON_ROOM_TYPES**

```python
_ICON_ROOM_TYPES: dict[str, list[str]] = {
    "bed_double":    ["bedroom"],
    "sofa":          ["living_room"],
    "sofa_set":      ["living_room"],
    "sofa_corner":   ["living_room"],
    "dining_table":  ["kitchen", "living_room"],
    "armchair":      ["living_room"],
    "desk":          ["bedroom", "living_room"],
    "rug":           ["living_room", "bedroom"],
    "curtain":       ["living_room", "bedroom"],
    "ottoman":       ["living_room", "bedroom"],
    "plant":         ["living_room", "hallway", "corridor", "balcony"],
    "sideboard":     ["living_room", "corridor", "hallway"],
    "sink":          ["bathroom", "kitchen"],
    "cabinet":       ["utility", "storage"],
}
```

**Step 2: Update tests to use new type names**

In `tests/test_furniture.py`, change `"room"` to `"living_room"` in the `place_furniture` calls.

**Step 3: Run tests**

Run: `uv run pytest tests/test_furniture.py -v`

**Step 4: Commit**

```
feat: update furniture icon mapping for 12 room types
```

---

### Task 7: Update COCO annotations for new room types

**Files:**
- Modify: `src/floorplan/annotations.py:29-44`
- Modify: `tests/test_annotations.py`

**Step 1: Update SPACE_TYPE_TO_CLASS and CATEGORIES**

```python
SPACE_TYPE_TO_CLASS: dict[SpaceType, int] = {
    SpaceType.HALLWAY: 2,
    SpaceType.CORRIDOR: 3,
    SpaceType.LIVING_ROOM: 4,
    SpaceType.BEDROOM: 5,
    SpaceType.KITCHEN: 6,
    SpaceType.BATHROOM: 7,
    SpaceType.TOILET: 8,
    SpaceType.BALCONY: 9,
    SpaceType.STORAGE: 10,
    SpaceType.UTILITY: 11,
    SpaceType.GARAGE: 12,
    SpaceType.TERRACE: 13,
}

CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "wall", "supercategory": "structure"},
    {"id": 2, "name": "hallway", "supercategory": "space"},
    {"id": 3, "name": "corridor", "supercategory": "space"},
    {"id": 4, "name": "living_room", "supercategory": "space"},
    {"id": 5, "name": "bedroom", "supercategory": "space"},
    {"id": 6, "name": "kitchen", "supercategory": "space"},
    {"id": 7, "name": "bathroom", "supercategory": "space"},
    {"id": 8, "name": "toilet", "supercategory": "space"},
    {"id": 9, "name": "balcony", "supercategory": "space"},
    {"id": 10, "name": "storage", "supercategory": "space"},
    {"id": 11, "name": "utility", "supercategory": "space"},
    {"id": 12, "name": "garage", "supercategory": "space"},
    {"id": 13, "name": "terrace", "supercategory": "space"},
    {"id": 14, "name": "door", "supercategory": "opening"},
    {"id": 15, "name": "window", "supercategory": "opening"},
]
```

**Step 2: Update annotation tests**

In `tests/test_annotations.py`, update `test_coco_categories` to expect 15 categories instead of 7. Update `test_semantic_mask_has_classes` to check for new class IDs.

**Step 3: Run tests**

Run: `uv run pytest tests/test_annotations.py -v`

**Step 4: Commit**

```
feat: update COCO annotations for 12 room types
```

---

### Task 8: Update CLI scripts with --layout-type

**Files:**
- Modify: `scripts/floorplan_generator.py`
- Modify: `scripts/generate_synthetic.py`
- Modify: `tests/test_cli_generator.py`
- Modify: `tests/test_cli_synthetic.py`

**Step 1: Add --layout-type to floorplan_generator.py**

Add option:
```python
@click.option("--layout-type", type=click.Choice(
    ["studio", "1room", "2room", "3room", "4room",
     "house_small", "house_medium", "house_large", "random"],
    case_sensitive=False,
), default=None, help="Layout type (overrides --num-rooms)")
```

When `layout_type` is set (and not "random"), pass it to `GeneratorConfig(layout_type=layout_type)`. When "random", pick randomly from the list. When None, use `--num-rooms` as before.

**Step 2: Update CLI tests**

Add test:
```python
def test_cli_layout_type(tmp_path):
    result = runner.invoke(cli, [
        "--count", "2", "--seed", "42",
        "--layout-type", "2room",
        "--output-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert len(list(tmp_path.glob("*.json"))) == 2
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_cli_generator.py tests/test_cli_synthetic.py -v`

**Step 4: Commit**

```
feat: add --layout-type CLI option for realistic apartment generation
```

---

### Task 9: Integration test and visual verification

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Add layout-type integration test**

```python
def test_full_pipeline_with_layout_type():
    cfg = GeneratorConfig(seed=42, layout_type="2room")
    fp = FloorplanGenerator(cfg).generate()
    assert len(fp.spaces) >= 5  # At least the required rooms
    assert fp.spaces[0].type.value == "hallway"

    style = generate_style(np.random.default_rng(42))
    renderer = FloorplanRenderer(RenderConfig(image_size=512))
    img = renderer.render(fp, style)
    assert img.size == (512, 512)
```

**Step 2: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

**Step 3: Generate sample images for visual inspection**

```bash
rm -rf output/
uv run python scripts/floorplan_generator.py --count 10 --seed 42 --layout-type random --output-dir output/json
uv run python scripts/generate_synthetic.py --input-dir output/json --output-dir output --image-size 512 --workers 1 --with-masks --with-coco --furniture-dir furniture_icons
```

**Step 4: Commit**

```
test: add integration tests for layout-type generation
```

---

## Summary

| Task | Description | Est. Lines |
|---|---|---|
| 1 | SpaceType enum (12 types) + backward compat | ~40 |
| 2 | Layout templates + room specs | ~120 |
| 3 | Adjacency rules module | ~50 |
| 4 | Generator rewrite (largest task) | ~200 |
| 5 | Style colors for new types | ~20 |
| 6 | Furniture icon mapping | ~20 |
| 7 | COCO annotations update | ~30 |
| 8 | CLI --layout-type | ~30 |
| 9 | Integration test + visual check | ~20 |

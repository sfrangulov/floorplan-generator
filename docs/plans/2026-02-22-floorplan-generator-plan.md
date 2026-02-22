# Floorplan Generator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a synthetic floorplan generation pipeline that produces diverse architectural plans as images with COCO annotations for CV training.

**Architecture:** Two-stage pipeline — Stage 1 generates structured JSON floorplan descriptions via a growing-structure algorithm with Shapely validation. Stage 2 renders images with randomized visual styles and produces segmentation masks + COCO annotations. Both stages share a common `floorplan` package with Pydantic models.

**Tech Stack:** Python 3.11+, UV, numpy, Pillow, shapely, pydantic, click, tqdm

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/floorplan/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `scripts/.gitkeep`
- Create: `.gitignore`

**Step 1: Initialize UV project**

```bash
uv init --lib --name floorplan
```

If UV created files in wrong structure, reorganize. The goal is `src/floorplan/` layout.

**Step 2: Edit pyproject.toml**

```toml
[project]
name = "floorplan"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "Pillow>=10.0",
    "shapely>=2.0",
    "pydantic>=2.0",
    "click>=8.0",
    "tqdm>=4.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/floorplan"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[dependency-groups]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]
```

**Step 3: Install dependencies**

```bash
uv sync
```

**Step 4: Create .gitignore**

```
output/
__pycache__/
*.egg-info/
.venv/
dist/
*.pyc
```

**Step 5: Create empty test conftest**

```python
# tests/conftest.py
import pytest
```

**Step 6: Verify setup**

```bash
uv run python -c "import floorplan; print('OK')"
uv run pytest --co
```

Expected: "OK" and pytest collects 0 tests.

**Step 7: Commit**

```bash
git add pyproject.toml src/ tests/ scripts/.gitkeep .gitignore
git commit -m "chore: scaffold project with UV, deps, src layout"
```

---

### Task 2: Pydantic Models

**Files:**
- Create: `src/floorplan/models.py`
- Create: `tests/test_models.py`

**Step 1: Write failing tests**

```python
# tests/test_models.py
import pytest
from floorplan.models import (
    Opening, Wall, Space, FloorplanMeta, Floorplan, SpaceType, OpeningType
)


def test_opening_door():
    o = Opening(type=OpeningType.DOOR, width=900, offset=0.3, swing="left")
    assert o.type == OpeningType.DOOR
    assert o.swing == "left"


def test_opening_window_no_swing():
    o = Opening(type=OpeningType.WINDOW, width=1200, offset=0.6)
    assert o.swing is None


def test_wall_creation():
    w = Wall(
        p1=[0.0, 0.0], p2=[3000.0, 0.0],
        thickness=120, is_external=True, openings=[]
    )
    assert w.is_external is True
    assert w.thickness == 120


def test_space_creation():
    s = Space(
        id="space_001",
        type=SpaceType.ROOM,
        polygon=[[0, 0], [3000, 0], [3000, 4000], [0, 4000]],
        walls=[]
    )
    assert s.type == SpaceType.ROOM
    assert len(s.polygon) == 4


def test_floorplan_roundtrip_json():
    fp = Floorplan(
        meta=FloorplanMeta(seed=42, global_wall_thickness=120),
        spaces=[
            Space(
                id="space_001",
                type=SpaceType.ROOM,
                polygon=[[0, 0], [3000, 0], [3000, 4000], [0, 4000]],
                walls=[
                    Wall(p1=[0, 0], p2=[3000, 0], thickness=120,
                         is_external=True, openings=[])
                ]
            )
        ]
    )
    json_str = fp.model_dump_json()
    fp2 = Floorplan.model_validate_json(json_str)
    assert fp2.meta.seed == 42
    assert len(fp2.spaces) == 1
    assert fp2.spaces[0].id == "space_001"


def test_floorplan_invalid_offset_rejected():
    with pytest.raises(Exception):
        Opening(type=OpeningType.DOOR, width=900, offset=1.5)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_models.py -v
```

Expected: FAIL — module not found.

**Step 3: Implement models**

```python
# src/floorplan/models.py
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SpaceType(str, Enum):
    ROOM = "room"
    CORRIDOR = "corridor"
    BATHROOM = "bathroom"
    UTILITY = "utility"


class OpeningType(str, Enum):
    DOOR = "door"
    WINDOW = "window"


class Opening(BaseModel):
    type: OpeningType
    width: float = Field(gt=0, description="Width in mm")
    offset: float = Field(ge=0.0, le=1.0, description="Position along wall (0-1)")
    swing: Optional[str] = Field(
        default=None, description="Door swing direction: left/right"
    )

    @field_validator("swing")
    @classmethod
    def validate_swing(cls, v: Optional[str], info) -> Optional[str]:
        if v is not None and v not in ("left", "right"):
            raise ValueError("swing must be 'left' or 'right'")
        return v


class Wall(BaseModel):
    p1: list[float] = Field(min_length=2, max_length=2)
    p2: list[float] = Field(min_length=2, max_length=2)
    thickness: float = Field(gt=0)
    is_external: bool = True
    openings: list[Opening] = Field(default_factory=list)


class Space(BaseModel):
    id: str
    type: SpaceType
    polygon: list[list[float]] = Field(min_length=3)
    walls: list[Wall] = Field(default_factory=list)


class FloorplanMeta(BaseModel):
    seed: int
    global_wall_thickness: float = Field(default=120, gt=0)
    units: str = "mm"


class Floorplan(BaseModel):
    meta: FloorplanMeta
    spaces: list[Space]
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_models.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add src/floorplan/models.py tests/test_models.py
git commit -m "feat: add Pydantic models for floorplan JSON schema"
```

---

### Task 3: Geometry Utilities

**Files:**
- Create: `src/floorplan/geometry.py`
- Create: `tests/test_geometry.py`

**Step 1: Write failing tests**

```python
# tests/test_geometry.py
import pytest
import numpy as np
from shapely.geometry import Polygon, box

from floorplan.geometry import (
    create_rectangle,
    create_indented_room,
    attach_room_to_wall,
    find_shared_walls,
    classify_external_walls,
    snap_to_grid,
    polygons_overlap,
)


def test_create_rectangle():
    poly = create_rectangle(0, 0, 3000, 4000)
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    assert abs(poly.area - 3000 * 4000) < 1e-6


def test_create_indented_room():
    rng = np.random.default_rng(42)
    poly = create_indented_room(
        width=4000, height=3000,
        indent_depth=500, indent_width=1000,
        rng=rng
    )
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    assert poly.area < 4000 * 3000  # indentation reduces area
    assert len(poly.exterior.coords) > 5  # more than rectangle


def test_snap_to_grid():
    coords = [(100.3, 200.7), (300.1, 400.9)]
    snapped = snap_to_grid(coords, grid_size=1.0)
    assert snapped == [(100.0, 201.0), (300.0, 401.0)]


def test_polygons_overlap():
    p1 = box(0, 0, 100, 100)
    p2 = box(50, 50, 150, 150)
    p3 = box(200, 200, 300, 300)
    assert polygons_overlap(p1, p2) is True
    assert polygons_overlap(p1, p3) is False


def test_polygons_touching_not_overlap():
    p1 = box(0, 0, 100, 100)
    p2 = box(100, 0, 200, 100)  # shares edge
    assert polygons_overlap(p1, p2) is False


def test_attach_room_to_wall():
    rng = np.random.default_rng(42)
    existing = box(0, 0, 3000, 4000)
    wall_start = np.array([3000.0, 0.0])
    wall_end = np.array([3000.0, 4000.0])
    new_width, new_height = 2500, 3000
    result = attach_room_to_wall(
        wall_start, wall_end, new_width, new_height, rng=rng
    )
    assert isinstance(result, Polygon)
    assert result.is_valid
    # New room should be to the right of x=3000
    minx, _, _, _ = result.bounds
    assert minx >= 3000 - 1  # tolerance for snapping


def test_find_shared_walls():
    p1 = box(0, 0, 3000, 4000)
    p2 = box(3000, 0, 6000, 4000)  # shares right wall of p1
    shared = find_shared_walls(p1, p2)
    assert len(shared) > 0
    # shared wall should be along x=3000
    for seg in shared:
        assert abs(seg[0][0] - 3000) < 1 or abs(seg[1][0] - 3000) < 1


def test_classify_external_walls():
    polys = [box(0, 0, 100, 100), box(100, 0, 200, 100)]
    walls_0 = classify_external_walls(0, polys)
    # The wall at x=100 (shared with poly 1) should be internal
    external_count = sum(1 for _, is_ext in walls_0 if is_ext)
    internal_count = sum(1 for _, is_ext in walls_0 if not is_ext)
    assert external_count == 3  # top, bottom, left
    assert internal_count == 1  # right (shared)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_geometry.py -v
```

**Step 3: Implement geometry utilities**

```python
# src/floorplan/geometry.py
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, LineString, box as shapely_box
from shapely.ops import snap, shared_paths
from shapely import intersection


def create_rectangle(x: float, y: float, width: float, height: float) -> Polygon:
    """Create a rectangle polygon at (x, y) with given dimensions."""
    return shapely_box(x, y, x + width, y + height)


def create_indented_room(
    width: float, height: float,
    indent_depth: float, indent_width: float,
    rng: np.random.Generator,
    origin_x: float = 0, origin_y: float = 0,
) -> Polygon:
    """Create a room with a rectangular indentation cut from one wall."""
    base = create_rectangle(origin_x, origin_y, width, height)
    # Choose which wall to indent: 0=bottom, 1=right, 2=top, 3=left
    wall_idx = rng.integers(0, 4)

    if wall_idx == 0:  # bottom wall
        offset = rng.uniform(0, width - indent_width)
        cutout = shapely_box(
            origin_x + offset, origin_y,
            origin_x + offset + indent_width, origin_y + indent_depth
        )
    elif wall_idx == 1:  # right wall
        offset = rng.uniform(0, height - indent_width)
        cutout = shapely_box(
            origin_x + width - indent_depth, origin_y + offset,
            origin_x + width, origin_y + offset + indent_width
        )
    elif wall_idx == 2:  # top wall
        offset = rng.uniform(0, width - indent_width)
        cutout = shapely_box(
            origin_x + offset, origin_y + height - indent_depth,
            origin_x + offset + indent_width, origin_y + height
        )
    else:  # left wall
        offset = rng.uniform(0, height - indent_width)
        cutout = shapely_box(
            origin_x, origin_y + offset,
            origin_x + indent_depth, origin_y + offset + indent_width
        )

    result = base.difference(cutout)
    if not result.is_valid or result.is_empty:
        return base  # fallback to rectangle
    return result


def snap_to_grid(coords: list[tuple[float, float]], grid_size: float) -> list[tuple[float, float]]:
    """Snap coordinates to nearest grid point."""
    return [(round(x / grid_size) * grid_size, round(y / grid_size) * grid_size)
            for x, y in coords]


def polygons_overlap(p1: Polygon, p2: Polygon) -> bool:
    """Check if two polygons overlap (sharing an edge is NOT overlap)."""
    if not p1.intersects(p2):
        return False
    inter = p1.intersection(p2)
    return inter.area > 1e-6  # only area overlap counts


def attach_room_to_wall(
    wall_start: np.ndarray, wall_end: np.ndarray,
    new_width: float, new_height: float,
    rng: np.random.Generator,
) -> Polygon:
    """Create a new room polygon attached to the given wall segment.

    The room is placed outward from the wall (away from the existing room).
    wall_start, wall_end define the wall segment.
    new_width is measured along the wall direction, new_height is perpendicular.
    """
    wall_vec = wall_end - wall_start
    wall_length = np.linalg.norm(wall_vec)
    wall_dir = wall_vec / wall_length
    # Outward normal (90 degrees clockwise for exterior wall convention)
    normal = np.array([wall_dir[1], -wall_dir[0]])

    # Position along wall — random offset so new room doesn't always start at wall_start
    max_offset = max(0, wall_length - new_width)
    offset = rng.uniform(0, max_offset) if max_offset > 0 else 0

    # Corner points of new room
    p0 = wall_start + wall_dir * offset
    p1 = p0 + wall_dir * min(new_width, wall_length)
    p2 = p1 + normal * new_height
    p3 = p0 + normal * new_height

    coords = [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]
    poly = Polygon(coords)

    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def find_shared_walls(
    p1: Polygon, p2: Polygon, tolerance: float = 5.0
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Find shared wall segments between two polygons."""
    shared = []
    boundary_inter = p1.boundary.intersection(p2.boundary)

    if boundary_inter.is_empty:
        return shared

    # Extract line segments from intersection
    if boundary_inter.geom_type == "MultiLineString":
        lines = list(boundary_inter.geoms)
    elif boundary_inter.geom_type == "LineString":
        lines = [boundary_inter]
    elif boundary_inter.geom_type == "GeometryCollection":
        lines = [g for g in boundary_inter.geoms if g.geom_type == "LineString"]
    else:
        return shared

    for line in lines:
        if line.length > tolerance:
            coords = list(line.coords)
            shared.append((coords[0], coords[-1]))

    return shared


def classify_external_walls(
    space_idx: int, all_polygons: list[Polygon], tolerance: float = 5.0
) -> list[tuple[tuple[tuple[float, float], tuple[float, float]], bool]]:
    """Classify each wall segment of a polygon as external or internal.

    Returns list of ((p1, p2), is_external) tuples.
    """
    poly = all_polygons[space_idx]
    coords = list(poly.exterior.coords)
    result = []

    for i in range(len(coords) - 1):
        seg_start = coords[i]
        seg_end = coords[i + 1]
        seg_line = LineString([seg_start, seg_end])

        is_external = True
        for j, other in enumerate(all_polygons):
            if j == space_idx:
                continue
            boundary_inter = seg_line.intersection(other.boundary)
            if not boundary_inter.is_empty and boundary_inter.length > tolerance:
                is_external = False
                break

        result.append(((seg_start, seg_end), is_external))

    return result
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_geometry.py -v
```

Expected: All PASS.

**Step 5: Commit**

```bash
git add src/floorplan/geometry.py tests/test_geometry.py
git commit -m "feat: add geometry utilities (rooms, snapping, shared walls)"
```

---

### Task 4: Floorplan Generator Core

**Files:**
- Create: `src/floorplan/generator.py`
- Create: `tests/test_generator.py`

**Step 1: Write failing tests**

```python
# tests/test_generator.py
import pytest
import json
from floorplan.generator import FloorplanGenerator, GeneratorConfig
from floorplan.models import Floorplan, SpaceType


def test_generator_config_defaults():
    cfg = GeneratorConfig()
    assert cfg.num_rooms == 5
    assert cfg.global_wall_thickness == 120


def test_generate_single_room():
    cfg = GeneratorConfig(num_rooms=1, seed=42)
    gen = FloorplanGenerator(cfg)
    fp = gen.generate()
    assert isinstance(fp, Floorplan)
    assert len(fp.spaces) == 1
    assert fp.meta.seed == 42


def test_generate_multiple_rooms():
    cfg = GeneratorConfig(num_rooms=5, seed=123)
    gen = FloorplanGenerator(cfg)
    fp = gen.generate()
    assert len(fp.spaces) == 5


def test_no_rooms_overlap():
    cfg = GeneratorConfig(num_rooms=8, seed=99)
    gen = FloorplanGenerator(cfg)
    fp = gen.generate()
    from shapely.geometry import Polygon
    polys = [Polygon(s.polygon) for s in fp.spaces]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            inter = polys[i].intersection(polys[j])
            assert inter.area < 1.0, f"Rooms {i} and {j} overlap"


def test_reproducible_with_seed():
    cfg = GeneratorConfig(num_rooms=4, seed=42)
    fp1 = FloorplanGenerator(cfg).generate()
    fp2 = FloorplanGenerator(cfg).generate()
    assert fp1.model_dump() == fp2.model_dump()


def test_json_roundtrip():
    cfg = GeneratorConfig(num_rooms=3, seed=42)
    fp = FloorplanGenerator(cfg).generate()
    json_str = fp.model_dump_json()
    fp2 = Floorplan.model_validate_json(json_str)
    assert len(fp2.spaces) == 3


def test_has_doors_between_neighbors():
    cfg = GeneratorConfig(num_rooms=5, seed=42, door_prob=1.0)
    fp = FloorplanGenerator(cfg).generate()
    door_count = sum(
        1
        for s in fp.spaces
        for w in s.walls
        for o in w.openings
        if o.type.value == "door"
    )
    assert door_count >= 4  # at least N-1 doors for connectivity


def test_windows_only_on_external_walls():
    cfg = GeneratorConfig(num_rooms=5, seed=42, window_prob=1.0)
    fp = FloorplanGenerator(cfg).generate()
    for s in fp.spaces:
        for w in s.walls:
            for o in w.openings:
                if o.type.value == "window":
                    assert w.is_external, f"Window on internal wall in {s.id}"


def test_connectivity_guaranteed():
    """All rooms reachable via doors."""
    cfg = GeneratorConfig(num_rooms=6, seed=42, door_prob=0.3)
    fp = FloorplanGenerator(cfg).generate()
    # Build adjacency graph from doors
    adj: dict[str, set[str]] = {s.id: set() for s in fp.spaces}
    space_walls: dict[str, list] = {s.id: s.walls for s in fp.spaces}
    # Find connections through shared internal walls with doors
    for s in fp.spaces:
        for w in s.walls:
            if not w.is_external and any(o.type.value == "door" for o in w.openings):
                # Find which other room shares this wall
                for other in fp.spaces:
                    if other.id == s.id:
                        continue
                    for ow in other.walls:
                        if (ow.p1 == w.p1 and ow.p2 == w.p2) or \
                           (ow.p1 == w.p2 and ow.p2 == w.p1):
                            adj[s.id].add(other.id)
                            adj[other.id].add(s.id)
    # BFS
    visited = set()
    queue = [fp.spaces[0].id]
    visited.add(fp.spaces[0].id)
    while queue:
        current = queue.pop(0)
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    assert visited == {s.id for s in fp.spaces}, "Not all rooms connected"


def test_room_types_present():
    cfg = GeneratorConfig(
        num_rooms=10, seed=42,
        room_prob=0.4, corridor_prob=0.3,
        bathroom_prob=0.2, utility_prob=0.1
    )
    fp = FloorplanGenerator(cfg).generate()
    types = {s.type for s in fp.spaces}
    assert SpaceType.ROOM in types


def test_max_extent_respected():
    cfg = GeneratorConfig(num_rooms=8, seed=42, max_extent=20000)
    fp = FloorplanGenerator(cfg).generate()
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    union = unary_union([Polygon(s.polygon) for s in fp.spaces])
    minx, miny, maxx, maxy = union.bounds
    assert (maxx - minx) <= 20000 + 500  # small tolerance
    assert (maxy - miny) <= 20000 + 500
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_generator.py -v
```

**Step 3: Implement generator**

This is the largest module. Key classes:

```python
# src/floorplan/generator.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np
from shapely.geometry import Polygon

from floorplan.models import (
    Floorplan, FloorplanMeta, Space, Wall, Opening,
    SpaceType, OpeningType,
)
from floorplan.geometry import (
    create_rectangle, create_indented_room, attach_room_to_wall,
    find_shared_walls, classify_external_walls, polygons_overlap,
    snap_to_grid,
)


@dataclass
class GeneratorConfig:
    num_rooms: int = 5
    min_room_size: float = 2500
    max_room_size: float = 5000
    indent_prob: float = 0.2
    indent_depth_range: tuple[float, float] = (300, 800)
    indent_width_range: tuple[float, float] = (600, 1500)
    global_wall_thickness: float = 120
    wall_thickness_variation_prob: float = 0.1
    wall_thickness_multiplier_range: tuple[float, float] = (1.2, 2.0)
    door_prob: float = 0.7
    window_prob: float = 0.5
    door_width: float = 900
    window_width: float = 1200
    room_prob: float = 0.4
    corridor_prob: float = 0.25
    bathroom_prob: float = 0.2
    utility_prob: float = 0.15
    max_extent: float = 30000
    seed: int = 42
    grid_size: float = 10.0
    max_attach_attempts: int = 50


class FloorplanGenerator:
    def __init__(self, config: GeneratorConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> Floorplan:
        polys: list[Polygon] = []
        types: list[SpaceType] = []

        # Step 1: Generate first room
        first_poly, first_type = self._generate_room(is_first=True)
        polys.append(first_poly)
        types.append(first_type)

        # Step 2: Iteratively attach rooms
        for _ in range(self.cfg.num_rooms - 1):
            new_poly, new_type = self._attach_new_room(polys, types)
            if new_poly is not None:
                polys.append(new_poly)
                types.append(new_type)

        # Step 3: Build walls, doors, windows
        spaces = self._build_spaces(polys, types)

        # Step 4: Ensure connectivity
        self._ensure_connectivity(spaces, polys)

        return Floorplan(
            meta=FloorplanMeta(
                seed=self.cfg.seed,
                global_wall_thickness=self.cfg.global_wall_thickness,
            ),
            spaces=spaces,
        )

    def _pick_room_type(self, neighbors: list[SpaceType] | None = None) -> SpaceType:
        probs = [self.cfg.room_prob, self.cfg.corridor_prob,
                 self.cfg.bathroom_prob, self.cfg.utility_prob]
        total = sum(probs)
        probs = [p / total for p in probs]
        types = [SpaceType.ROOM, SpaceType.CORRIDOR,
                 SpaceType.BATHROOM, SpaceType.UTILITY]
        chosen = self.rng.choice(len(types), p=probs)
        result = types[chosen]

        # Constraint: bathrooms only attach to corridors or rooms
        if result == SpaceType.BATHROOM and neighbors:
            if all(n == SpaceType.BATHROOM for n in neighbors):
                result = SpaceType.ROOM
        return result

    def _room_dimensions(self, space_type: SpaceType) -> tuple[float, float]:
        if space_type == SpaceType.CORRIDOR:
            short = self.rng.uniform(self.cfg.min_room_size * 0.4,
                                      self.cfg.min_room_size * 0.7)
            long = self.rng.uniform(self.cfg.max_room_size,
                                     self.cfg.max_room_size * 1.5)
            return (long, short) if self.rng.random() > 0.5 else (short, long)
        elif space_type in (SpaceType.BATHROOM, SpaceType.UTILITY):
            s = self.rng.uniform(self.cfg.min_room_size * 0.5,
                                  self.cfg.min_room_size)
            return s, s * self.rng.uniform(0.8, 1.2)
        else:
            w = self.rng.uniform(self.cfg.min_room_size, self.cfg.max_room_size)
            h = self.rng.uniform(self.cfg.min_room_size, self.cfg.max_room_size)
            return w, h

    def _generate_room(self, is_first: bool = False) -> tuple[Polygon, SpaceType]:
        room_type = SpaceType.ROOM if is_first else self._pick_room_type()
        w, h = self._room_dimensions(room_type)

        if self.rng.random() < self.cfg.indent_prob:
            d = self.rng.uniform(*self.cfg.indent_depth_range)
            iw = self.rng.uniform(*self.cfg.indent_width_range)
            iw = min(iw, w * 0.4)
            d = min(d, h * 0.3)
            poly = create_indented_room(w, h, d, iw, self.rng)
        else:
            poly = create_rectangle(0, 0, w, h)

        return poly, room_type

    def _attach_new_room(
        self, existing: list[Polygon], types: list[SpaceType]
    ) -> tuple[Polygon | None, SpaceType]:
        # Weight: corridors get 3x, rooms with more external walls get more
        weights = []
        for i, (p, t) in enumerate(zip(existing, types)):
            ext_walls = classify_external_walls(i, existing)
            ext_count = sum(1 for _, is_ext in ext_walls if is_ext)
            w = ext_count
            if t == SpaceType.CORRIDOR:
                w *= 3
            weights.append(max(w, 0.1))

        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        for _ in range(self.cfg.max_attach_attempts):
            parent_idx = int(self.rng.choice(len(existing), p=probs))
            parent_poly = existing[parent_idx]
            parent_type = types[parent_idx]

            ext_walls = classify_external_walls(parent_idx, existing)
            external_walls = [(seg, is_ext) for seg, is_ext in ext_walls if is_ext]
            if not external_walls:
                continue

            wall_idx = self.rng.integers(0, len(external_walls))
            (seg_start, seg_end), _ = external_walls[wall_idx]

            new_type = self._pick_room_type(neighbors=[parent_type])
            w, h = self._room_dimensions(new_type)

            start = np.array(seg_start)
            end = np.array(seg_end)
            new_poly = attach_room_to_wall(start, end, w, h, self.rng)

            # Snap to grid
            snapped_coords = snap_to_grid(
                list(new_poly.exterior.coords), self.cfg.grid_size
            )
            new_poly = Polygon(snapped_coords)
            if not new_poly.is_valid:
                new_poly = new_poly.buffer(0)
                if new_poly.geom_type == "MultiPolygon":
                    continue

            # Check no overlap
            overlap = False
            for ep in existing:
                if polygons_overlap(new_poly, ep):
                    overlap = True
                    break
            if overlap:
                continue

            # Check max_extent
            from shapely.ops import unary_union
            union = unary_union(existing + [new_poly])
            minx, miny, maxx, maxy = union.bounds
            if (maxx - minx) > self.cfg.max_extent or \
               (maxy - miny) > self.cfg.max_extent:
                continue

            return new_poly, new_type

        # Fallback: couldn't place, skip
        return None, SpaceType.ROOM

    def _build_spaces(
        self, polys: list[Polygon], types: list[SpaceType]
    ) -> list[Space]:
        spaces = []
        for i, (poly, stype) in enumerate(zip(polys, types)):
            ext_walls_info = classify_external_walls(i, polys)
            walls = []
            for (seg_start, seg_end), is_ext in ext_walls_info:
                thickness = self.cfg.global_wall_thickness
                if self.rng.random() < self.cfg.wall_thickness_variation_prob:
                    mult = self.rng.uniform(*self.cfg.wall_thickness_multiplier_range)
                    thickness *= mult

                openings: list[Opening] = []
                seg_len = np.linalg.norm(
                    np.array(seg_end) - np.array(seg_start)
                )

                if is_ext and self.rng.random() < self.cfg.window_prob:
                    if seg_len > self.cfg.window_width * 1.5:
                        offset = self.rng.uniform(0.2, 0.8)
                        openings.append(Opening(
                            type=OpeningType.WINDOW,
                            width=self.cfg.window_width,
                            offset=round(offset, 3),
                        ))
                elif not is_ext and self.rng.random() < self.cfg.door_prob:
                    if seg_len > self.cfg.door_width * 1.5:
                        offset = self.rng.uniform(0.2, 0.8)
                        swing = self.rng.choice(["left", "right"])
                        openings.append(Opening(
                            type=OpeningType.DOOR,
                            width=self.cfg.door_width,
                            offset=round(offset, 3),
                            swing=str(swing),
                        ))

                walls.append(Wall(
                    p1=list(seg_start),
                    p2=list(seg_end),
                    thickness=round(thickness, 1),
                    is_external=is_ext,
                    openings=openings,
                ))

            coords = [list(c) for c in poly.exterior.coords[:-1]]
            spaces.append(Space(
                id=f"space_{i + 1:03d}",
                type=stype,
                polygon=coords,
                walls=walls,
            ))
        return spaces

    def _ensure_connectivity(
        self, spaces: list[Space], polys: list[Polygon]
    ) -> None:
        """Ensure all rooms are connected via doors (BFS check + fix)."""
        n = len(spaces)
        adj: dict[int, set[int]] = {i: set() for i in range(n)}

        # Build adjacency from existing doors on internal walls
        for i, s in enumerate(spaces):
            for w in s.walls:
                if not w.is_external and any(
                    o.type == OpeningType.DOOR for o in w.openings
                ):
                    for j, other_s in enumerate(spaces):
                        if j == i:
                            continue
                        for ow in other_s.walls:
                            if self._walls_match(w, ow):
                                adj[i].add(j)
                                adj[j].add(i)

        # BFS to find connected components
        visited = set()
        components: list[set[int]] = []
        for start in range(n):
            if start in visited:
                continue
            comp = set()
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in comp:
                    continue
                comp.add(node)
                visited.add(node)
                for nb in adj[node]:
                    if nb not in comp:
                        queue.append(nb)
            components.append(comp)

        # Connect components by adding doors on shared walls
        while len(components) > 1:
            c1 = components[0]
            merged = False
            for ci in range(1, len(components)):
                c2 = components[ci]
                for i in c1:
                    for j in c2:
                        shared = find_shared_walls(polys[i], polys[j])
                        if shared:
                            seg = shared[0]
                            seg_len = np.linalg.norm(
                                np.array(seg[1]) - np.array(seg[0])
                            )
                            if seg_len > self.cfg.door_width:
                                offset = self.rng.uniform(0.2, 0.8)
                                door = Opening(
                                    type=OpeningType.DOOR,
                                    width=self.cfg.door_width,
                                    offset=round(offset, 3),
                                    swing=str(self.rng.choice(["left", "right"])),
                                )
                                # Add door to matching wall in space i
                                for w in spaces[i].walls:
                                    if not w.is_external and self._seg_matches(
                                        w, seg
                                    ):
                                        w.openings.append(door)
                                        break
                                else:
                                    # Add as new internal wall
                                    spaces[i].walls.append(Wall(
                                        p1=list(seg[0]),
                                        p2=list(seg[1]),
                                        thickness=self.cfg.global_wall_thickness,
                                        is_external=False,
                                        openings=[door],
                                    ))
                                components[0] = c1 | c2
                                components.pop(ci)
                                merged = True
                                break
                    if merged:
                        break
                if merged:
                    break
            if not merged:
                break  # can't connect (no shared walls between components)

    @staticmethod
    def _walls_match(w1: Wall, w2: Wall) -> bool:
        return (w1.p1 == w2.p1 and w1.p2 == w2.p2) or \
               (w1.p1 == w2.p2 and w1.p2 == w2.p1)

    @staticmethod
    def _seg_matches(
        wall: Wall,
        seg: tuple[tuple[float, float], tuple[float, float]],
        tol: float = 10.0
    ) -> bool:
        def close(a: list[float], b: tuple[float, float]) -> bool:
            return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol
        return (close(wall.p1, seg[0]) and close(wall.p2, seg[1])) or \
               (close(wall.p1, seg[1]) and close(wall.p2, seg[0]))
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_generator.py -v
```

Expected: All PASS. Some tests may need tuning based on actual geometry — fix iteratively.

**Step 5: Commit**

```bash
git add src/floorplan/generator.py tests/test_generator.py
git commit -m "feat: implement floorplan generator with growing structure algorithm"
```

---

### Task 5: Style Configuration

**Files:**
- Create: `src/floorplan/styles.py`
- Create: `tests/test_styles.py`

**Step 1: Write failing tests**

```python
# tests/test_styles.py
import numpy as np
from floorplan.styles import StyleConfig, generate_style


def test_generate_style_deterministic():
    s1 = generate_style(np.random.default_rng(42))
    s2 = generate_style(np.random.default_rng(42))
    assert s1.wall_color == s2.wall_color
    assert s1.line_width == s2.line_width


def test_style_fields():
    s = generate_style(np.random.default_rng(42))
    assert 1 <= s.line_width <= 4
    assert len(s.wall_color) == 3
    assert len(s.bg_color) == 3
    assert s.door_style in ("arc", "gap", "arc_line")
    assert isinstance(s.show_dimensions, bool)
    assert isinstance(s.show_labels, bool)
    assert isinstance(s.fill_rooms, bool)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_styles.py -v
```

**Step 3: Implement styles**

```python
# src/floorplan/styles.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


WALL_PALETTES = [
    (0, 0, 0),         # black
    (50, 50, 50),      # dark gray
    (30, 30, 80),      # dark blue
    (60, 40, 20),      # brown
    (40, 40, 40),      # charcoal
]

BG_PALETTES = [
    (255, 255, 255),   # white
    (252, 248, 240),   # cream
    (240, 240, 240),   # light gray
    (248, 245, 235),   # parchment
]

ROOM_FILLS = {
    "room": [(230, 240, 255), (255, 245, 230), (240, 255, 240)],
    "corridor": [(245, 245, 245), (235, 235, 235)],
    "bathroom": [(220, 240, 255), (200, 230, 255)],
    "utility": [(255, 240, 220), (245, 235, 225)],
}


@dataclass
class StyleConfig:
    line_width: int
    wall_color: tuple[int, int, int]
    bg_color: tuple[int, int, int]
    door_style: str  # "arc", "gap", "arc_line"
    fill_rooms: bool
    show_dimensions: bool
    show_labels: bool
    furniture_density: float  # 0.0 to 1.0
    room_fill_colors: dict[str, tuple[int, int, int]]


def generate_style(rng: np.random.Generator) -> StyleConfig:
    line_width = int(rng.integers(1, 5))
    wall_color = WALL_PALETTES[int(rng.integers(0, len(WALL_PALETTES)))]
    bg_color = BG_PALETTES[int(rng.integers(0, len(BG_PALETTES)))]
    door_style = str(rng.choice(["arc", "gap", "arc_line"]))
    fill_rooms = bool(rng.random() < 0.6)
    show_dimensions = bool(rng.random() < 0.4)
    show_labels = bool(rng.random() < 0.5)
    furniture_density = float(rng.uniform(0.0, 1.0))

    room_fill_colors = {}
    for room_type, palette in ROOM_FILLS.items():
        idx = int(rng.integers(0, len(palette)))
        room_fill_colors[room_type] = palette[idx]

    return StyleConfig(
        line_width=line_width,
        wall_color=wall_color,
        bg_color=bg_color,
        door_style=door_style,
        fill_rooms=fill_rooms,
        show_dimensions=show_dimensions,
        show_labels=show_labels,
        furniture_density=furniture_density,
        room_fill_colors=room_fill_colors,
    )
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_styles.py -v
```

**Step 5: Commit**

```bash
git add src/floorplan/styles.py tests/test_styles.py
git commit -m "feat: add randomized style configuration for rendering"
```

---

### Task 6: Renderer — Walls, Doors, Windows

**Files:**
- Create: `src/floorplan/renderer.py`
- Create: `tests/test_renderer.py`

**Step 1: Write failing tests**

```python
# tests/test_renderer.py
import pytest
import numpy as np
from PIL import Image

from floorplan.generator import FloorplanGenerator, GeneratorConfig
from floorplan.renderer import FloorplanRenderer, RenderConfig
from floorplan.styles import generate_style


@pytest.fixture
def sample_floorplan():
    cfg = GeneratorConfig(num_rooms=4, seed=42)
    return FloorplanGenerator(cfg).generate()


def test_renderer_produces_image(sample_floorplan):
    style = generate_style(np.random.default_rng(42))
    renderer = FloorplanRenderer(RenderConfig(image_size=512))
    img = renderer.render(sample_floorplan, style)
    assert isinstance(img, Image.Image)
    assert img.size == (512, 512)
    assert img.mode == "RGB"


def test_renderer_different_sizes(sample_floorplan):
    style = generate_style(np.random.default_rng(42))
    for size in [256, 512, 1024]:
        renderer = FloorplanRenderer(RenderConfig(image_size=size))
        img = renderer.render(sample_floorplan, style)
        assert img.size == (size, size)


def test_renderer_deterministic(sample_floorplan):
    style = generate_style(np.random.default_rng(42))
    r1 = FloorplanRenderer(RenderConfig(image_size=256))
    r2 = FloorplanRenderer(RenderConfig(image_size=256))
    img1 = r1.render(sample_floorplan, style)
    img2 = r2.render(sample_floorplan, style)
    assert list(img1.getdata()) == list(img2.getdata())


def test_renderer_not_all_background(sample_floorplan):
    style = generate_style(np.random.default_rng(42))
    renderer = FloorplanRenderer(RenderConfig(image_size=512))
    img = renderer.render(sample_floorplan, style)
    pixels = list(img.getdata())
    bg = style.bg_color
    non_bg = [p for p in pixels if p != bg]
    assert len(non_bg) > 100  # walls drawn
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_renderer.py -v
```

**Step 3: Implement renderer**

```python
# src/floorplan/renderer.py
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

from floorplan.models import Floorplan, Space, Wall, Opening, OpeningType
from floorplan.styles import StyleConfig


@dataclass
class RenderConfig:
    image_size: int = 512
    margin: float = 0.1  # fraction of image as margin


class FloorplanRenderer:
    def __init__(self, config: RenderConfig):
        self.cfg = config

    def render(self, floorplan: Floorplan, style: StyleConfig) -> Image.Image:
        img = Image.new("RGB", (self.cfg.image_size, self.cfg.image_size),
                        color=style.bg_color)
        draw = ImageDraw.Draw(img)

        # Compute transform: floorplan coords (mm) -> pixel coords
        transform = self._compute_transform(floorplan)

        # Draw room fills
        if style.fill_rooms:
            for space in floorplan.spaces:
                self._draw_room_fill(draw, space, style, transform)

        # Draw walls
        for space in floorplan.spaces:
            for wall in space.walls:
                self._draw_wall(draw, wall, style, transform)

        # Draw openings (doors and windows) on top
        for space in floorplan.spaces:
            for wall in space.walls:
                for opening in wall.openings:
                    if opening.type == OpeningType.DOOR:
                        self._draw_door(draw, wall, opening, style, transform)
                    elif opening.type == OpeningType.WINDOW:
                        self._draw_window(draw, wall, opening, style, transform)

        # Draw dimension lines
        if style.show_dimensions:
            for space in floorplan.spaces:
                for wall in space.walls:
                    if wall.is_external:
                        self._draw_dimension(draw, wall, transform, style)

        # Draw labels
        if style.show_labels:
            for space in floorplan.spaces:
                self._draw_label(draw, space, style, transform)

        return img

    def _compute_transform(
        self, floorplan: Floorplan
    ) -> dict:
        """Compute scale and offset to fit floorplan in image."""
        all_x, all_y = [], []
        for s in floorplan.spaces:
            for pt in s.polygon:
                all_x.append(pt[0])
                all_y.append(pt[1])

        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        extent_x = max_x - min_x or 1
        extent_y = max_y - min_y or 1

        margin_px = self.cfg.image_size * self.cfg.margin
        available = self.cfg.image_size - 2 * margin_px
        scale = min(available / extent_x, available / extent_y)

        # Center
        offset_x = margin_px + (available - extent_x * scale) / 2 - min_x * scale
        offset_y = margin_px + (available - extent_y * scale) / 2 - min_y * scale

        return {"scale": scale, "offset_x": offset_x, "offset_y": offset_y}

    def _to_px(self, x: float, y: float, transform: dict) -> tuple[float, float]:
        return (
            x * transform["scale"] + transform["offset_x"],
            y * transform["scale"] + transform["offset_y"],
        )

    def _draw_room_fill(
        self, draw: ImageDraw.Draw, space: Space,
        style: StyleConfig, transform: dict,
    ) -> None:
        fill_color = style.room_fill_colors.get(space.type.value)
        if fill_color is None:
            return
        pts = [self._to_px(p[0], p[1], transform) for p in space.polygon]
        draw.polygon(pts, fill=fill_color)

    def _draw_wall(
        self, draw: ImageDraw.Draw, wall: Wall,
        style: StyleConfig, transform: dict,
    ) -> None:
        p1 = self._to_px(wall.p1[0], wall.p1[1], transform)
        p2 = self._to_px(wall.p2[0], wall.p2[1], transform)
        width = max(1, int(wall.thickness * transform["scale"] * 0.5))
        width = min(width, style.line_width * 3)
        draw.line([p1, p2], fill=style.wall_color, width=max(style.line_width, width))

    def _draw_door(
        self, draw: ImageDraw.Draw, wall: Wall, opening: Opening,
        style: StyleConfig, transform: dict,
    ) -> None:
        p1 = np.array(self._to_px(wall.p1[0], wall.p1[1], transform))
        p2 = np.array(self._to_px(wall.p2[0], wall.p2[1], transform))
        wall_vec = p2 - p1
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 1:
            return
        wall_dir = wall_vec / wall_len

        door_center = p1 + wall_vec * opening.offset
        door_width_px = opening.width * transform["scale"]
        half_w = door_width_px / 2

        start = door_center - wall_dir * half_w
        end = door_center + wall_dir * half_w

        # Clear wall behind door
        draw.line([tuple(start), tuple(end)], fill=style.bg_color,
                  width=style.line_width + 2)

        if style.door_style == "gap":
            return  # just the gap

        # Draw arc
        normal = np.array([-wall_dir[1], wall_dir[0]])
        hinge = start if opening.swing == "left" else end
        arc_radius = door_width_px

        angle_base = math.degrees(math.atan2(wall_dir[1], wall_dir[0]))
        if opening.swing == "left":
            bbox = [
                hinge[0] - arc_radius, hinge[1] - arc_radius,
                hinge[0] + arc_radius, hinge[1] + arc_radius,
            ]
            draw.arc(bbox, start=angle_base - 90, end=angle_base,
                     fill=style.wall_color, width=1)
        else:
            bbox = [
                hinge[0] - arc_radius, hinge[1] - arc_radius,
                hinge[0] + arc_radius, hinge[1] + arc_radius,
            ]
            draw.arc(bbox, start=angle_base, end=angle_base + 90,
                     fill=style.wall_color, width=1)

        if style.door_style == "arc_line":
            draw.line([tuple(start), tuple(end)], fill=style.wall_color, width=1)

    def _draw_window(
        self, draw: ImageDraw.Draw, wall: Wall, opening: Opening,
        style: StyleConfig, transform: dict,
    ) -> None:
        p1 = np.array(self._to_px(wall.p1[0], wall.p1[1], transform))
        p2 = np.array(self._to_px(wall.p2[0], wall.p2[1], transform))
        wall_vec = p2 - p1
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 1:
            return
        wall_dir = wall_vec / wall_len
        normal = np.array([-wall_dir[1], wall_dir[0]])

        center = p1 + wall_vec * opening.offset
        win_w = opening.width * transform["scale"]
        half_w = win_w / 2

        start = center - wall_dir * half_w
        end = center + wall_dir * half_w

        # 3 parallel lines
        for offset in [-2, 0, 2]:
            s = start + normal * offset
            e = end + normal * offset
            draw.line([tuple(s), tuple(e)], fill=style.wall_color, width=1)

    def _draw_dimension(
        self, draw: ImageDraw.Draw, wall: Wall,
        transform: dict, style: StyleConfig,
    ) -> None:
        p1 = np.array(self._to_px(wall.p1[0], wall.p1[1], transform))
        p2 = np.array(self._to_px(wall.p2[0], wall.p2[1], transform))
        wall_vec = p2 - p1
        wall_len = np.linalg.norm(wall_vec)
        if wall_len < 30:
            return
        wall_dir = wall_vec / wall_len
        normal = np.array([-wall_dir[1], wall_dir[0]])

        # Dimension line offset from wall
        offset = 12
        dp1 = p1 + normal * offset
        dp2 = p2 + normal * offset

        draw.line([tuple(dp1), tuple(dp2)], fill=style.wall_color, width=1)
        # Tick marks
        for dp in [dp1, dp2]:
            tick = normal * 4
            draw.line([tuple(dp - tick), tuple(dp + tick)],
                      fill=style.wall_color, width=1)

        # Length text in mm
        real_len = math.sqrt(
            (wall.p2[0] - wall.p1[0]) ** 2 +
            (wall.p2[1] - wall.p1[1]) ** 2
        )
        text = f"{int(real_len)}"
        mid = (dp1 + dp2) / 2
        try:
            draw.text(tuple(mid), text, fill=style.wall_color,
                      anchor="mm", font_size=8)
        except Exception:
            draw.text(tuple(mid), text, fill=style.wall_color)

    def _draw_label(
        self, draw: ImageDraw.Draw, space: Space,
        style: StyleConfig, transform: dict,
    ) -> None:
        labels = {
            "room": "Room", "corridor": "Corr.",
            "bathroom": "WC", "utility": "Util.",
        }
        label = labels.get(space.type.value, "")
        if not label:
            return

        poly = Polygon(space.polygon)
        centroid = poly.centroid
        cx, cy = self._to_px(centroid.x, centroid.y, transform)
        try:
            draw.text((cx, cy), label, fill=style.wall_color,
                      anchor="mm", font_size=10)
        except Exception:
            draw.text((cx, cy), label, fill=style.wall_color)
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_renderer.py -v
```

**Step 5: Commit**

```bash
git add src/floorplan/renderer.py tests/test_renderer.py
git commit -m "feat: implement floorplan renderer with walls, doors, windows, labels"
```

---

### Task 7: Furniture Placement

**Files:**
- Create: `src/floorplan/furniture.py`
- Create: `tests/test_furniture.py`

**Step 1: Write failing tests**

```python
# tests/test_furniture.py
import numpy as np
import pytest
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from floorplan.furniture import FurniturePlacer


def test_programmatic_furniture_draws():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    poly = Polygon([(20, 20), (180, 20), (180, 180), (20, 180)])
    placer = FurniturePlacer(furniture_dir=None)
    rng = np.random.default_rng(42)
    placer.place_furniture(draw, poly, "room", rng, density=0.8,
                           transform={"scale": 1.0, "offset_x": 0, "offset_y": 0})
    # Check some pixels changed
    pixels = list(img.getdata())
    white_count = sum(1 for p in pixels if p == (255, 255, 255))
    assert white_count < 200 * 200  # something was drawn


def test_no_furniture_at_zero_density():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    poly = Polygon([(20, 20), (180, 20), (180, 180), (20, 180)])
    placer = FurniturePlacer(furniture_dir=None)
    rng = np.random.default_rng(42)
    placer.place_furniture(draw, poly, "room", rng, density=0.0,
                           transform={"scale": 1.0, "offset_x": 0, "offset_y": 0})
    pixels = list(img.getdata())
    white_count = sum(1 for p in pixels if p == (255, 255, 255))
    assert white_count == 200 * 200
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_furniture.py -v
```

**Step 3: Implement furniture placer**

```python
# src/floorplan/furniture.py
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, box as shapely_box, Point


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

        area = room_poly.area
        max_items = max(0, int(density * 5))
        n_items = rng.integers(0, max_items + 1)

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
        # Shrink bounds to avoid wall overlap
        pad = (maxx - minx) * 0.15
        inner_minx = minx + pad
        inner_maxx = maxx - pad
        inner_miny = miny + pad
        inner_maxy = maxy - pad
        if inner_maxx <= inner_minx or inner_maxy <= inner_miny:
            return

        # Try placing
        for _ in range(10):
            cx = rng.uniform(inner_minx, inner_maxx)
            cy = rng.uniform(inner_miny, inner_maxy)

            if not room_poly.contains(Point(cx, cy)):
                continue

            # Draw shape based on room type
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
            elif shape == "l_shape":
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

        # Paste onto image (need to access underlying image)
        img = draw._image
        paste_x = px_cx - rotated.width // 2
        paste_y = px_cy - rotated.height // 2
        try:
            img.paste(rotated, (paste_x, paste_y), rotated)
        except Exception:
            pass
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_furniture.py -v
```

**Step 5: Commit**

```bash
git add src/floorplan/furniture.py tests/test_furniture.py
git commit -m "feat: add furniture placement with PNG + programmatic fallback"
```

---

### Task 8: COCO Annotations

**Files:**
- Create: `src/floorplan/annotations.py`
- Create: `tests/test_annotations.py`

**Step 1: Write failing tests**

```python
# tests/test_annotations.py
import json
import numpy as np
import pytest
from PIL import Image

from floorplan.generator import FloorplanGenerator, GeneratorConfig
from floorplan.annotations import COCOAnnotator


@pytest.fixture
def sample_floorplan():
    cfg = GeneratorConfig(num_rooms=4, seed=42)
    return FloorplanGenerator(cfg).generate()


def test_semantic_mask_shape(sample_floorplan):
    ann = COCOAnnotator(image_size=512)
    mask = ann.generate_semantic_mask(sample_floorplan)
    assert isinstance(mask, Image.Image)
    assert mask.size == (512, 512)
    assert mask.mode == "L"


def test_semantic_mask_has_classes(sample_floorplan):
    ann = COCOAnnotator(image_size=512)
    mask = ann.generate_semantic_mask(sample_floorplan)
    pixels = set(mask.getdata())
    assert 0 in pixels  # background
    assert len(pixels) > 1  # at least one non-bg class


def test_instance_mask_shape(sample_floorplan):
    ann = COCOAnnotator(image_size=512)
    mask = ann.generate_instance_mask(sample_floorplan)
    assert mask.size == (512, 512)
    assert mask.mode == "I"  # 32-bit integer


def test_coco_json_structure(sample_floorplan):
    ann = COCOAnnotator(image_size=512)
    coco = ann.generate_coco_annotation(sample_floorplan, image_id=1,
                                         file_name="test.png")
    assert "image" in coco
    assert "annotations" in coco
    assert coco["image"]["id"] == 1
    assert coco["image"]["width"] == 512
    assert len(coco["annotations"]) > 0
    for a in coco["annotations"]:
        assert "bbox" in a
        assert "segmentation" in a
        assert "category_id" in a
        assert len(a["bbox"]) == 4


def test_coco_categories():
    ann = COCOAnnotator(image_size=512)
    cats = ann.get_categories()
    assert len(cats) == 7  # wall, room, corridor, bathroom, utility, door, window
    names = {c["name"] for c in cats}
    assert "wall" in names
    assert "room" in names
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_annotations.py -v
```

**Step 3: Implement annotations**

```python
# src/floorplan/annotations.py
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon

from floorplan.models import Floorplan, OpeningType


# Semantic class IDs
CLASS_BG = 0
CLASS_WALL = 1
CLASS_ROOM = 2
CLASS_CORRIDOR = 3
CLASS_BATHROOM = 4
CLASS_UTILITY = 5
CLASS_DOOR = 6
CLASS_WINDOW = 7

SPACE_TYPE_TO_CLASS = {
    "room": CLASS_ROOM,
    "corridor": CLASS_CORRIDOR,
    "bathroom": CLASS_BATHROOM,
    "utility": CLASS_UTILITY,
}

CATEGORIES = [
    {"id": CLASS_WALL, "name": "wall", "supercategory": "structure"},
    {"id": CLASS_ROOM, "name": "room", "supercategory": "space"},
    {"id": CLASS_CORRIDOR, "name": "corridor", "supercategory": "space"},
    {"id": CLASS_BATHROOM, "name": "bathroom", "supercategory": "space"},
    {"id": CLASS_UTILITY, "name": "utility", "supercategory": "space"},
    {"id": CLASS_DOOR, "name": "door", "supercategory": "opening"},
    {"id": CLASS_WINDOW, "name": "window", "supercategory": "opening"},
]


class COCOAnnotator:
    def __init__(self, image_size: int = 512, margin: float = 0.1):
        self.image_size = image_size
        self.margin = margin

    def _compute_transform(self, floorplan: Floorplan) -> dict:
        all_x, all_y = [], []
        for s in floorplan.spaces:
            for pt in s.polygon:
                all_x.append(pt[0])
                all_y.append(pt[1])
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        extent_x = max_x - min_x or 1
        extent_y = max_y - min_y or 1
        margin_px = self.image_size * self.margin
        available = self.image_size - 2 * margin_px
        scale = min(available / extent_x, available / extent_y)
        offset_x = margin_px + (available - extent_x * scale) / 2 - min_x * scale
        offset_y = margin_px + (available - extent_y * scale) / 2 - min_y * scale
        return {"scale": scale, "offset_x": offset_x, "offset_y": offset_y}

    def _to_px(self, x: float, y: float, t: dict) -> tuple[float, float]:
        return (x * t["scale"] + t["offset_x"], y * t["scale"] + t["offset_y"])

    def generate_semantic_mask(self, floorplan: Floorplan) -> Image.Image:
        mask = Image.new("L", (self.image_size, self.image_size), CLASS_BG)
        draw = ImageDraw.Draw(mask)
        t = self._compute_transform(floorplan)

        # Draw room fills
        for space in floorplan.spaces:
            class_id = SPACE_TYPE_TO_CLASS.get(space.type.value, CLASS_ROOM)
            pts = [self._to_px(p[0], p[1], t) for p in space.polygon]
            draw.polygon(pts, fill=class_id)

        # Draw walls on top
        for space in floorplan.spaces:
            for wall in space.walls:
                p1 = self._to_px(wall.p1[0], wall.p1[1], t)
                p2 = self._to_px(wall.p2[0], wall.p2[1], t)
                width = max(2, int(wall.thickness * t["scale"] * 0.3))
                draw.line([p1, p2], fill=CLASS_WALL, width=width)

        # Draw openings
        for space in floorplan.spaces:
            for wall in space.walls:
                for opening in wall.openings:
                    p1 = np.array(self._to_px(wall.p1[0], wall.p1[1], t))
                    p2 = np.array(self._to_px(wall.p2[0], wall.p2[1], t))
                    vec = p2 - p1
                    center = p1 + vec * opening.offset
                    length = np.linalg.norm(vec)
                    if length < 1:
                        continue
                    direction = vec / length
                    half_w = opening.width * t["scale"] / 2
                    s = center - direction * half_w
                    e = center + direction * half_w
                    cls = CLASS_DOOR if opening.type == OpeningType.DOOR else CLASS_WINDOW
                    draw.line([tuple(s), tuple(e)], fill=cls, width=3)

        return mask

    def generate_instance_mask(self, floorplan: Floorplan) -> Image.Image:
        mask = Image.new("I", (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(mask)
        t = self._compute_transform(floorplan)

        for i, space in enumerate(floorplan.spaces):
            pts = [self._to_px(p[0], p[1], t) for p in space.polygon]
            draw.polygon(pts, fill=i + 1)

        return mask

    def generate_coco_annotation(
        self, floorplan: Floorplan, image_id: int, file_name: str,
    ) -> dict:
        t = self._compute_transform(floorplan)
        annotations = []
        ann_id = 1

        for space in floorplan.spaces:
            class_id = SPACE_TYPE_TO_CLASS.get(space.type.value, CLASS_ROOM)
            pts_px = [self._to_px(p[0], p[1], t) for p in space.polygon]

            # Flatten for COCO segmentation format
            seg = []
            for px, py in pts_px:
                seg.extend([round(px, 1), round(py, 1)])

            xs = [p[0] for p in pts_px]
            ys = [p[1] for p in pts_px]
            x_min, y_min = min(xs), min(ys)
            w = max(xs) - x_min
            h = max(ys) - y_min

            poly = Polygon(pts_px)

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": class_id,
                "segmentation": [seg],
                "bbox": [round(x_min, 1), round(y_min, 1), round(w, 1), round(h, 1)],
                "area": round(poly.area, 1),
                "iscrowd": 0,
            })
            ann_id += 1

        return {
            "image": {
                "id": image_id,
                "file_name": file_name,
                "width": self.image_size,
                "height": self.image_size,
            },
            "annotations": annotations,
        }

    @staticmethod
    def get_categories() -> list[dict]:
        return CATEGORIES

    @staticmethod
    def build_coco_dataset(
        image_annotations: list[dict],
    ) -> dict:
        """Merge per-image annotations into a single COCO dataset JSON."""
        images = []
        all_anns = []
        for item in image_annotations:
            images.append(item["image"])
            all_anns.extend(item["annotations"])
        return {
            "images": images,
            "annotations": all_anns,
            "categories": CATEGORIES,
        }
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_annotations.py -v
```

**Step 5: Commit**

```bash
git add src/floorplan/annotations.py tests/test_annotations.py
git commit -m "feat: add COCO annotation generator (semantic/instance masks + JSON)"
```

---

### Task 9: CLI — floorplan_generator.py

**Files:**
- Create: `scripts/floorplan_generator.py`
- Create: `tests/test_cli_generator.py`

**Step 1: Write failing tests**

```python
# tests/test_cli_generator.py
import json
import os
import tempfile
import pytest
from click.testing import CliRunner

from scripts.floorplan_generator import cli


def test_cli_generates_json_files():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(cli, [
            "--count", "3", "--seed", "42",
            "--num-rooms", "3-5",
            "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0, result.output
        files = sorted(os.listdir(tmpdir))
        assert len(files) == 3
        assert all(f.endswith(".json") for f in files)
        # Validate JSON
        with open(os.path.join(tmpdir, files[0])) as f:
            data = json.load(f)
        assert "meta" in data
        assert "spaces" in data


def test_cli_reproducible():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as d1, \
         tempfile.TemporaryDirectory() as d2:
        runner.invoke(cli, ["--count", "2", "--seed", "42", "--output-dir", d1])
        runner.invoke(cli, ["--count", "2", "--seed", "42", "--output-dir", d2])
        for fname in os.listdir(d1):
            with open(os.path.join(d1, fname)) as f1, \
                 open(os.path.join(d2, fname)) as f2:
                assert json.load(f1) == json.load(f2)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli_generator.py -v
```

**Step 3: Implement CLI**

```python
# scripts/floorplan_generator.py
"""CLI for batch floorplan JSON generation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from tqdm import tqdm

# Add src to path for script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floorplan.generator import FloorplanGenerator, GeneratorConfig


def parse_range(value: str) -> tuple[int, int]:
    if "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    v = int(value)
    return v, v


@click.command()
@click.option("--count", type=int, required=True, help="Number of floorplans")
@click.option("--seed", type=int, default=42, help="Base random seed")
@click.option("--num-rooms", type=str, default="3-8",
              help="Room count range (e.g. 3-8)")
@click.option("--output-dir", type=click.Path(), required=True,
              help="Output directory for JSON files")
@click.option("--indent-prob", type=float, default=0.2)
@click.option("--door-prob", type=float, default=0.7)
@click.option("--window-prob", type=float, default=0.5)
@click.option("--wall-thickness", type=float, default=120,
              help="Global wall thickness in mm")
def cli(
    count: int, seed: int, num_rooms: str, output_dir: str,
    indent_prob: float, door_prob: float, window_prob: float,
    wall_thickness: float,
) -> None:
    """Generate synthetic floorplan JSON files."""
    import numpy as np

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lo, hi = parse_range(num_rooms)
    rng = np.random.default_rng(seed)

    for i in tqdm(range(count), desc="Generating floorplans"):
        n = int(rng.integers(lo, hi + 1))
        item_seed = seed + i

        cfg = GeneratorConfig(
            num_rooms=n,
            seed=item_seed,
            indent_prob=indent_prob,
            door_prob=door_prob,
            window_prob=window_prob,
            global_wall_thickness=wall_thickness,
        )

        fp = FloorplanGenerator(cfg).generate()
        fname = out / f"floorplan_{i:05d}.json"
        fname.write_text(fp.model_dump_json(indent=2))

    click.echo(f"Generated {count} floorplans in {out}")


if __name__ == "__main__":
    cli()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_cli_generator.py -v
```

**Step 5: Commit**

```bash
git add scripts/floorplan_generator.py tests/test_cli_generator.py
git commit -m "feat: add CLI for batch floorplan JSON generation"
```

---

### Task 10: CLI — generate_synthetic.py

**Files:**
- Create: `scripts/generate_synthetic.py`
- Create: `tests/test_cli_synthetic.py`

**Step 1: Write failing tests**

```python
# tests/test_cli_synthetic.py
import json
import os
import tempfile
import pytest
from click.testing import CliRunner

from scripts.floorplan_generator import cli as gen_cli
from scripts.generate_synthetic import cli as synth_cli


@pytest.fixture
def json_dir():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(gen_cli, [
            "--count", "3", "--seed", "42",
            "--num-rooms", "3-5", "--output-dir", tmpdir,
        ])
        assert result.exit_code == 0
        yield tmpdir


def test_render_produces_images(json_dir):
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as outdir:
        result = runner.invoke(synth_cli, [
            "--input-dir", json_dir,
            "--output-dir", outdir,
            "--image-size", "256",
            "--workers", "1",
        ])
        assert result.exit_code == 0, result.output
        images = [f for f in os.listdir(os.path.join(outdir, "images"))
                  if f.endswith(".png")]
        assert len(images) == 3


def test_render_with_masks(json_dir):
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as outdir:
        result = runner.invoke(synth_cli, [
            "--input-dir", json_dir,
            "--output-dir", outdir,
            "--image-size", "256",
            "--workers", "1",
            "--with-masks",
        ])
        assert result.exit_code == 0
        masks = os.listdir(os.path.join(outdir, "masks"))
        assert len(masks) == 3


def test_render_with_coco(json_dir):
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as outdir:
        result = runner.invoke(synth_cli, [
            "--input-dir", json_dir,
            "--output-dir", outdir,
            "--image-size", "256",
            "--workers", "1",
            "--with-coco",
        ])
        assert result.exit_code == 0
        coco_path = os.path.join(outdir, "annotations", "coco.json")
        assert os.path.exists(coco_path)
        with open(coco_path) as f:
            coco = json.load(f)
        assert len(coco["images"]) == 3
        assert len(coco["annotations"]) > 0
        assert len(coco["categories"]) == 7
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli_synthetic.py -v
```

**Step 3: Implement CLI**

```python
# scripts/generate_synthetic.py
"""CLI for batch rendering floorplan images + annotations."""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path
from functools import partial

import click
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floorplan.models import Floorplan
from floorplan.renderer import FloorplanRenderer, RenderConfig
from floorplan.annotations import COCOAnnotator
from floorplan.furniture import FurniturePlacer
from floorplan.styles import generate_style


def render_single(
    args: tuple[Path, int],
    output_dir: Path,
    image_size: int,
    with_masks: bool,
    with_coco: bool,
    furniture_dir: str | None,
) -> dict | None:
    json_path, idx = args

    fp = Floorplan.model_validate_json(json_path.read_text())
    seed = fp.meta.seed
    rng = np.random.default_rng(seed)

    style = generate_style(rng)
    renderer = FloorplanRenderer(RenderConfig(image_size=image_size))
    img = renderer.render(fp, style)

    # Furniture
    placer = FurniturePlacer(furniture_dir=furniture_dir)
    from PIL import ImageDraw
    from shapely.geometry import Polygon
    draw = ImageDraw.Draw(img)
    transform = renderer._compute_transform(fp)
    for space in fp.spaces:
        poly = Polygon(space.polygon)
        placer.place_furniture(
            draw, poly, space.type.value, rng,
            density=style.furniture_density, transform=transform,
        )

    stem = json_path.stem
    img_path = output_dir / "images" / f"{stem}.png"
    img.save(img_path)

    result = None

    if with_masks:
        ann = COCOAnnotator(image_size=image_size)
        sem_mask = ann.generate_semantic_mask(fp)
        sem_mask.save(output_dir / "masks" / f"{stem}.png")

    if with_coco:
        ann = COCOAnnotator(image_size=image_size)
        result = ann.generate_coco_annotation(
            fp, image_id=idx + 1, file_name=f"{stem}.png"
        )

    return result


@click.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True,
              help="Directory with floorplan JSON files")
@click.option("--output-dir", type=click.Path(), required=True,
              help="Output directory")
@click.option("--image-size", type=int, default=512)
@click.option("--workers", type=int, default=1,
              help="Number of parallel workers")
@click.option("--with-masks", is_flag=True, help="Generate segmentation masks")
@click.option("--with-coco", is_flag=True, help="Generate COCO annotations")
@click.option("--furniture-dir", type=click.Path(), default=None,
              help="Path to furniture PNG icons directory")
def cli(
    input_dir: str, output_dir: str, image_size: int,
    workers: int, with_masks: bool, with_coco: bool,
    furniture_dir: str | None,
) -> None:
    """Render floorplan images from JSON descriptions."""
    inp = Path(input_dir)
    out = Path(output_dir)

    (out / "images").mkdir(parents=True, exist_ok=True)
    if with_masks:
        (out / "masks").mkdir(parents=True, exist_ok=True)
    if with_coco:
        (out / "annotations").mkdir(parents=True, exist_ok=True)

    json_files = sorted(inp.glob("*.json"))
    if not json_files:
        click.echo("No JSON files found.")
        return

    args_list = [(f, i) for i, f in enumerate(json_files)]

    fn = partial(
        render_single,
        output_dir=out,
        image_size=image_size,
        with_masks=with_masks,
        with_coco=with_coco,
        furniture_dir=furniture_dir,
    )

    coco_results = []

    if workers <= 1:
        for a in tqdm(args_list, desc="Rendering"):
            result = fn(a)
            if result:
                coco_results.append(result)
    else:
        with Pool(workers) as pool:
            for result in tqdm(
                pool.imap(fn, args_list),
                total=len(args_list), desc="Rendering",
            ):
                if result:
                    coco_results.append(result)

    if with_coco and coco_results:
        from floorplan.annotations import COCOAnnotator
        coco_dataset = COCOAnnotator.build_coco_dataset(coco_results)
        coco_path = out / "annotations" / "coco.json"
        coco_path.write_text(json.dumps(coco_dataset, indent=2))

    click.echo(f"Rendered {len(json_files)} images to {out}")


if __name__ == "__main__":
    cli()
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_cli_synthetic.py -v
```

**Step 5: Commit**

```bash
git add scripts/generate_synthetic.py tests/test_cli_synthetic.py
git commit -m "feat: add CLI for batch image rendering with masks and COCO output"
```

---

### Task 11: Integration Test — End-to-End Pipeline

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end pipeline test: generate JSON -> render images -> verify annotations."""
import json
import os
import tempfile
import pytest
from click.testing import CliRunner
from PIL import Image

from scripts.floorplan_generator import cli as gen_cli
from scripts.generate_synthetic import cli as synth_cli


def test_full_pipeline():
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as workdir:
        json_dir = os.path.join(workdir, "json")
        output_dir = os.path.join(workdir, "output")

        # Step 1: Generate JSONs
        result = runner.invoke(gen_cli, [
            "--count", "5", "--seed", "42",
            "--num-rooms", "3-6",
            "--output-dir", json_dir,
        ])
        assert result.exit_code == 0, f"Generator failed: {result.output}"
        json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
        assert len(json_files) == 5

        # Step 2: Render with all outputs
        result = runner.invoke(synth_cli, [
            "--input-dir", json_dir,
            "--output-dir", output_dir,
            "--image-size", "512",
            "--workers", "1",
            "--with-masks",
            "--with-coco",
        ])
        assert result.exit_code == 0, f"Renderer failed: {result.output}"

        # Verify images
        img_dir = os.path.join(output_dir, "images")
        images = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        assert len(images) == 5
        img = Image.open(os.path.join(img_dir, images[0]))
        assert img.size == (512, 512)

        # Verify masks
        mask_dir = os.path.join(output_dir, "masks")
        masks = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
        assert len(masks) == 5
        mask = Image.open(os.path.join(mask_dir, masks[0]))
        assert mask.mode == "L"

        # Verify COCO
        coco_path = os.path.join(output_dir, "annotations", "coco.json")
        assert os.path.exists(coco_path)
        with open(coco_path) as f:
            coco = json.load(f)
        assert len(coco["images"]) == 5
        assert len(coco["categories"]) == 7
        assert len(coco["annotations"]) >= 5  # at least 1 per image

        # Verify annotation references valid images
        image_ids = {im["id"] for im in coco["images"]}
        for ann in coco["annotations"]:
            assert ann["image_id"] in image_ids
            assert ann["bbox"][2] > 0  # width > 0
            assert ann["bbox"][3] > 0  # height > 0
```

**Step 2: Run integration test**

```bash
uv run pytest tests/test_integration.py -v
```

Expected: PASS — full pipeline works.

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test for full pipeline"
```

---

### Task 12: Run Full Test Suite & Fix Issues

**Step 1: Run all tests**

```bash
uv run pytest tests/ -v --tb=short
```

**Step 2: Fix any failures iteratively**

Debug and fix until all tests pass.

**Step 3: Run with coverage**

```bash
uv run pytest tests/ --cov=floorplan --cov-report=term-missing
```

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve test failures and edge cases"
```

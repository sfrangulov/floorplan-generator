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
    # Find connections through shared internal walls with doors
    for s in fp.spaces:
        for w in s.walls:
            if not w.is_external and any(o.type.value == "door" for o in w.openings):
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
    assert SpaceType.LIVING_ROOM in types


def test_max_extent_respected():
    cfg = GeneratorConfig(num_rooms=8, seed=42, max_extent=20000)
    fp = FloorplanGenerator(cfg).generate()
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    union = unary_union([Polygon(s.polygon) for s in fp.spaces])
    minx, miny, maxx, maxy = union.bounds
    assert (maxx - minx) <= 20000 + 500  # small tolerance
    assert (maxy - miny) <= 20000 + 500

# tests/test_generator.py
import pytest
import json
from floorplan.generator import FloorplanGenerator, GeneratorConfig
from floorplan.models import Floorplan, SpaceType, OpeningType


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


# ------------------------------------------------------------------ #
# Layout-type generation tests
# ------------------------------------------------------------------ #


def test_layout_type_config():
    """layout_type defaults to None and can be set."""
    cfg = GeneratorConfig()
    assert cfg.layout_type is None

    cfg2 = GeneratorConfig(layout_type="studio")
    assert cfg2.layout_type == "studio"


def test_layout_type_studio_generates():
    """Studio layout produces a valid floorplan with at least 3 rooms."""
    cfg = GeneratorConfig(layout_type="studio", seed=42)
    fp = FloorplanGenerator(cfg).generate()
    assert isinstance(fp, Floorplan)
    assert len(fp.spaces) >= 3  # hallway + bathroom + living_room minimum


def test_layout_type_1room_generates():
    """1room layout produces a floorplan with expected rooms."""
    cfg = GeneratorConfig(layout_type="1room", seed=42)
    fp = FloorplanGenerator(cfg).generate()
    assert len(fp.spaces) >= 4  # hallway + bathroom + kitchen + living_room


def test_layout_type_2room_generates():
    """2room layout produces a floorplan."""
    cfg = GeneratorConfig(layout_type="2room", seed=42)
    fp = FloorplanGenerator(cfg).generate()
    assert len(fp.spaces) >= 5


def test_layout_type_3room_generates():
    """3room layout generates successfully."""
    cfg = GeneratorConfig(layout_type="3room", seed=42)
    fp = FloorplanGenerator(cfg).generate()
    assert len(fp.spaces) >= 5


def test_layout_type_house_medium_generates():
    """house_medium layout generates successfully."""
    cfg = GeneratorConfig(layout_type="house_medium", seed=42)
    fp = FloorplanGenerator(cfg).generate()
    assert len(fp.spaces) >= 5


def test_layout_type_no_overlap():
    """Layout-generated rooms should not overlap."""
    cfg = GeneratorConfig(layout_type="2room", seed=42)
    fp = FloorplanGenerator(cfg).generate()
    from shapely.geometry import Polygon
    polys = [Polygon(s.polygon) for s in fp.spaces]
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            inter = polys[i].intersection(polys[j])
            assert inter.area < 1.0, f"Rooms {i} and {j} overlap"


def test_layout_type_reproducible():
    """Layout generation is reproducible with the same seed."""
    cfg = GeneratorConfig(layout_type="1room", seed=42)
    fp1 = FloorplanGenerator(cfg).generate()
    fp2 = FloorplanGenerator(cfg).generate()
    assert fp1.model_dump() == fp2.model_dump()


def test_layout_type_hallway_first():
    """The first room in layout generation should be a hallway."""
    cfg = GeneratorConfig(layout_type="studio", seed=42)
    fp = FloorplanGenerator(cfg).generate()
    assert fp.spaces[0].type == SpaceType.HALLWAY


def test_layout_type_json_roundtrip():
    """Layout-generated floorplan survives JSON roundtrip."""
    cfg = GeneratorConfig(layout_type="2room", seed=42)
    fp = FloorplanGenerator(cfg).generate()
    json_str = fp.model_dump_json()
    fp2 = Floorplan.model_validate_json(json_str)
    assert len(fp2.spaces) == len(fp.spaces)


def test_layout_type_windows_only_on_external():
    """Windows should only appear on external walls in layout mode."""
    cfg = GeneratorConfig(layout_type="3room", seed=42)
    fp = FloorplanGenerator(cfg).generate()
    for s in fp.spaces:
        for w in s.walls:
            for o in w.openings:
                if o.type.value == "window":
                    assert w.is_external, (
                        f"Window on internal wall in {s.id} ({s.type.value})"
                    )


def test_layout_type_living_room_has_window():
    """Living rooms (must_have_window) should have at least one window."""
    cfg = GeneratorConfig(layout_type="1room", seed=42, window_prob=1.0)
    fp = FloorplanGenerator(cfg).generate()
    for s in fp.spaces:
        if s.type == SpaceType.LIVING_ROOM:
            ext_walls = [w for w in s.walls if w.is_external]
            if ext_walls:
                has_window = any(
                    o.type == OpeningType.WINDOW
                    for w in ext_walls
                    for o in w.openings
                )
                assert has_window, "Living room with external walls should have a window"


def test_layout_type_door_widths_from_spec():
    """Door widths placed by _place_openings should be within ROOM_SPECS ranges.

    Note: doors added by _ensure_connectivity use cfg.door_width (900) and are
    excluded from this check since they serve a different purpose.
    """
    from floorplan.layout_templates import ROOM_SPECS

    cfg = GeneratorConfig(layout_type="2room", seed=42, door_prob=1.0)
    fp = FloorplanGenerator(cfg).generate()

    for s in fp.spaces:
        spec = ROOM_SPECS.get(s.type.value)
        if spec is None:
            continue
        for w in s.walls:
            if not w.is_external:
                for o in w.openings:
                    if o.type == OpeningType.DOOR:
                        # Allow connectivity-enforcement doors (cfg.door_width=900)
                        # and spec-based doors
                        in_spec = spec.door_width_min - 1 <= o.width <= spec.door_width_max + 1
                        is_connectivity_door = abs(o.width - cfg.door_width) < 1
                        assert in_spec or is_connectivity_door, (
                            f"Door in {s.type.value} has width {o.width}, "
                            f"expected {spec.door_width_min}-{spec.door_width_max} "
                            f"or {cfg.door_width} (connectivity)"
                        )


def test_random_mode_backward_compat():
    """Without layout_type, the generator should behave identically to before."""
    cfg = GeneratorConfig(num_rooms=5, seed=42)
    fp = FloorplanGenerator(cfg).generate()
    assert isinstance(fp, Floorplan)
    assert len(fp.spaces) == 5
    # No hallway required in random mode
    types = {s.type for s in fp.spaces}
    assert len(types) >= 1


def test_all_layout_types_generate():
    """Every layout type from LAYOUT_TEMPLATES should produce a valid floorplan."""
    from floorplan.layout_templates import LAYOUT_TEMPLATES

    for lt in LAYOUT_TEMPLATES:
        cfg = GeneratorConfig(layout_type=lt, seed=42)
        fp = FloorplanGenerator(cfg).generate()
        assert isinstance(fp, Floorplan), f"Failed for layout_type={lt}"
        assert len(fp.spaces) >= 2, f"Too few rooms for layout_type={lt}"

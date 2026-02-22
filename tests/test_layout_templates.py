import numpy as np
import pytest

from floorplan.layout_templates import LAYOUT_TEMPLATES, ROOM_SPECS, get_room_list


def test_all_template_types_exist():
    expected = {"studio", "1room", "2room", "3room", "4room",
                "house_small", "house_medium", "house_large"}
    assert set(LAYOUT_TEMPLATES.keys()) == expected


def test_get_room_list_deterministic():
    rng = np.random.default_rng(42)
    rooms = get_room_list("2room", rng)
    assert "hallway" in rooms
    assert "bathroom" in rooms
    assert "kitchen" in rooms
    assert "living_room" in rooms
    assert "bedroom" in rooms


def test_room_specs_have_all_types():
    from floorplan.models import SpaceType
    for st in SpaceType:
        assert st.value in ROOM_SPECS, f"Missing spec for {st.value}"


def test_studio_has_no_separate_bedroom():
    rng = np.random.default_rng(42)
    rooms = get_room_list("studio", rng)
    assert "bedroom" not in rooms


def test_hallway_always_first():
    rng = np.random.default_rng(42)
    for layout_type in LAYOUT_TEMPLATES:
        rooms = get_room_list(layout_type, rng)
        assert rooms[0] == "hallway"


def test_room_list_reproducible():
    rooms1 = get_room_list("3room", np.random.default_rng(42))
    rooms2 = get_room_list("3room", np.random.default_rng(42))
    assert rooms1 == rooms2

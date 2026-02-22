# tests/test_adjacency.py
from floorplan.adjacency import can_connect, get_allowed_parents, MUST_HAVE_EXTERNAL_WALL


def test_kitchen_never_connects_to_bedroom():
    assert not can_connect("kitchen", "bedroom")
    assert not can_connect("bedroom", "kitchen")


def test_hallway_connects_to_kitchen():
    assert can_connect("hallway", "kitchen")


def test_bathroom_never_connects_to_living_room():
    assert not can_connect("bathroom", "living_room")
    assert not can_connect("living_room", "bathroom")


def test_bathroom_never_connects_to_bathroom():
    assert not can_connect("bathroom", "bathroom")


def test_bedroom_never_connects_to_bedroom():
    assert not can_connect("bedroom", "bedroom")


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


def test_hallway_has_no_parents():
    assert get_allowed_parents("hallway") == []


def test_must_have_external_wall():
    assert "living_room" in MUST_HAVE_EXTERNAL_WALL
    assert "bedroom" in MUST_HAVE_EXTERNAL_WALL
    assert "kitchen" in MUST_HAVE_EXTERNAL_WALL
    assert "corridor" not in MUST_HAVE_EXTERNAL_WALL
    assert "hallway" not in MUST_HAVE_EXTERNAL_WALL

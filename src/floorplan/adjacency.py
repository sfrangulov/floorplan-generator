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

# Allowed parent rooms for each type (where it can be attached)
ALLOWED_PARENTS: dict[str, list[str]] = {
    "hallway":     [],
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

# Rooms that MUST have at least one external wall
MUST_HAVE_EXTERNAL_WALL: set[str] = {
    "living_room", "bedroom", "kitchen", "balcony", "terrace", "garage",
}


def can_connect(type_a: str, type_b: str) -> bool:
    """Check if two room types are allowed to have a door between them."""
    return frozenset({type_a, type_b}) not in NEVER_CONNECT


def get_allowed_parents(room_type: str) -> list[str]:
    """Return list of room types this room can be attached to."""
    return ALLOWED_PARENTS.get(room_type, [])

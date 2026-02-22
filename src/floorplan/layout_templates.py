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
    may_have_window: bool


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
    required: list[str]
    optional: list[str]
    optional_prob: float = 0.5


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


_GENERATION_ORDER: dict[str, int] = {
    "hallway": 0, "kitchen": 1, "bathroom": 2, "toilet": 3,
    "corridor": 4, "living_room": 5, "bedroom": 6, "storage": 7,
    "utility": 8, "balcony": 9, "terrace": 10, "garage": 11,
}


def get_room_list(layout_type: str, rng: np.random.Generator) -> list[str]:
    """Return ordered list of room types for the given layout."""
    template = LAYOUT_TEMPLATES[layout_type]
    rooms = list(template.required)

    for opt in template.optional:
        if rng.random() < template.optional_prob:
            rooms.append(opt)

    rooms.sort(key=lambda r: _GENERATION_ORDER.get(r, 99))
    return rooms

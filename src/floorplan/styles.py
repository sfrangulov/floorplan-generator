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
    "hallway": [(245, 245, 245), (235, 235, 235)],
    "corridor": [(245, 245, 245), (235, 235, 235)],
    "living_room": [(230, 240, 255), (255, 245, 230), (240, 255, 240)],
    "bedroom": [(230, 230, 250), (245, 235, 245)],
    "kitchen": [(255, 248, 220), (250, 240, 210)],
    "bathroom": [(220, 240, 255), (200, 230, 255)],
    "toilet": [(220, 240, 255), (210, 235, 250)],
    "balcony": [(240, 255, 240), (230, 250, 230)],
    "storage": [(240, 235, 225), (235, 230, 220)],
    "utility": [(255, 240, 220), (245, 235, 225)],
    "garage": [(230, 230, 230), (220, 220, 220)],
    "terrace": [(240, 255, 240), (235, 250, 235)],
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

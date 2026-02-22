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

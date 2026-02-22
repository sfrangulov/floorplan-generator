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
        type=SpaceType.LIVING_ROOM,
        polygon=[[0, 0], [3000, 0], [3000, 4000], [0, 4000]],
        walls=[]
    )
    assert s.type == SpaceType.LIVING_ROOM
    assert len(s.polygon) == 4


def test_floorplan_roundtrip_json():
    fp = Floorplan(
        meta=FloorplanMeta(seed=42, global_wall_thickness=120),
        spaces=[
            Space(
                id="space_001",
                type=SpaceType.LIVING_ROOM,
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


def test_legacy_room_type_migration():
    space = Space(id="s1", type="room", polygon=[[0, 0], [1, 0], [1, 1]])
    assert space.type == SpaceType.LIVING_ROOM


def test_floorplan_invalid_offset_rejected():
    with pytest.raises(Exception):
        Opening(type=OpeningType.DOOR, width=900, offset=1.5)

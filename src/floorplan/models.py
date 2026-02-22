# src/floorplan/models.py
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SpaceType(str, Enum):
    HALLWAY = "hallway"
    CORRIDOR = "corridor"
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    TOILET = "toilet"
    BALCONY = "balcony"
    STORAGE = "storage"
    UTILITY = "utility"
    GARAGE = "garage"
    TERRACE = "terrace"


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

    @field_validator("type", mode="before")
    @classmethod
    def migrate_legacy_type(cls, v):
        if v == "room":
            return "living_room"
        return v


class FloorplanMeta(BaseModel):
    seed: int
    global_wall_thickness: float = Field(default=120, gt=0)
    units: str = "mm"


class Floorplan(BaseModel):
    meta: FloorplanMeta
    spaces: list[Space]

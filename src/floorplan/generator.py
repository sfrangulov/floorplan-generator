# src/floorplan/generator.py
"""Floorplan generator using a growing-structure algorithm."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from floorplan.geometry import (
    create_rectangle,
    create_indented_room,
    attach_room_to_wall,
    find_shared_walls,
    classify_external_walls,
    polygons_overlap,
    snap_to_grid,
)
from floorplan.models import (
    Floorplan,
    FloorplanMeta,
    Space,
    Wall,
    Opening,
    SpaceType,
    OpeningType,
)
from floorplan.layout_templates import ROOM_SPECS, get_room_list
from floorplan.adjacency import (
    ALLOWED_PARENTS,
    MUST_HAVE_EXTERNAL_WALL,
    can_connect,
    get_allowed_parents,
)


def _attach_room_outward(
    wall_start: np.ndarray,
    wall_end: np.ndarray,
    new_width: float,
    new_height: float,
    parent_poly: Polygon,
    rng: np.random.Generator,
) -> Polygon:
    """Attach a room to a wall segment, ensuring it extends outward from the parent.

    Tries the default normal direction first; if the resulting polygon overlaps
    with the parent, flips the normal to the opposite side.
    """
    wall_vec = wall_end - wall_start
    wall_length = float(np.linalg.norm(wall_vec))
    if wall_length < 1e-6:
        return Polygon()
    wall_dir = wall_vec / wall_length

    max_offset = max(0, wall_length - new_width)
    offset = rng.uniform(0, max_offset) if max_offset > 0 else 0

    clamped_w = min(new_width, wall_length)

    for sign in (1.0, -1.0):
        normal = sign * np.array([wall_dir[1], -wall_dir[0]])
        p0 = wall_start + wall_dir * offset
        p1 = p0 + wall_dir * clamped_w
        p2 = p1 + normal * new_height
        p3 = p0 + normal * new_height
        coords = [tuple(p0), tuple(p1), tuple(p2), tuple(p3)]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        # Check if it overlaps with parent
        if not polygons_overlap(poly, parent_poly):
            return poly
    # Both sides overlap; return last attempt anyway
    return poly


@dataclass
class GeneratorConfig:
    """Configuration for the floorplan generator."""

    num_rooms: int = 5
    min_room_size: float = 2500
    max_room_size: float = 5000
    indent_prob: float = 0.2
    indent_depth_range: tuple[float, float] = (300, 800)
    indent_width_range: tuple[float, float] = (600, 1500)
    global_wall_thickness: float = 120
    wall_thickness_variation_prob: float = 0.1
    wall_thickness_multiplier_range: tuple[float, float] = (1.2, 2.0)
    door_prob: float = 0.7
    window_prob: float = 0.5
    door_width: float = 900
    window_width: float = 1200
    room_prob: float = 0.4
    corridor_prob: float = 0.25
    bathroom_prob: float = 0.2
    utility_prob: float = 0.15
    max_extent: float = 30000
    seed: int = 42
    grid_size: float = 10.0
    max_attach_attempts: int = 50
    layout_type: str | None = None


class FloorplanGenerator:
    """Generates floorplans using a growing-structure algorithm."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def generate(self) -> Floorplan:
        """Main generation method."""
        cfg = self.config

        if cfg.layout_type is not None:
            return self._generate_with_layout()

        # Phase 1: Create room polygons using growing-structure algorithm
        polygons: list[Polygon] = []
        room_types: list[SpaceType] = []

        # Create the first room at origin
        first_type = self._pick_room_type(parent_type=None)
        w, h = self._room_dimensions(first_type)
        first_poly = self._create_room_polygon(0, 0, w, h)
        first_poly = self._snap_polygon(first_poly)
        polygons.append(first_poly)
        room_types.append(first_type)

        # Iteratively attach rooms
        for _ in range(1, cfg.num_rooms):
            placed = self._try_place_room(polygons, room_types)
            if not placed:
                break  # Could not place room; skip gracefully

        num_placed = len(polygons)

        # Phase 2: Build Wall objects with external/internal classification
        all_spaces: list[Space] = []
        for idx in range(num_placed):
            space_id = f"space_{idx + 1:03d}"
            poly = polygons[idx]
            stype = room_types[idx]

            # Get polygon coordinates (closed ring -> drop last duplicate)
            coords = list(poly.exterior.coords)
            polygon_coords = [[c[0], c[1]] for c in coords[:-1]]

            # Classify walls
            wall_classifications = classify_external_walls(idx, polygons)
            walls: list[Wall] = []
            for (seg_start, seg_end), is_external in wall_classifications:
                thickness = cfg.global_wall_thickness
                if self.rng.random() < cfg.wall_thickness_variation_prob:
                    mult = self.rng.uniform(*cfg.wall_thickness_multiplier_range)
                    thickness = cfg.global_wall_thickness * mult

                walls.append(
                    Wall(
                        p1=[seg_start[0], seg_start[1]],
                        p2=[seg_end[0], seg_end[1]],
                        thickness=thickness,
                        is_external=is_external,
                        openings=[],
                    )
                )

            space = Space(
                id=space_id,
                type=stype,
                polygon=polygon_coords,
                walls=walls,
            )
            all_spaces.append(space)

        # Phase 3: Place doors on internal walls, windows on external walls
        self._place_openings(all_spaces, polygons)

        # Phase 4: Ensure connectivity via BFS + adding doors between
        # disconnected components
        self._ensure_connectivity(all_spaces, polygons)

        meta = FloorplanMeta(
            seed=cfg.seed,
            global_wall_thickness=cfg.global_wall_thickness,
        )
        return Floorplan(meta=meta, spaces=all_spaces)

    # ------------------------------------------------------------------ #
    # Layout-aware generation
    # ------------------------------------------------------------------ #

    def _generate_with_layout(self) -> Floorplan:
        """Generate a floorplan using a layout template with adjacency rules."""
        cfg = self.config
        room_list = get_room_list(cfg.layout_type, self.rng)  # type: ignore[arg-type]

        # Layout mode uses more attempts per room to ensure successful placement
        layout_max_attempts = max(cfg.max_attach_attempts, 200)

        polygons: list[Polygon] = []
        room_types: list[SpaceType] = []
        room_type_strs: list[str] = []

        # First room (hallway) is placed at origin as a simple rectangle
        # (no indentation to maximise attachment surface for subsequent rooms)
        first_type_str = room_list[0]
        first_stype = SpaceType(first_type_str)
        w, h = self._room_dimensions(first_stype)
        first_poly = create_rectangle(0, 0, w, h)
        first_poly = self._snap_polygon(first_poly)
        polygons.append(first_poly)
        room_types.append(first_stype)
        room_type_strs.append(first_type_str)

        # Attach remaining rooms following adjacency rules
        for room_str in room_list[1:]:
            placed = self._try_place_room_with_layout(
                room_str, polygons, room_types, room_type_strs,
                layout_max_attempts,
            )
            if not placed:
                # Fallback: try attaching to any existing room
                placed = self._try_place_room_fallback(
                    room_str, polygons, room_types, room_type_strs,
                    layout_max_attempts,
                )
            # If still not placed, skip this room

        num_placed = len(polygons)

        # Build spaces with wall classification
        all_spaces: list[Space] = []
        for idx in range(num_placed):
            space_id = f"space_{idx + 1:03d}"
            poly = polygons[idx]
            stype = room_types[idx]

            coords = list(poly.exterior.coords)
            polygon_coords = [[c[0], c[1]] for c in coords[:-1]]

            wall_classifications = classify_external_walls(idx, polygons)
            walls: list[Wall] = []
            for (seg_start, seg_end), is_external in wall_classifications:
                thickness = cfg.global_wall_thickness
                if self.rng.random() < cfg.wall_thickness_variation_prob:
                    mult = self.rng.uniform(*cfg.wall_thickness_multiplier_range)
                    thickness = cfg.global_wall_thickness * mult

                walls.append(
                    Wall(
                        p1=[seg_start[0], seg_start[1]],
                        p2=[seg_end[0], seg_end[1]],
                        thickness=thickness,
                        is_external=is_external,
                        openings=[],
                    )
                )

            space = Space(
                id=space_id,
                type=stype,
                polygon=polygon_coords,
                walls=walls,
            )
            all_spaces.append(space)

        # Place openings using room-type-specific rules
        self._place_openings(all_spaces, polygons)

        # Ensure connectivity
        self._ensure_connectivity(all_spaces, polygons)

        meta = FloorplanMeta(
            seed=cfg.seed,
            global_wall_thickness=cfg.global_wall_thickness,
        )
        return Floorplan(meta=meta, spaces=all_spaces)

    def _try_place_room_with_layout(
        self,
        room_str: str,
        polygons: list[Polygon],
        room_types: list[SpaceType],
        room_type_strs: list[str],
        max_attempts: int,
    ) -> bool:
        """Try to attach a room to a valid parent per ALLOWED_PARENTS rules."""
        allowed = get_allowed_parents(room_str)
        if not allowed:
            # No parent constraints; treat as fallback
            return False

        # Find candidate parent indices (rooms whose type is in allowed list
        # and passes NEVER_CONNECT validation)
        candidate_indices = [
            i for i, rt in enumerate(room_type_strs)
            if rt in allowed and can_connect(room_str, rt)
        ]
        if not candidate_indices:
            return False

        new_stype = SpaceType(room_str)
        must_external = room_str in MUST_HAVE_EXTERNAL_WALL

        for _ in range(max_attempts):
            parent_idx = int(self.rng.choice(candidate_indices))
            result = self._attach_to_parent_layout(
                parent_idx, new_stype, must_external, polygons, room_types
            )
            if result is not None:
                polygons.append(result)
                room_types.append(new_stype)
                room_type_strs.append(room_str)
                return True

        return False

    def _try_place_room_fallback(
        self,
        room_str: str,
        polygons: list[Polygon],
        room_types: list[SpaceType],
        room_type_strs: list[str],
        max_attempts: int,
    ) -> bool:
        """Fallback: try attaching to any room, ignoring ALLOWED_PARENTS."""
        new_stype = SpaceType(room_str)
        must_external = room_str in MUST_HAVE_EXTERNAL_WALL

        for _ in range(max_attempts):
            parent_idx = int(self.rng.integers(0, len(polygons)))
            # Still respect NEVER_CONNECT
            if not can_connect(room_str, room_type_strs[parent_idx]):
                continue
            result = self._attach_to_parent_layout(
                parent_idx, new_stype, must_external, polygons, room_types
            )
            if result is not None:
                polygons.append(result)
                room_types.append(new_stype)
                room_type_strs.append(room_str)
                return True

        return False

    def _attach_to_parent_layout(
        self,
        parent_idx: int,
        new_stype: SpaceType,
        must_external: bool,
        polygons: list[Polygon],
        room_types: list[SpaceType],
    ) -> Polygon | None:
        """Try to attach a new room to a parent's external wall (layout mode).

        Uses _attach_room_outward to ensure the room extends away from the
        parent polygon rather than into it. Tries multiple dimension variants
        (original, swapped, minimum) to improve placement success rate.

        Returns the new polygon if successful, None otherwise.
        """
        cfg = self.config

        ext_walls = classify_external_walls(parent_idx, polygons)
        external_segs = [(s, e) for (s, e), is_ext in ext_walls if is_ext]
        if not external_segs:
            return None

        seg_idx = int(self.rng.integers(0, len(external_segs)))
        seg_start, seg_end = external_segs[seg_idx]

        wall_start = np.array(seg_start)
        wall_end = np.array(seg_end)
        parent_poly = polygons[parent_idx]

        new_w, new_h = self._room_dimensions(new_stype)

        # Try the room as-is, then with swapped dims, then with minimum dims
        candidates = [(new_w, new_h)]
        if abs(new_w - new_h) > 100:
            candidates.append((new_h, new_w))
        type_key = new_stype.value
        if type_key in ROOM_SPECS:
            spec = ROOM_SPECS[type_key]
            candidates.append((spec.width_min, spec.height_min))
            candidates.append((spec.height_min, spec.width_min))

        for cw, ch in candidates:
            new_poly = _attach_room_outward(
                wall_start, wall_end, cw, ch, parent_poly, self.rng
            )
            new_poly = self._snap_polygon(new_poly)

            if not new_poly.is_valid or new_poly.is_empty:
                continue

            # Check overlap with ALL existing polygons
            has_overlap = False
            for existing in polygons:
                if polygons_overlap(new_poly, existing):
                    has_overlap = True
                    break
            if has_overlap:
                continue

            # Check max extent
            all_polys_candidate = polygons + [new_poly]
            union = unary_union(all_polys_candidate)
            minx, miny, maxx, maxy = union.bounds
            if (maxx - minx) > cfg.max_extent or (maxy - miny) > cfg.max_extent:
                continue

            # If room must have external wall, verify it does
            if must_external:
                test_idx = len(polygons)
                test_polys = polygons + [new_poly]
                new_ext_walls = classify_external_walls(test_idx, test_polys)
                has_external = any(is_ext for _, is_ext in new_ext_walls)
                if not has_external:
                    continue

            return new_poly

        return None

    # ------------------------------------------------------------------ #
    # Room type selection
    # ------------------------------------------------------------------ #

    def _pick_room_type(self, parent_type: Optional[SpaceType]) -> SpaceType:
        """Pick a room type using configured probabilities, with constraints."""
        cfg = self.config
        types = [SpaceType.LIVING_ROOM, SpaceType.CORRIDOR, SpaceType.BATHROOM, SpaceType.UTILITY]
        probs = [cfg.room_prob, cfg.corridor_prob, cfg.bathroom_prob, cfg.utility_prob]

        # Constraint: bathroom only attaches to corridor or room, not another bathroom
        if parent_type == SpaceType.BATHROOM:
            # Remove bathroom from options
            idx_bath = types.index(SpaceType.BATHROOM)
            types.pop(idx_bath)
            probs.pop(idx_bath)

        total = sum(probs)
        probs = [p / total for p in probs]
        choice = self.rng.choice(len(types), p=probs)
        return types[choice]

    # ------------------------------------------------------------------ #
    # Room dimensions by type
    # ------------------------------------------------------------------ #

    def _room_dimensions(self, stype: SpaceType) -> tuple[float, float]:
        """Return (width, height) for a room based on type.

        When ROOM_SPECS contains the type, use spec-based ranges.
        Otherwise fall back to config-based ranges for backward compat.
        """
        type_key = stype.value
        if type_key in ROOM_SPECS:
            spec = ROOM_SPECS[type_key]
            w = self.rng.uniform(spec.width_min, spec.width_max)
            h = self.rng.uniform(spec.height_min, spec.height_max)
            return w, h

        # Fallback for any unknown type (should not happen with current SpaceType)
        cfg = self.config
        w = self.rng.uniform(cfg.min_room_size, cfg.max_room_size)
        h = self.rng.uniform(cfg.min_room_size, cfg.max_room_size)
        return w, h

    # ------------------------------------------------------------------ #
    # Room polygon creation
    # ------------------------------------------------------------------ #

    def _create_room_polygon(
        self, x: float, y: float, width: float, height: float
    ) -> Polygon:
        """Create a room polygon, possibly with an indentation."""
        cfg = self.config
        if (
            self.rng.random() < cfg.indent_prob
            and width > cfg.indent_width_range[1] * 1.5
            and height > cfg.indent_depth_range[1] * 1.5
        ):
            indent_depth = self.rng.uniform(*cfg.indent_depth_range)
            indent_width = self.rng.uniform(*cfg.indent_width_range)
            return create_indented_room(
                width, height, indent_depth, indent_width,
                self.rng, origin_x=x, origin_y=y,
            )
        return create_rectangle(x, y, width, height)

    def _snap_polygon(self, poly: Polygon) -> Polygon:
        """Snap polygon coordinates to grid."""
        coords = list(poly.exterior.coords)
        snapped = snap_to_grid(coords, self.config.grid_size)
        result = Polygon(snapped)
        if not result.is_valid:
            result = result.buffer(0)
        return result

    # ------------------------------------------------------------------ #
    # Room placement (growing structure)
    # ------------------------------------------------------------------ #

    def _try_place_room(
        self, polygons: list[Polygon], room_types: list[SpaceType]
    ) -> bool:
        """Try to attach a new room to an existing one. Returns True if placed."""
        cfg = self.config

        for _ in range(cfg.max_attach_attempts):
            # Pick parent room weighted by external wall count; corridors get 3x weight
            weights = []
            for idx, poly in enumerate(polygons):
                ext_walls = classify_external_walls(idx, polygons)
                ext_count = sum(1 for _, is_ext in ext_walls if is_ext)
                w = ext_count
                if room_types[idx] == SpaceType.CORRIDOR:
                    w *= 3
                weights.append(max(w, 1))

            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            parent_idx = int(self.rng.choice(len(polygons), p=probs))
            parent_poly = polygons[parent_idx]
            parent_type = room_types[parent_idx]

            # Pick an external wall of parent
            ext_walls = classify_external_walls(parent_idx, polygons)
            external_segs = [(s, e) for (s, e), is_ext in ext_walls if is_ext]
            if not external_segs:
                continue

            seg_idx = int(self.rng.integers(0, len(external_segs)))
            seg_start, seg_end = external_segs[seg_idx]

            # Pick room type and dimensions
            new_type = self._pick_room_type(parent_type=parent_type)
            new_w, new_h = self._room_dimensions(new_type)

            # Attach room to wall
            wall_start = np.array(seg_start)
            wall_end = np.array(seg_end)
            new_poly = attach_room_to_wall(
                wall_start, wall_end, new_w, new_h, self.rng
            )
            new_poly = self._snap_polygon(new_poly)

            if not new_poly.is_valid or new_poly.is_empty:
                continue

            # Check overlap with all existing rooms
            overlap = False
            for existing in polygons:
                if polygons_overlap(new_poly, existing):
                    overlap = True
                    break

            if overlap:
                continue

            # Check max extent
            all_polys_candidate = polygons + [new_poly]
            union = unary_union(all_polys_candidate)
            minx, miny, maxx, maxy = union.bounds
            if (maxx - minx) > cfg.max_extent or (maxy - miny) > cfg.max_extent:
                continue

            polygons.append(new_poly)
            room_types.append(new_type)
            return True

        return False

    # ------------------------------------------------------------------ #
    # Openings (doors and windows)
    # ------------------------------------------------------------------ #

    def _place_openings(
        self, spaces: list[Space], polygons: list[Polygon]
    ) -> None:
        """Place doors on internal walls, windows on external walls.

        When a room type is found in ROOM_SPECS, uses spec-based door widths
        and window rules (must_have_window / may_have_window). Otherwise falls
        back to config-based probabilities.
        """
        cfg = self.config

        for space in spaces:
            type_key = space.type.value
            spec = ROOM_SPECS.get(type_key)

            for wall in space.walls:
                wall_length = np.sqrt(
                    (wall.p2[0] - wall.p1[0]) ** 2
                    + (wall.p2[1] - wall.p1[1]) ** 2
                )

                if wall.is_external:
                    # Window placement using room-type-specific rules
                    place_window = False
                    if spec is not None:
                        if spec.must_have_window:
                            place_window = True
                        elif spec.may_have_window:
                            place_window = self.rng.random() < 0.5
                        # else: no windows (e.g., hallway, corridor, storage)
                    else:
                        # Fallback to config probability
                        place_window = self.rng.random() < cfg.window_prob

                    # For must_have_window rooms, relax length check to exact fit
                    if spec is not None and spec.must_have_window:
                        min_wall_for_window = cfg.window_width
                    else:
                        min_wall_for_window = cfg.window_width * 1.2

                    if place_window and wall_length >= min_wall_for_window:
                        offset = self.rng.uniform(0.1, 0.9)
                        # Ensure window fits within wall
                        half_w = cfg.window_width / (2 * wall_length)
                        offset = max(half_w, min(1 - half_w, offset))
                        wall.openings.append(
                            Opening(
                                type=OpeningType.WINDOW,
                                width=cfg.window_width,
                                offset=round(offset, 4),
                            )
                        )
                else:
                    # Door placement on internal walls
                    # Use spec-based door width if available
                    if spec is not None:
                        door_w = self.rng.uniform(
                            spec.door_width_min, spec.door_width_max
                        )
                    else:
                        door_w = cfg.door_width

                    if (
                        self.rng.random() < cfg.door_prob
                        and wall_length >= door_w * 1.2
                    ):
                        offset = self.rng.uniform(0.15, 0.85)
                        half_d = door_w / (2 * wall_length)
                        offset = max(half_d, min(1 - half_d, offset))
                        swing = self.rng.choice(["left", "right"])
                        wall.openings.append(
                            Opening(
                                type=OpeningType.DOOR,
                                width=door_w,
                                offset=round(offset, 4),
                                swing=swing,
                            )
                        )

    # ------------------------------------------------------------------ #
    # Connectivity enforcement
    # ------------------------------------------------------------------ #

    def _ensure_connectivity(
        self, spaces: list[Space], polygons: list[Polygon]
    ) -> None:
        """Ensure all rooms are connected via doors.

        Build adjacency graph from existing doors. If disconnected components
        exist, add doors on shared walls between components.
        """
        cfg = self.config

        while True:
            # Build adjacency graph from doors on internal walls
            adj: dict[str, set[str]] = {s.id: set() for s in spaces}
            self._build_adjacency(spaces, adj)

            # BFS from first room
            visited: set[str] = set()
            queue = [spaces[0].id]
            visited.add(spaces[0].id)
            while queue:
                current = queue.pop(0)
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            all_ids = {s.id for s in spaces}
            if visited == all_ids:
                break  # Fully connected

            # Find disconnected component
            disconnected = all_ids - visited

            # Try to add a door between a visited room and a disconnected room
            connected = False
            for s_id in visited:
                s_idx = next(i for i, s in enumerate(spaces) if s.id == s_id)
                for d_id in disconnected:
                    d_idx = next(i for i, s in enumerate(spaces) if s.id == d_id)
                    shared = find_shared_walls(polygons[s_idx], polygons[d_idx])
                    if shared:
                        seg_start, seg_end = shared[0]
                        wall_len = np.sqrt(
                            (seg_end[0] - seg_start[0]) ** 2
                            + (seg_end[1] - seg_start[1]) ** 2
                        )
                        if wall_len < cfg.door_width:
                            continue

                        p1 = [seg_start[0], seg_start[1]]
                        p2 = [seg_end[0], seg_end[1]]

                        offset = 0.5
                        swing = str(self.rng.choice(["left", "right"]))
                        door_opening = Opening(
                            type=OpeningType.DOOR,
                            width=cfg.door_width,
                            offset=offset,
                            swing=swing,
                        )

                        # Add/update wall on the source space
                        self._add_door_to_shared_wall(
                            spaces[s_idx], p1, p2, door_opening, cfg.global_wall_thickness
                        )
                        # Add/update wall on the destination space (reversed p1/p2)
                        door_opening_copy = Opening(
                            type=OpeningType.DOOR,
                            width=cfg.door_width,
                            offset=offset,
                            swing=swing,
                        )
                        self._add_door_to_shared_wall(
                            spaces[d_idx], p2, p1, door_opening_copy, cfg.global_wall_thickness
                        )
                        connected = True
                        break
                if connected:
                    break

            if not connected:
                # Cannot connect further; break to prevent infinite loop
                break

    def _build_adjacency(
        self, spaces: list[Space], adj: dict[str, set[str]]
    ) -> None:
        """Build adjacency graph from internal walls that have doors."""
        for s in spaces:
            for w in s.walls:
                if not w.is_external and any(
                    o.type == OpeningType.DOOR for o in w.openings
                ):
                    for other in spaces:
                        if other.id == s.id:
                            continue
                        for ow in other.walls:
                            if self._walls_match(w, ow):
                                adj[s.id].add(other.id)
                                adj[other.id].add(s.id)

    def _walls_match(self, w1: Wall, w2: Wall) -> bool:
        """Check if two walls represent the same shared wall (possibly reversed)."""
        tol = 1.0
        match_forward = (
            abs(w1.p1[0] - w2.p1[0]) < tol
            and abs(w1.p1[1] - w2.p1[1]) < tol
            and abs(w1.p2[0] - w2.p2[0]) < tol
            and abs(w1.p2[1] - w2.p2[1]) < tol
        )
        match_reverse = (
            abs(w1.p1[0] - w2.p2[0]) < tol
            and abs(w1.p1[1] - w2.p2[1]) < tol
            and abs(w1.p2[0] - w2.p1[0]) < tol
            and abs(w1.p2[1] - w2.p1[1]) < tol
        )
        return match_forward or match_reverse

    def _add_door_to_shared_wall(
        self,
        space: Space,
        p1: list[float],
        p2: list[float],
        door: Opening,
        thickness: float,
    ) -> None:
        """Add a door to a shared wall, creating the wall entry if needed."""
        tol = 1.0
        # Try to find existing wall matching these endpoints
        for wall in space.walls:
            forward = (
                abs(wall.p1[0] - p1[0]) < tol
                and abs(wall.p1[1] - p1[1]) < tol
                and abs(wall.p2[0] - p2[0]) < tol
                and abs(wall.p2[1] - p2[1]) < tol
            )
            reverse = (
                abs(wall.p1[0] - p2[0]) < tol
                and abs(wall.p1[1] - p2[1]) < tol
                and abs(wall.p2[0] - p1[0]) < tol
                and abs(wall.p2[1] - p1[1]) < tol
            )
            if forward or reverse:
                # Wall exists; add door if not already present
                has_door = any(o.type == OpeningType.DOOR for o in wall.openings)
                if not has_door:
                    wall.openings.append(door)
                return

        # Wall not found in space walls — create it
        new_wall = Wall(
            p1=p1,
            p2=p2,
            thickness=thickness,
            is_external=False,
            openings=[door],
        )
        space.walls.append(new_wall)

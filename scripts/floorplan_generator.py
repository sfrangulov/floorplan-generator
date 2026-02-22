# scripts/floorplan_generator.py
"""CLI for batch floorplan JSON generation."""
from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floorplan.generator import FloorplanGenerator, GeneratorConfig


def parse_range(value: str) -> tuple[int, int]:
    if "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    v = int(value)
    return v, v


@click.command()
@click.option("--count", type=int, required=True, help="Number of floorplans")
@click.option("--seed", type=int, default=42, help="Base random seed")
@click.option("--num-rooms", type=str, default="3-8", help="Room count range (e.g. 3-8)")
@click.option("--output-dir", type=click.Path(), required=True, help="Output directory for JSON files")
@click.option("--indent-prob", type=float, default=0.2)
@click.option("--door-prob", type=float, default=0.7)
@click.option("--window-prob", type=float, default=0.5)
@click.option("--wall-thickness", type=float, default=120, help="Global wall thickness in mm")
def cli(count, seed, num_rooms, output_dir, indent_prob, door_prob, window_prob, wall_thickness):
    """Generate synthetic floorplan JSON files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lo, hi = parse_range(num_rooms)
    rng = np.random.default_rng(seed)

    for i in tqdm(range(count), desc="Generating floorplans"):
        n = int(rng.integers(lo, hi + 1))
        item_seed = seed + i

        cfg = GeneratorConfig(
            num_rooms=n, seed=item_seed,
            indent_prob=indent_prob, door_prob=door_prob,
            window_prob=window_prob, global_wall_thickness=wall_thickness,
        )
        fp = FloorplanGenerator(cfg).generate()
        fname = out / f"floorplan_{i:05d}.json"
        fname.write_text(fp.model_dump_json(indent=2))

    click.echo(f"Generated {count} floorplans in {out}")


if __name__ == "__main__":
    cli()

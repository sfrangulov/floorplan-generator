# scripts/generate_synthetic.py
"""CLI for batch rendering floorplan images + annotations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
from PIL import ImageDraw
from shapely.geometry import Polygon
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from floorplan.models import Floorplan
from floorplan.renderer import FloorplanRenderer, RenderConfig
from floorplan.annotations import COCOAnnotator
from floorplan.furniture import FurniturePlacer
from floorplan.styles import generate_style


def render_single(
    json_path: Path, idx: int,
    output_dir: Path, image_size: int,
    with_masks: bool, with_coco: bool,
    furniture_dir: str | None,
) -> dict | None:
    fp = Floorplan.model_validate_json(json_path.read_text())
    seed = fp.meta.seed
    rng = np.random.default_rng(seed)

    style = generate_style(rng)
    renderer = FloorplanRenderer(RenderConfig(image_size=image_size))
    img = renderer.render(fp, style)

    # Furniture
    placer = FurniturePlacer(furniture_dir=furniture_dir)
    draw = ImageDraw.Draw(img)
    transform = renderer._compute_transform(fp)
    for space in fp.spaces:
        poly = Polygon(space.polygon)
        placer.place_furniture(
            draw, poly, space.type.value, rng,
            density=style.furniture_density, transform=transform,
        )

    stem = json_path.stem
    img_path = output_dir / "images" / f"{stem}.png"
    img.save(img_path)

    result = None

    if with_masks:
        ann = COCOAnnotator(image_size=image_size)
        sem_mask = ann.generate_semantic_mask(fp)
        sem_mask.save(output_dir / "masks" / f"{stem}.png")

    if with_coco:
        ann = COCOAnnotator(image_size=image_size)
        result = ann.generate_coco_annotation(
            fp, image_id=idx + 1, file_name=f"{stem}.png"
        )

    return result


@click.command()
@click.option("--input-dir", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), required=True)
@click.option("--image-size", type=int, default=512)
@click.option("--workers", type=int, default=1)
@click.option("--with-masks", is_flag=True)
@click.option("--with-coco", is_flag=True)
@click.option("--furniture-dir", type=click.Path(), default=None)
def cli(input_dir, output_dir, image_size, workers, with_masks, with_coco, furniture_dir):
    """Render floorplan images from JSON descriptions."""
    inp = Path(input_dir)
    out = Path(output_dir)

    (out / "images").mkdir(parents=True, exist_ok=True)
    if with_masks:
        (out / "masks").mkdir(parents=True, exist_ok=True)
    if with_coco:
        (out / "annotations").mkdir(parents=True, exist_ok=True)

    json_files = sorted(inp.glob("*.json"))
    if not json_files:
        click.echo("No JSON files found.")
        return

    coco_results = []

    # Single-process for simplicity (multiprocessing can be added later for workers > 1)
    for i, jf in enumerate(tqdm(json_files, desc="Rendering")):
        result = render_single(jf, i, out, image_size, with_masks, with_coco, furniture_dir)
        if result:
            coco_results.append(result)

    if with_coco and coco_results:
        annotator = COCOAnnotator(image_size=image_size)
        coco_dataset = annotator.build_coco_dataset(coco_results)
        coco_path = out / "annotations" / "coco.json"
        coco_path.write_text(json.dumps(coco_dataset, indent=2))

    click.echo(f"Rendered {len(json_files)} images to {out}")


if __name__ == "__main__":
    cli()

"""Command-line interface: ``clickshot analyze INPUT -o OUTDIR``."""

from __future__ import annotations

import typer

from . import __version__, pipeline
from .config import Config

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def analyze(
    video: str = typer.Argument(..., help="Path to the pre-recorded screen video."),
    out: str = typer.Option("out", "-o", "--out", help="Output directory."),
    config: str = typer.Option(None, "--config", help="TOML file overriding thresholds."),
    fps: float = typer.Option(None, "--fps", help="Frames/sec to process (default 12)."),
    min_confidence: float = typer.Option(
        None, "--min-confidence", help="Drop events below this confidence (0..1)."),
    ignore_region: list[str] = typer.Option(
        None, "--ignore-region",
        help="Rect to ignore as 'x0,y0,x1,y1' fractions (0..1), e.g. a webcam: "
             "0.72,0,1,0.38. Repeatable."),
    debug: bool = typer.Option(False, "--debug", help="Also write a debug overlay video."),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress progress output."),
):
    """Detect click consequences and write PNGs + events.json + index.html."""
    regions = None
    if ignore_region:
        regions = tuple(tuple(float(v) for v in r.split(",")) for r in ignore_region)
        if any(len(r) != 4 for r in regions):
            raise typer.BadParameter("--ignore-region must be 'x0,y0,x1,y1'")
    cfg = Config.load(config, sample_fps=fps, min_confidence=min_confidence,
                      ignore_regions=regions)
    pipeline.analyze(video, out, cfg, debug=debug, verbose=not quiet)


@app.command()
def version():
    """Print the clickshot version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()

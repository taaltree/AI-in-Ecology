"""
run_birdnet.py

Lab 6 helper — run BirdNET on every WAV in a directory and write a
tidy detections CSV.

    python run_birdnet.py \
        --audio data/dawn_chorus/recordings \
        --out   birdnet_detections.csv \
        --lat   44.06 --lon -123.4 --week 24 --min_conf 0.1
"""
import argparse
import csv
from pathlib import Path

# The exact import surface of birdnet-analyzer changes occasionally between
# releases. If this import fails, run `birdnet-analyzer --help` to find the
# correct CLI entry point and adapt accordingly.
try:
    from birdnet_analyzer.analyze import analyze
except ImportError as e:
    raise SystemExit(
        "Could not import birdnet_analyzer. Install with `pip install "
        "birdnet-analyzer` or clone https://github.com/kahst/BirdNET-Analyzer. "
        f"Underlying error: {e}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio",    required=True)
    p.add_argument("--out",      required=True)
    p.add_argument("--lat",      type=float, default=44.06)
    p.add_argument("--lon",      type=float, default=-123.4)
    p.add_argument("--week",     type=int,   default=24)
    p.add_argument("--min_conf", type=float, default=0.1)
    args = p.parse_args()

    rows = []
    for wav in sorted(Path(args.audio).glob("*.wav")):
        detections = analyze(
            wav,
            lat=args.lat, lon=args.lon, week=args.week,
            min_conf=args.min_conf,
            overlap=0.0, sensitivity=1.0,
        )
        for d in detections:
            rows.append({"file": wav.name, **d})

    if not rows:
        raise SystemExit("No detections returned — check audio path and threshold.")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} detections to {args.out}")


if __name__ == "__main__":
    main()

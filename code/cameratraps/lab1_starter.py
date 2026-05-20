"""
lab1_starter.py

Lab 1 starter — Python warm-up on camera-trap metadata.

Run as a script:
    python lab1_starter.py \
        --meta   data_sample/oregon_critters_subset.csv \
        --images data_sample/cameratraps

Or paste cells into Jupyter. Marked "TASK n" where you have work to do.
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ExifTags


def read_exif_timestamp(path: Path):
    """Return EXIF DateTimeOriginal as a datetime, or None if missing/unreadable."""
    try:
        img = Image.open(path)
        exif = img._getexif() or {}
        tagged = {ExifTags.TAGS.get(k): v for k, v in exif.items()}
        raw = tagged.get("DateTimeOriginal")
        if not raw:
            return None
        return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meta",   required=True,
                   help="oregon_critters_subset.csv")
    p.add_argument("--images", required=True,
                   help="folder holding the .JPG files")
    args = p.parse_args()

    # Each image may have multiple rows in the CSV (one per bounding box);
    # for Lab 1 we want one row per image.
    df_full = pd.read_csv(args.meta, parse_dates=["datetime"])
    df = df_full.drop_duplicates("image").reset_index(drop=True)

    # ── TASK 1: warm-up summaries ────────────────────────────────────────────
    print("Rows (images)    :", len(df))
    print("Unique locations :", df["location"].nunique())
    nonempty = df.loc[df["label"] != "empty", "label"]
    print("Species (non-empty):", nonempty.nunique())
    print("Date range       :", df["datetime"].min(), "→", df["datetime"].max())

    # ── TASK 2: bar chart of images per location ─────────────────────────────
    per_loc = df["location"].value_counts().sort_values(ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(10, 4))
    per_loc.plot.bar(ax=ax, color="#2f6a3c")
    ax.set_ylabel("# images")
    ax.set_title(f"Images per camera location (top {len(per_loc)})")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig("fig1_per_site.png", dpi=150)
    plt.close(fig)

    # ── TASK 3: diel activity for top-3 species ──────────────────────────────
    top3 = nonempty.value_counts().head(3).index.tolist()
    df["hour"] = df["datetime"].dt.hour
    fig, ax = plt.subplots(figsize=(8, 4))
    for sp in top3:
        sub = df[df["label"] == sp]
        diel = sub.groupby("hour").size().reindex(range(24), fill_value=0)
        if diel.sum() > 0:
            diel = diel / diel.sum()
        ax.plot(diel.index, diel.values, marker="o", label=sp)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Fraction of detections")
    ax.set_title("Diel activity, top 3 species")
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig2_diel.png", dpi=150)
    plt.close(fig)

    # ── TASK 4: EXIF sanity check on 30 random images ────────────────────────
    sample = df.sample(min(30, len(df)), random_state=42)
    discrepancies = []
    for _, row in sample.iterrows():
        ts_exif = read_exif_timestamp(Path(args.images) / row["image"])
        if ts_exif is None:
            continue
        delta = abs((ts_exif - row["datetime"]).total_seconds())
        if delta > 60:
            discrepancies.append((row["image"], row["datetime"], ts_exif, delta))
    print("\nEXIF vs CSV disagreements (>60s):", len(discrepancies))
    for d in discrepancies[:10]:
        print(" ", d)

    # ── TASK 5: interpretation — write in your PDF, not here ─────────────────


if __name__ == "__main__":
    main()

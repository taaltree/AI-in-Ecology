"""
detect_oregon_critters.py

Run a pretrained YOLO model on every image in a directory and write a
tidy CSV of detections.

    python detect_oregon_critters.py \
        --images /data/oregon_critters/images/site_A \
        --out    site_A_detections.csv \
        --model  yolov8n.pt --conf 0.25
"""
import argparse
import csv
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--out",    required=True)
    p.add_argument("--model",  default="yolov8n.pt")
    p.add_argument("--conf",   type=float, default=0.25)
    args = p.parse_args()

    model = YOLO(args.model)
    image_paths = sorted(Path(args.images).glob("**/*.JPG")) \
                + sorted(Path(args.images).glob("**/*.jpg"))
    print(f"Found {len(image_paths)} images")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "class_id", "class_name", "confidence",
                    "x1", "y1", "x2", "y2"])
        # Iterate per-image so the original filename is preserved.
        # (Passing a list to predict() can clobber per-result path info.)
        for img_path in image_paths:
            result = model.predict(source=str(img_path), conf=args.conf,
                                    verbose=False)[0]
            name = img_path.name
            for box in result.boxes:
                cls_id = int(box.cls.item())
                xy = box.xyxy[0].tolist()
                w.writerow([name, cls_id, model.names[cls_id],
                            round(box.conf.item(), 3),
                            *[round(v, 1) for v in xy]])

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

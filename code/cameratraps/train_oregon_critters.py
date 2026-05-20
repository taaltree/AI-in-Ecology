"""
train_oregon_critters.py

Fine-tune YOLOv8 on the 4-class Oregon Critters subset for Lab 4.
"""
from ultralytics import YOLO


def main():
    model = YOLO("yolov8s.pt")

    model.train(
        data="oregon_critters_yolo/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        name="oregon_critters_v1",
        pretrained=True,
        optimizer="AdamW",
        cos_lr=True,
        augment=True,
        seed=42,
    )

    metrics = model.val()
    print("mAP@0.5     :", round(metrics.box.map50, 3))
    print("mAP@0.5:0.95:", round(metrics.box.map, 3))


if __name__ == "__main__":
    main()

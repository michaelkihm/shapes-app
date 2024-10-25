from ultralytics import YOLO
import os
import wandb

if __name__ == "__main__":
    api_key = os.environ.get("WANDB_API_KEY")

    if not api_key:
        raise ValueError("No WandB API key set. Please set valid key as env variable WANDB_API_KEY")

    wandb.login(key=api_key)
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="/datasets/african-wildlife/african-wildlife.yml",
        epochs=5,
        imgsz=640,
        device="mps",
        project="ultralytics-african-wildlife",
        name="yolo11n"
    )

    print(results)

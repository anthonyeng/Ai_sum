import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.models as models
import torchvision.transforms as transforms

FRAMES_DIR = "data/processed/frames"
OUTPUT_DIR = "data/processed/features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # output 512-dim feature vector
    model.eval()
    model.to(DEVICE)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_feature(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feature = model(tensor)

    return feature.squeeze().cpu().numpy()


def main():
    ensure_dir(OUTPUT_DIR)

    model = load_model()
    transform = get_transform()

    video_folders = sorted([
        f for f in os.listdir(FRAMES_DIR)
        if os.path.isdir(os.path.join(FRAMES_DIR, f))
    ])

    for video_id in video_folders:
        video_path = os.path.join(FRAMES_DIR, video_id)
        image_files = sorted([
            f for f in os.listdir(video_path)
            if f.endswith(".jpg")
        ])

        features = []

        for image_file in tqdm(image_files, desc=f"Extracting {video_id}"):
            image_path = os.path.join(video_path, image_file)
            feature = extract_feature(image_path, model, transform)
            features.append(feature)

        features = np.array(features)
        save_path = os.path.join(OUTPUT_DIR, f"{video_id}.npy")
        np.save(save_path, features)

        print(f"[OK] Saved {video_id}: {features.shape}")


if __name__ == "__main__":
    main()
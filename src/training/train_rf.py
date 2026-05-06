import os
import ast
import random
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURE_DIR = "data/processed/features"
ANNO_FILE = "data/raw/tvsum/ydata-tvsum50-anno.tsv"
MODEL_OUT = "outputs/models/random_forest_video_split.pkl"

TEST_RATIO = 0.2
RANDOM_SEED = 42


def load_annotations():
    df = pd.read_csv(
        ANNO_FILE,
        sep="\t",
        header=None,
        names=["video_id", "category", "scores"]
    )

    annotations = {}

    for _, row in df.iterrows():
        video_id = str(row["video_id"]).strip()
        score_str = str(row["scores"]).strip()

        scores = np.array([float(x) for x in score_str.split(",") if x != ""], dtype=np.float32)
        annotations[video_id] = scores

    return annotations


def get_available_videos(feature_dir, annotations):
    videos = []

    for file_name in os.listdir(feature_dir):
        if not file_name.endswith(".npy"):
            continue

        video_id = file_name.replace(".npy", "")
        if video_id in annotations:
            videos.append(video_id)

    videos.sort()
    return videos


def split_videos(video_ids, test_ratio=0.2, seed=42):
    rng = random.Random(seed)
    shuffled = video_ids[:]
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_ratio))
    test_videos = shuffled[:n_test]
    train_videos = shuffled[n_test:]

    return train_videos, test_videos


def build_dataset(video_ids, feature_dir, annotations):
    X_list = []
    y_list = []

    for video_id in video_ids:
        feature_path = os.path.join(feature_dir, f"{video_id}.npy")
        if not os.path.exists(feature_path):
            print(f"[SKIP] Missing feature file for {video_id}")
            continue

        features = np.load(feature_path)
        scores = annotations[video_id]

        min_len = min(len(features), len(scores))
        features = features[:min_len]
        scores = scores[:min_len]

        X_list.append(features)
        y_list.append(scores)

        print(f"[OK] {video_id} -> features {features.shape}, scores {scores.shape}")

    if not X_list or not y_list:
        raise ValueError("No data found while building dataset.")

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y


def main():
    print("Loading annotations...")
    annotations = load_annotations()

    print("Finding available videos...")
    video_ids = get_available_videos(FEATURE_DIR, annotations)

    if len(video_ids) < 2:
        raise ValueError("Need at least 2 videos for video-level split.")

    print(f"Total videos available: {len(video_ids)}")

    train_videos, test_videos = split_videos(
        video_ids,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )

    print("\nTrain videos:")
    print(train_videos)
    print("\nTest videos:")
    print(test_videos)

    print("\nBuilding training dataset...")
    X_train, y_train = build_dataset(train_videos, FEATURE_DIR, annotations)

    print("\nBuilding testing dataset...")
    X_test, y_test = build_dataset(test_videos, FEATURE_DIR, annotations)

    print("\nDataset summary:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    print("Predicting on held-out videos...")
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\nMetrics on video-level split:")
    print("MSE:", mse)
    print("MAE:", mae)
    print("R2 :", r2)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)

    print(f"\nModel saved to: {MODEL_OUT}")


if __name__ == "__main__":
    main()
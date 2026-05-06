import os
import random
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

FEATURE_DIR = "data/processed/features"
ANNO_FILE = "data/raw/tvsum/ydata-tvsum50-anno.tsv"
MODEL_OUT = "outputs/models/xgboost_video_split_temporal.pkl"

TEST_RATIO = 0.2
RANDOM_SEED = 42

# number of neighboring segments on each side
TEMPORAL_RADIUS = 1   # 1 => [prev, current, next]


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
        scores = np.array(
            [float(x) for x in score_str.split(",") if x != ""],
            dtype=np.float32
        )
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


def build_temporal_features(features, radius=1):
    """
    Input:
        features: (num_segments, feature_dim)

    Output:
        temporal_features: (num_segments, feature_dim * (2 * radius + 1))

    For radius=1:
        [prev, current, next]
    Edge handling:
        first segment reuses itself as prev
        last segment reuses itself as next
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")

    num_segments, feature_dim = features.shape
    window_size = 2 * radius + 1

    temporal_rows = []

    for i in range(num_segments):
        context_parts = []

        for offset in range(-radius, radius + 1):
            j = i + offset

            if j < 0:
                j = 0
            elif j >= num_segments:
                j = num_segments - 1

            context_parts.append(features[j])

        temporal_row = np.concatenate(context_parts, axis=0)
        temporal_rows.append(temporal_row)

    temporal_features = np.vstack(temporal_rows).astype(np.float32)

    expected_dim = feature_dim * window_size
    if temporal_features.shape != (num_segments, expected_dim):
        raise ValueError(
            f"Temporal feature shape mismatch. "
            f"Got {temporal_features.shape}, expected {(num_segments, expected_dim)}"
        )

    return temporal_features


def build_dataset(video_ids, feature_dir, annotations, temporal_radius=1):
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

        if len(features) == 0:
            print(f"[SKIP] Empty aligned data for {video_id}")
            continue

        temporal_features = build_temporal_features(features, radius=temporal_radius)

        X_list.append(temporal_features)
        y_list.append(scores)

        print(
            f"[OK] {video_id} -> "
            f"raw {features.shape}, "
            f"temporal {temporal_features.shape}, "
            f"scores {scores.shape}"
        )

    if not X_list or not y_list:
        raise ValueError("No data found while building dataset.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.hstack(y_list).astype(np.float32)

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
    X_train, y_train = build_dataset(
        train_videos,
        FEATURE_DIR,
        annotations,
        temporal_radius=TEMPORAL_RADIUS
    )

    print("\nBuilding testing dataset...")
    X_test, y_test = build_dataset(
        test_videos,
        FEATURE_DIR,
        annotations,
        temporal_radius=TEMPORAL_RADIUS
    )

    print("\nDataset summary:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    print("\nTraining XGBoost with temporal context...")
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
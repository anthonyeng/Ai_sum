import os
import numpy as np
import pandas as pd

FEATURE_DIR = "data/processed/features"
ANNO_FILE = "data/raw/tvsum/ydata-tvsum50-anno.tsv"


def load_annotations():
    df = pd.read_csv(
        ANNO_FILE,
        sep="\t",
        header=None,
        names=["video_id", "category", "scores"]
    )

    data = {}

    for _, row in df.iterrows():
        video_id = str(row["video_id"]).strip()

        score_str = str(row["scores"]).strip()
        scores = np.array([float(x) for x in score_str.split(",") if x != ""])

        data[video_id] = scores

    return data


def main():
    annotations = load_annotations()

    X = []
    y = []

    for file in os.listdir(FEATURE_DIR):
        if not file.endswith(".npy"):
            continue

        video_id = file.replace(".npy", "")

        if video_id not in annotations:
            print(f"[SKIP] No annotation for {video_id}")
            continue

        features = np.load(os.path.join(FEATURE_DIR, file))
        scores = annotations[video_id]

        min_len = min(len(features), len(scores))
        features = features[:min_len]
        scores = scores[:min_len]

        X.append(features)
        y.append(scores)

        print(f"[OK] {video_id} -> features {features.shape}, scores {scores.shape}")

    X = np.vstack(X)
    y = np.hstack(y)

    print("\nFinal dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    np.save("data/processed/X.npy", X)
    np.save("data/processed/y.npy", y)

    print("\nSaved dataset!")


if __name__ == "__main__":
    main()
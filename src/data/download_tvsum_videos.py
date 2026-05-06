import os
import subprocess
import pandas as pd

INFO_FILE = "data/raw/tvsum/ydata-tvsum50-info.tsv"
OUTPUT_DIR = "data/raw/tvsum/videos"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def download_video(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_template = os.path.join(OUTPUT_DIR, f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "mp4/best",
        "-o", output_template,
        url,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] {video_id}")
    except:
        print(f"[FAIL] {video_id}")


def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.exists(INFO_FILE):
        print("❌ Missing dataset file")
        return

    df = pd.read_csv(INFO_FILE, sep="\t")
    print("Columns:", df.columns)

    video_id_col = None
    for col in df.columns:
        if "video" in col.lower() and "id" in col.lower():
            video_id_col = col
            break

    if video_id_col is None:
        print("❌ Could not find video id column")
        return

    video_ids = df[video_id_col].dropna().unique()

    print(f"Downloading {len(video_ids)} videos...")

    for vid in video_ids:
        download_video(str(vid))


if __name__ == "__main__":
    main()


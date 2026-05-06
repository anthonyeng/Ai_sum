import os
import cv2

VIDEO_DIR = "data/raw/tvsum/videos"
OUTPUT_DIR = "data/processed/frames"
SECONDS_PER_FRAME = 2  # align with TVSum 2-second segments


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"[SKIP] {video_path} (no fps)")
        return

    frame_interval = int(fps * SECONDS_PER_FRAME)
    count = 0
    saved = 0

    ensure_dir(output_folder)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"segment_{saved:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"[OK] {os.path.basename(video_path)} → {saved} segment frames")


def main():
    ensure_dir(OUTPUT_DIR)

    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    print(f"Found {len(videos)} videos")

    for video_file in videos:
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_folder = os.path.join(OUTPUT_DIR, video_name)

        extract_frames(video_path, output_folder)


if __name__ == "__main__":
    main()
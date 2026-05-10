import cv2
import json
import pandas as pd
from src.detect_track import PlayerTracker, BallTracker
from src.shot_classifier import ShotClassifier
from src.analytics import Analytics
from src.visualizer import Visualizer

# ─── CONFIG ───────────────────────────────────────────────
INPUT_VIDEO  = "input/input_sample_video.mp4"
OUTPUT_VIDEO = "output/shot_classification_output.mp4"
OUTPUT_JSON  = "output/shot_classification_results.json"
OUTPUT_CSV   = "output/shot_classification_results.csv"
DEVICE       = 0        # 0 = GPU (RTX 5070)
SHOW_LIVE    = False    # True = show window while processing
# ──────────────────────────────────────────────────────────


def main():
    print("=" * 55)
    print("   Shot Classification System")
    print("=" * 55)

    # Open video
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {INPUT_VIDEO}")
        return

    fps        = cap.get(cv2.CAP_PROP_FPS)
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video     : {width}x{height} @ {fps:.1f}fps")
    print(f"[INFO] Duration  : {total/fps:.1f}s  ({total} frames)")
    print(f"[INFO] Device    : CUDA GPU {DEVICE}")
    print()

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Initialise all modules
    player_tracker = PlayerTracker(device=DEVICE)
    ball_tracker   = BallTracker()
    classifier     = ShotClassifier(fps=fps)
    visualizer     = Visualizer(width, height)

    all_shots  = []
    frame_idx  = 0

    print("[INFO] Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # 1. Detect players
        players = player_tracker.detect(frame)

        # 2. Detect ball
        ball = ball_tracker.detect(frame)

        # 3. Classify shot (returns shot dict or None)
        shot = classifier.update(
            frame_idx  = frame_idx,
            timestamp  = timestamp,
            players    = players,
            ball_pos   = ball,
            ball_traj  = ball_tracker.trajectory,
        )

        if shot:
            all_shots.append(shot)
            print(f"  [SHOT] Frame {frame_idx:5d} | {timestamp:6.2f}s | "
                  f"Player {shot['player_id']} | {shot['shot_type'].upper()}")

        # 4. Draw everything on frame
        annotated = visualizer.draw(
            frame        = frame.copy(),
            players      = players,
            ball         = ball,
            trajectory   = ball_tracker.trajectory,
            shot         = shot,
            ball_tracker = ball_tracker,
        )

        writer.write(annotated)

        if SHOW_LIVE:
            cv2.imshow("Shot Classification System", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Progress every 250 frames
        if frame_idx % 250 == 0:
            pct = frame_idx / total * 100
            print(f"  [PROGRESS] {pct:.1f}%  ({frame_idx}/{total} frames)")

        frame_idx += 1

    cap.release()
    writer.release()
    if SHOW_LIVE:
        cv2.destroyAllWindows()

    # ── Save results ─────────────────────────────────────
    print()
    print("[INFO] Saving results...")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_shots, f, indent=2)

    if all_shots:
        pd.DataFrame(all_shots).to_csv(OUTPUT_CSV, index=False)

    # ── Analytics summary ────────────────────────────────
    analytics = Analytics(all_shots)
    analytics.print_summary()
    analytics.save_chart("output/shot_classification_chart.png")

    print()
    print("[DONE] Files saved:")
    print(f"   Video  → {OUTPUT_VIDEO}")
    print(f"   JSON   → {OUTPUT_JSON}")
    print(f"   CSV    → {OUTPUT_CSV}")
    print(f"   Chart  → output/shot_classification_chart.png")


if __name__ == "__main__":
    main()

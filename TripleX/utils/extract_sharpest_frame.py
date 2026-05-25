import argparse
import os

import cv2
import numpy as np


def detect_text(frame):
    """
    Detects whether a frame contains significant text.
    Uses edge detection and contour analysis to filter frames with text overlays.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_regions = sum(
        cv2.contourArea(c) > 500 for c in contours
    )  # Adjust threshold as needed
    return (
        text_regions > 5
    )  # If many text-like contours exist, assume it's a text frame


def is_mostly_black_or_white(frame, threshold=0.9):
    """
    Detects if a frame is mostly black or white.
    Args:
        frame (numpy.ndarray): The video frame.
        threshold (float): Proportion of pixels that are near black or white to consider frame invalid.
    Returns:
        bool: True if the frame is mostly black or white, False otherwise.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.size
    black_pixels = np.sum(gray < 30)  # Dark threshold
    white_pixels = np.sum(gray > 225)  # Light threshold

    return (black_pixels / total_pixels > threshold) or (
        white_pixels / total_pixels > threshold
    )


def extract_sharpest_frame(video_path):
    """
    Extracts the sharpest frame from a video file based on Laplacian variance,
    excluding frames that contain significant text or are mostly black/white.

    Args:
        video_path (str): Path to the video file.

    Returns:
        str or None: Path to the saved sharpest frame, or None if extraction fails.
    """
    print(f"[INFO] Starting process for video: {video_path}")

    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open video file: {video_path}")
        return None

    max_variance = 0.0
    sharpest_frame = None
    best_frame_number = -1
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Processing {total_frames} frames...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[DEBUG] End of video reached.")
                break

            if detect_text(frame):
                print(f"[DEBUG] Skipping frame {frame_number} due to detected text.")
                frame_number += 1
                continue

            if is_mostly_black_or_white(frame):
                print(
                    f"[DEBUG] Skipping frame {frame_number} due to being mostly black or white."
                )
                frame_number += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            if variance > max_variance:
                max_variance = variance
                sharpest_frame = frame.copy()
                best_frame_number = frame_number

            frame_number += 1
            if frame_number % 100 == 0:
                print(f"[DEBUG] Processed {frame_number}/{total_frames} frames...")
    except Exception as e:
        print(f"[ERROR] An error occurred during processing: {e}")
    finally:
        cap.release()

    if sharpest_frame is not None:
        output_dir = os.path.join("data", "images")
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_filename = f"{base_name}_sharpest_frame_{best_frame_number}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, sharpest_frame)

        print(f"[INFO] Sharpest frame saved: {frame_path} (Variance: {max_variance})")
        return frame_path

    print("[WARNING] No valid frames found in video.")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract the sharpest frame from a video, excluding text frames and mostly black/white frames."
    )
    parser.add_argument("--input", required=True, help="Path to the input video file")
    args = parser.parse_args()

    result = extract_sharpest_frame(args.input)
    if result:
        print(f"[INFO] Sharpest frame saved: {result}")
    else:
        print("[ERROR] Failed to extract sharpest frame.")


if __name__ == "__main__":
    main()

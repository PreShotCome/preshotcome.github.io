import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np

def get_nearest_frame_count(frame_count, allowed_counts):
    for count in sorted(allowed_counts, reverse=True):
        if frame_count >= count:
            return count
    return min(allowed_counts)

def get_target_resolution(width, height, allowed_resolutions):
    max_size = max(width, height)
    for size in sorted(allowed_resolutions):
        if max_size <= size:
            scale_ratio = size / max_size
            new_width = int(round((width * scale_ratio) / 32) * 32)
            new_height = int(round((height * scale_ratio) / 32) * 32)
            return new_width, new_height
    # If larger than max allowed resolution, scale down to max allowed
    max_allowed = max(allowed_resolutions)
    scale_ratio = max_allowed / max_size
    new_width = int(round((width * scale_ratio) / 32) * 32)
    new_height = int(round((height * scale_ratio) / 32) * 32)
    return new_width, new_height

def process_video(video_path, output_path, allowed_frame_counts, allowed_resolutions):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    target_frame_count = get_nearest_frame_count(frame_count, allowed_frame_counts)
    target_width, target_height = get_target_resolution(orig_width, orig_height, allowed_resolutions)

    # Read frames and process
    frames = []
    for i in range(target_frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_width, target_height))
        frames.append(frame)
    cap.release()

    # Write output video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    for frame in frames:
        out.write(frame)
    out.release()

def process_videos(input_dir, output_dir):
    allowed_frame_counts = [17, 49, 61, 129]
    allowed_resolutions = [512, 768, 960, 1280]

    videos_output_dir = os.path.join(output_dir, 'videos')
    os.makedirs(videos_output_dir, exist_ok=True)
    videos_txt = []
    prompts_txt = []

    for video_file in tqdm(os.listdir(input_dir), desc='Processing videos'):
        if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue
        video_path = os.path.join(input_dir, video_file)
        output_video_path = os.path.join(videos_output_dir, video_file)

        process_video(video_path, output_video_path, allowed_frame_counts, allowed_resolutions)

        relative_video_path = os.path.relpath(output_video_path, output_dir)
        videos_txt.append(relative_video_path)
        prompts_txt.append('')  # Placeholder for prompts

    with open(os.path.join(output_dir, 'videos.txt'), 'w') as f:
        for line in videos_txt:
            f.write(line + '\n')

    with open(os.path.join(output_dir, 'prompt.txt'), 'w') as f:
        for line in prompts_txt:
            f.write(line + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_videos_dir', type=str, required=True, help='Path to input video directory')
    parser.add_argument('--output_dataset_dir', type=str, required=True, help='Path to output dataset directory')
    args = parser.parse_args()

    process_videos(args.input_videos_dir, args.output_dataset_dir)

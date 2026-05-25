import json
import os
import subprocess
import sys
from multiprocessing import Pool


def get_frame_rate(video_path):
    """
    Get the frame rate of the video using ffprobe.
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        '-show_entries', 'stream=avg_frame_rate',
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Failed to get frame rate for {video_path}. Error: {result.stderr}")
        return None

    frame_rate_str = result.stdout.strip()
    if frame_rate_str:
        nums = frame_rate_str.split('/')
        if len(nums) == 2 and nums[1] != '0':
            return float(nums[0]) / float(nums[1])
        else:
            return float(nums[0])
    else:
        return None


def get_video_codec(video_path):
    """
    Get the video codec of the input video.
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    codec = result.stdout.strip()
    if codec:
        return codec
    else:
        return 'libx264'  # Default to libx264 if detection fails


def get_audio_codec(video_path):
    """
    Get the audio codec of the input video.
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    codec = result.stdout.strip()
    if codec:
        codec_mapping = {
            'aac': 'aac',
            'mp3': 'libmp3lame',
            'ac3': 'ac3',
            'flac': 'flac',
            'vorbis': 'libvorbis',
            'opus': 'libopus'
        }
        return codec_mapping.get(codec, 'aac')  # Default to 'aac' if unknown
    else:
        return 'aac'  # Default to 'aac' if no audio or detection fails


def trim_video(input_path, num_frames):
    """
    Trim the first num_frames from the video at input_path and overwrite the file.
    """
    frame_rate = get_frame_rate(input_path)
    if frame_rate is None:
        print(f"Could not determine frame rate for {input_path}. Skipping.")
        return

    # Calculate time to trim in seconds
    time_to_trim = num_frames / frame_rate

    # Determine codecs so that we preserve the original formats
    video_codec = get_video_codec(input_path)
    audio_codec = get_audio_codec(input_path)

    # Create a temporary output file
    temp_output_path = input_path + '.tmp.mp4'
    command = [
        'ffmpeg',
        '-y',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', input_path,
        '-ss', f'{time_to_trim}',
        '-c:v', video_codec,
        '-c:a', audio_codec,
        '-movflags', '+faststart',
        temp_output_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        # Replace the original file with the trimmed file
        os.replace(temp_output_path, input_path)
        print(f"Trimmed {input_path} by {num_frames} frames.")
    else:
        print(f"Failed to trim {input_path}. Error: {result.stderr}")
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def load_first_condition_timestamp(json_path):
    """
    Load the 'first_condition_timestamp' from a JSON file if present.
    Returns a float or None if not available.
    """
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Adjust the following lines to match your JSON structure
        return data.get('metadata', {}).get('first_condition_timestamp')
    except Exception as e:
        print(f"Could not read or parse JSON file {json_path}: {e}")
        return None


def process_video(args):
    """
    Helper function for multiprocessing.
    Expects tuple: (input_path, frames_to_trim).
    """
    (input_path, frames_to_trim) = args
    trim_video(input_path, frames_to_trim)


def main():
    """
    Main entry point for the script.
    Usage:
        python script.py [NUM_FRAMES or 'auto']
    """
    # Default number of frames if not provided
    num_frames = 5
    auto_detect = False

    # Check command-line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == 'auto':
            auto_detect = True
            print("Auto-detect mode: will parse JSON for first_condition_timestamp.")
        else:
            try:
                num_frames = int(arg)
                print(f"Trimming {num_frames} frames from the beginning of each video.")
            except ValueError:
                print(f"Invalid argument '{arg}'. Using default value of 5 frames.")

    # Fixed input directory (you can adjust this path)
    input_directory = os.path.join(os.getcwd(), "data", "captioned")

    # Video extensions you want to process
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.ts')

    # Prepare arguments for multiprocessing
    args_list = []

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(video_extensions):
                input_path = os.path.join(root, file)

                # By default, we'll trim num_frames unless auto_detect is on
                frames_to_trim = num_frames

                if auto_detect:
                    # Attempt to find JSON with the same basename
                    base, _ = os.path.splitext(input_path)
                    json_path = base + '.json'  # for "example.mp4", JSON is "example.json"
                    first_condition = load_first_condition_timestamp(json_path)

                    if first_condition is not None:
                        # We'll determine the video’s actual FPS and compute how many frames that timestamp represents
                        fps = get_frame_rate(input_path)
                        if fps is not None:
                            frames_to_trim = int(round(first_condition * fps))
                            print(
                                f"Auto-detect: {file} -> first_condition_timestamp={first_condition}s ({frames_to_trim} frames)")

                args_list.append((input_path, frames_to_trim))

    # Use multiprocessing to trim in parallel if there are any videos
    if args_list:
        with Pool() as pool:
            pool.map(process_video, args_list)
    else:
        print("No video files found to process.")


if __name__ == "__main__":
    main()
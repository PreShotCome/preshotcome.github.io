"""
This script processes video files as follows:
1. Upload each video file to a specified GCS bucket.
2. Use Vertex AI’s Gemini models (with a fallback sequence) to generate extended caption outputs.
   The prompt instructs the model to return a valid JSON object with keys:
   • "caption": A single, concise, natural language sentence describing the scene.
     It must not start with any meta commentary (e.g., phrases like "Here is a description").
   • "timestamped_captions": An array of objects with "timestamp" (seconds) and "description" for that moment.
   • "metadata": An object containing additional details (e.g., file name, resolution, frame rate).
   • "confirmation": A statement confirming that all depicted individuals are over 21 and have signed consent waivers.
3. The script then writes the JSON output to a .json file and the "caption" text to a .txt file.
4. On successful caption generation, the original video and output files are moved to the specified output directory.

Usage:
  python vertex_video_caption_extended.py \
    --dir /path/to/videos \
    --bucket your-bucket-name \
    --project your-gcp-project-id \
    [--location us-central1] \
    [--prompt "Your custom prompt"] \
    [--output_dir /path/to/output]
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess

import vertexai
from google.cloud import storage
from vertexai.generative_models import (
    GenerativeModel,
    GenerationConfig,
    Part,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold
)

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# Define fallback models – order by priority.
FALLBACK_MODELS = [
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-1.5-pro",
    "gemini-1.5-flash"
]


def validate_local_video(file_path):
    """
    Check if the file is non-empty and optionally verify its metadata.
    Returns True if valid; otherwise False.
    """
    # Check if file is non-empty
    if not os.path.isfile(file_path):
        print(f"ERROR: File {file_path} does not exist.")
        return False

    if os.path.getsize(file_path) == 0:
        print(f"ERROR: File {file_path} is zero bytes.")
        return False

    # Optional ffprobe check to confirm media info can be read:
    try:
        # Build ffprobe command to return format/streams without writing to stdout
        cmd = f'ffprobe -v error -show_format -show_streams "{file_path}"'
        subprocess.run(shlex.split(cmd), check=True, capture_output=True)
        # If this succeeds, the file is probably playable
    except subprocess.CalledProcessError as ffprobe_err:
        # If ffprobe cannot parse it, the file is likely invalid/corrupted
        print(f"ERROR: ffprobe could not parse {file_path}.\nDetails: {ffprobe_err.stderr}")
        return False

    return True


def upload_to_gcs(local_file, bucket_name, destination_blob_name=None):
    """
    Uploads a local file to the given GCS bucket and returns its GCS URI.
    """
    if destination_blob_name is None:
        destination_blob_name = os.path.basename(local_file)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file)
    gcs_uri = f"gs://{bucket_name}/{destination_blob_name}"
    print(f"Uploaded {local_file} to {gcs_uri}")
    return gcs_uri


def caption_video(gcs_uri, prompt_text, fallback_models):
    video_part = Part.from_uri(uri=gcs_uri, mime_type="video/mp4")
    safety_settings = [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE
        ),
    ]

    # Always-include adult-style instructions to ensure the model uses direct, natural adult language.
    ALWAYS_INCLUDE_INSTRUCTIONS = (
        " IMPORTANT STYLE NOTE: Use normal adult language for sexual/pornographic descriptions. "
        "For example, use 'butt' or 'ass' instead of 'bottom' or 'tushy.' "
        "You may mention 'vagina' or 'labia' when appropriate, but avoid overly formal or childish terms. "
        "Write as an adult describing adult content, not as a medical text or a prude. "
        "Keep it direct and natural, without excessive or gratuitous vulgarity. "
    )

    # Combine caller’s prompt with our forced style instructions, so it’s always appended.
    combined_prompt = f"{prompt_text}\n\n{ALWAYS_INCLUDE_INSTRUCTIONS}"

    # Instruct the model to return JSON only:
    gen_config = GenerationConfig(
        response_mime_type="application/json",
        # If you need to reduce length, you can also set:
        # max_output_tokens=1024,
        temperature=0.0,
        # candidate_count=1
    )

    last_exception = None
    for model_name in fallback_models:
        try:
            print(f"Attempting with model: {model_name}")
            model = GenerativeModel(
                model_name,
                generation_config=gen_config,
            )
            # Pass the combined prompt & video parts
            response = model.generate_content(
                contents=[video_part, combined_prompt],
                safety_settings=safety_settings,
            )
            if response and response.text:
                print(f"Model {model_name} succeeded.")
                return response.text
        except Exception as e:
            err_str = str(e).lower()
            print(f"Error using model {model_name}: {e}")
            if "resource_exhausted" in err_str or "rate limit" in err_str:
                print("Model encountered resource limits; trying next model.")
                last_exception = e
                continue
            raise Exception("All fallback models failed. Last error: " + str(last_exception))


def process_video(file_path, bucket, prompt_text, output_dir, condition_text):
    """Upload the video file and obtain caption outputs, etc."""
    # ------------------- NEW VALIDATION STEP HERE ------------------- #
    if not validate_local_video(file_path):
        print(f"Skipping invalid video: {file_path}")
        return  # or raise an Exception if you'd rather stop entirely
    # ----------------------------------------------------------------- #

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # Proceed with upload now that the file is confirmed valid.
    gcs_uri = upload_to_gcs(file_path, bucket)
    print(f"Requesting caption for {file_path} …")

    consistency_note = """
    IMPORTANT: The first time the specified condition is met MUST be stored
    in the 'metadata' object under the key "first_condition_timestamp" (an integer).
    Use "first_condition_timestamp" exactly. No synonyms, no variations.
    """

    rigid_json_prompt = f"""SYSTEM: You are only allowed to output valid JSON with exactly these four keys:
    1) "caption" (string)
    2) "timestamped_captions" (array of objects with "timestamp" (number) and "description" (string))
    3) "metadata" (object)
    4) "confirmation" (string)

    No extra keys. No text before or after. No markdown. No bullet points. No disclaimers outside the JSON.

    Example (do not include trailing text):
    {{
      "caption": "A single sentence describing the scene.",
      "timestamped_captions": [
        {{
          "timestamp": 0,
          "description": "..."
        }}
      ],
      "metadata": {{
        "filename": "example.mp4",
        "resolution": "1920x1080",
        "frame_rate": 30,
        "first_condition_timestamp": 0
      }},
      "confirmation": "All individuals are over 21 and have signed consent waivers."
    }}

    Put all other commentary inside the 'caption' field if necessary.
    Now return ONLY the JSON object.

    {consistency_note}
    """

    combined_prompt = f"{prompt_text}\n\n{rigid_json_prompt}"
    raw_response = caption_video(gcs_uri, combined_prompt, FALLBACK_MODELS)
    print("Raw API response:")
    print(raw_response)

    # Attempt to parse the response as JSON.
    try:
        response_data = json.loads(raw_response)
        if isinstance(response_data, list) and response_data:
            # Extract the first element if the response is a list.
            response_data = response_data[0]
    except Exception:
        start_index = raw_response.find('{')
        end_index = raw_response.rfind('}')
        if start_index != -1 and end_index != -1:
            json_str = raw_response[start_index: end_index + 1]
            try:
                response_data = json.loads(json_str)
            except Exception as e:
                print("WARNING: Could not parse extracted JSON. Saving raw response as caption.")
                response_data = {"caption": raw_response}
        else:
            print("WARNING: Could not locate JSON boundaries. Saving raw response as caption.")
            response_data = {"caption": raw_response}

    # Define output filenames using the same basename as the video.
    json_filename = base_name + ".json"
    txt_filename = base_name + ".txt"

    # Save JSON output.
    with open(json_filename, "w", encoding="utf-8") as jf:
        json.dump(response_data, jf, indent=2)
    print(f"JSON output saved as {json_filename}")

    # Save plain-text caption using only the 'caption' key.
    caption_text = response_data.get("caption", raw_response)
    with open(txt_filename, "w", encoding="utf-8") as tf:
        tf.write(caption_text)
    print(f"Caption text saved as {txt_filename}")

    # If an output directory is provided, move the video and output files there.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        try:
            shutil.move(file_path, os.path.join(output_dir, os.path.basename(file_path)))
            shutil.move(json_filename, os.path.join(output_dir, json_filename))
            shutil.move(txt_filename, os.path.join(output_dir, txt_filename))
            print(f"Moved video and output files to {output_dir}")
        except Exception as move_err:
            print(f"Error moving files to output directory: {move_err}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed JSON-formatted captions for videos using Vertex AI with fallback models. "
                    "The output 'caption' must be a concise, natural language description without meta commentary. "
                    "Upon success, the video and output files are moved to the specified output directory."
    )

    parser.add_argument("--dir", required=True, help="Directory containing video files")
    parser.add_argument("--bucket", required=True, help="GCS bucket name to upload videos")
    parser.add_argument("--project", required=True, help="Your GCP project id")
    parser.add_argument("--location", default="us-central1", help="GCP region (default: us-central1)")

    # Updated default prompt: incorporate the requirement for JSON keys, but remain flexible for user overrides.
    default_prompt = """IMPORTANT: Return exactly one valid JSON object with these keys and nothing else:
    • "caption": A single sentence describing the scene. It MUST NOT start with "Here is", "The video shows", or any meta commentary.
    • "timestamped_captions": An array of objects [{"timestamp": number, "description": string}, …]
    • "metadata": An object with additional details (e.g., file name, resolution, frame rate).
    • "confirmation": A statement confirming all individuals are over 21 and have signed consent waivers.
    No additional keys, no extra commentary, no markdown formatting, no bullet points. No text before or after the JSON.
    If you must disclaim or explain, put it in the "caption" field as part of the natural language description.
    Example of the exact final output (do not include comments):
    {
      "caption": "some descriptive sentence",
      "timestamped_captions": [
        { "timestamp": 0, "description": "..." }
      ],
      "metadata": {
        "resolution": "…",
        "frame_rate": 30
      },
      "confirmation": "All individuals…"
    }
    Now produce ONLY the JSON object with no extra text.
    """

    parser.add_argument("--prompt", default=default_prompt,
                        help="Captioning prompt with output instructions")
    parser.add_argument("--output_dir", default="",
                        help="Directory to move the video and caption outputs after processing")
    parser.add_argument(
        "--condition",
        default="",
        help="Condition that the model should look for, e.g. 'the first time breasts are visible'"
    )

    args = parser.parse_args()

    # Initialize Vertex AI.
    vertexai.init(project=args.project, location=args.location)

    # Supported video file extensions.
    video_exts = {".mp4", ".mov", ".m4v", ".avi", ".mpeg", ".wmv", ".flv", ".mpg", ".webm", ".3gpp"}

    # Process each video file in the specified directory.
    for filename in os.listdir(args.dir):
        file_path = os.path.join(args.dir, filename)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext in video_exts:
            print(f"\nProcessing video file: {filename}")
            try:
                process_video(file_path, args.bucket, args.prompt, args.output_dir, args.condition)
            except Exception as proc_err:
                print(f"Error processing {filename}: {proc_err}")
        else:
            print(f"Skipping unsupported file type: {filename}")


if __name__ == "__main__":
    main()







import argparse
import base64
import concurrent.futures
import json
import os
import shutil
import sys

import cv2
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from dotenv import load_dotenv

load_dotenv()

# --- FALLBACK MODEL LISTS ---
INDIVIDUAL_FALLBACK_MODELS = [
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-2.0-pro-exp-02-05"
]

COMPOSITE_FALLBACK_MODELS = [
    "gemini-2.0-pro-exp-02-05",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash"
]

# Define safety settings using the imported enums
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
}


def rewrite_composite_caption(composite_caption):
    # Build a rewriting prompt that asks the model to reframe the caption.
    # IMPORTANT: The rewritten caption must keep all the scene details but avoid redundant phrases.
    rewriting_prompt = (
        f"""
            You are a language rewriter. Below is a composite caption generated from multiple video frames. Your goal is to merge any repeated sentences or phrases into a single mention while preserving every specific detail. Do not remove or alter information about nudity, positions, or body parts. Keep the same plain, everyday language used in the composite caption. Do not introduce new information or alter the tone.

            Detailed Requirements:
            • If multiple frames mention the same action (e.g., 'pulling down panties'), consolidate them into a single statement without removing the action altogether.  
            • Keep the final output as one cohesive paragraph or a short set of paragraphs that flows naturally as if describing the situation in casual conversation.  
            • Preserve the mention of all important details from the original text, including clothing, nudity, poses, lighting, or background details.  
            • Do not add explanations, disclaimers, or meta phrases like ‘video opens with…’ or ‘here’s your final caption.’  
            
            Final Caption to Rewrite:
            {composite_caption}
            """
    )

    # Call the gemini-1.5-flash model using the rewriting prompt.
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    result = model.generate_content([rewriting_prompt], generation_config=None)

    if result and result.text:
        return result.text
    else:
        raise Exception("Failed to rewrite composite caption.")


def call_gemini(inputs, generation_config, model_list):
    for model_name in model_list:
        try:
            print(f"Calling Gemini model {model_name} …")
            model = genai.GenerativeModel(model_name=model_name)
            result = model.generate_content(
                inputs,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            if result and result.text:
                return result.text
        except Exception as e:
            err_str = str(e).lower()
            if "resource_exhausted" in err_str or "rate limit" in err_str:
                print(f"Rate limit hit for model {model_name}. Trying next model.")
                continue
            else:
                print(f"Error using model {model_name}: {e}")
                continue
    raise Exception("All fallback models failed (likely all rate limited).")


def get_frame_caption(image_bytes, timestamp, model_list, custom_prompt=""):
    # Build the image input
    image_input = {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(image_bytes).decode('utf-8')
    }
    # Default prompt instructions for this frame.
    default_prompt = f"""
    You are a detailed visual description expert. For the provided video frame at timestamp {timestamp} seconds, generate a caption in Markdown format that adheres to the following rules:

    • Include an entry for every visible person or discernible human element.  
      – Describe each person's appearance, attire, pose, any accessories, and any visible nudity.  
      – Use everyday terms for body parts (e.g., 'butt' instead of 'buttocks') and avoid overly technical or scientific language.  
      – If a female subject’s upper chest is visible, refer to it as “breasts.”  
    • Provide a clear 'scene_description' that would let someone recreate the frame exactly (location, background details, ambiance, lighting).  
    • Describe any movement or action evident in the frame.  
    • Always use plain, colloquial language. Use straightforward terms like 'Lifting up her bra to show her breasts' or 'Pulling down her panties'.  
    • Do not add extra commentary or interpretations beyond what is visible.  
    • If you have custom instructions (e.g., mention a specific pose or detail), integrate them seamlessly.  
    
    Return the final output as a structured Markdown block matching the basic schema:
    {{
    "persons": [...],
      "location": ...,
      "scene_description": ...,
      "movement": ...
    }}
    (or a similar format). 
    """
    # If the user provided extra instructions, append them to the default prompt.
    if custom_prompt:
        default_prompt += "\nAdditional instructions: " + custom_prompt.strip() + "\n"
    inputs = [image_input, default_prompt]
    result_text = call_gemini(inputs, generation_config=None, model_list=model_list)
    return (result_text, image_input)


def get_composite_caption(frame_data_list, composite_model_list, custom_prompt=""):
    # Build the composite input list from each frame.
    inputs = []
    for frame_data in frame_data_list:
        ts = frame_data["timestamp"]
        inputs.append(f"Frame at {ts}s:")
        inputs.append(frame_data["image_input"])
        caption_str = frame_data["caption"]
        inputs.append("Caption: " + caption_str)
    composite_prompt = """
    Using the provided data from each frame (including images, timestamps, and detailed captions), generate a single composite caption that captures everything happening throughout the video in plain language. Incorporate all details (poses, attire, nudity, actions), but if the same point is repeated across frames, merge it into one mention rather than repeating it verbatim.

    Instructions:  
    • Do not leave out any unique detail from the individual captions.  
    • Do not introduce new information or meta commentary such as ‘the video starts with…’ or ‘here is your caption.’  
    • Merge repeated descriptions of the same action or pose into one statement if they are truly redundant.  
    • Always use simple, everyday language (e.g., 'pulling down panties,' 'vagina,' 'breasts,' 'butt').  
    • Keep the flow as if an average person is describing the progression of events.  
    • At the end, your composite caption should sound like a natural, single-paragraph narrative describing the entire sequence of events and actions across the video.  
    """
    # Append extra custom instructions if provided.
    if custom_prompt:
        composite_prompt += "\nAdditional instructions: " + custom_prompt.strip() + "\n"
    inputs.append(composite_prompt)
    composite_caption = call_gemini(inputs, generation_config=None, model_list=composite_model_list)
    return composite_caption


def process_video(file_path, fps, individual_model_list, composite_model_list, custom_prompt="", max_frames=None,
                  output_dir=None):
    base_name = os.path.splitext(file_path)[0]
    output_json_filename = base_name + ".json"
    output_txt_filename = base_name + ".txt"

    if os.path.exists(output_json_filename) and os.path.exists(output_txt_filename):
        try:
            with open(output_json_filename, "r") as f:
                data = json.load(f)
            with open(output_txt_filename, "r") as f:
                composite_text = f.read().strip()
            composite_caption_text = data.get("composite_caption", "").strip() if isinstance(data, dict) else ""
            if composite_caption_text and composite_text:
                print(f"Skipping video file {file_path} because composite captions already exist.")
                return
        except Exception as e:
            print(f"Error reading existing caption files for {file_path}: {e}")

    print(f"Processing video file: {file_path}")
    cap = cv2.VideoCapture(file_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 25  # fallback if not provided

    frame_interval = max(1, round(video_fps / fps))
    count = 0
    # We'll accumulate (timestamp, future) pairs in this list.
    future_results = []
    # Create a ThreadPoolExecutor for parallel frame captioning.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % frame_interval == 0:
                timestamp = int(count / video_fps)
                success, buffer = cv2.imencode('.jpg', frame)
                if not success:
                    print(f"Failed to encode frame at timestamp {timestamp}.")
                    count += 1
                    continue
                image_bytes = buffer.tobytes()
                # Submit the get_frame_caption call to the executor with the custom prompt.
                future = executor.submit(get_frame_caption, image_bytes, timestamp, individual_model_list,
                                         custom_prompt)
                future_results.append((timestamp, future))
                # Optionally, limit the total number of frames processed:
                if max_frames is not None and len(future_results) >= max_frames:
                    print(f"Reached maximum number of frames ({max_frames}). Stopping frame sampling.")
                    break
            count += 1
    cap.release()

    # Gather results, ensuring we keep the timestamp order.
    frames_data = []
    for ts, future in future_results:
        try:
            caption, image_input = future.result()
            frames_data.append({
                "timestamp": ts,
                "caption": caption,
                "image_input": image_input
            })
            print(f"Caption for timestamp {ts}s:\n{caption}\n")
        except Exception as err:
            print(f"Failed to caption frame at timestamp {ts}s: {err}")

    # Ensure the order is correct.
    frames_data.sort(key=lambda x: x["timestamp"])

    # Now that all individual frame captioning is complete, get the composite caption.
    try:
        composite = get_composite_caption(frames_data, composite_model_list, custom_prompt)
        print("Composite caption for video:")
        print(composite)
    except Exception as e:
        print(f"Failed to get composite caption: {e}")
        composite = ""

    if composite.strip():
        try:
            final_caption = rewrite_composite_caption(composite)
            print("Final rewritten caption:")
            print(final_caption)
            composite = final_caption
        except Exception as e:
            print(f"Error rewriting composite caption: {e}")

    # Save the individual captions (JSON) including the composite caption and the plain text separately.
    with open(output_json_filename, "w") as f:
        json.dump({"frames": frames_data, "composite_caption": composite}, f, indent=2)
    print(f"Captions saved to {output_json_filename}")

    # Write only the composite caption text to the final txt file.
    with open(output_txt_filename, "w") as f:
        f.write(composite)
    print(f"Composite caption text saved to {output_txt_filename}")

    if output_dir and composite.strip():
        try:
            os.makedirs(output_dir, exist_ok=True)
            shutil.move(file_path, os.path.join(output_dir, os.path.basename(file_path)))
            shutil.move(output_json_filename, os.path.join(output_dir, os.path.basename(output_json_filename)))
            shutil.move(output_txt_filename, os.path.join(output_dir, os.path.basename(output_txt_filename)))
            print(f"Moved video file and captions to {output_dir}")
        except Exception as e:
            print(f"Error moving files to {output_dir}: {e}")


def process_image(file_path, individual_model_list, composite_model_list, custom_prompt="", output_dir=None):
    print(f"Processing image file: {file_path}")
    try:
        with open(file_path, "rb") as f:
            image_bytes = f.read()

        # Get the caption for the image
        caption, image_input = get_frame_caption(image_bytes, 0, individual_model_list, custom_prompt)
        print(f"Initial caption for image {file_path}:\n{caption}\n")

        # Create a frame_data structure as done for videos (even though there is only one frame)
        frame_data = {
            "timestamp": 0,
            "caption": caption,
            "image_input": image_input
        }
        frames_data = [frame_data]

        # Get the composite caption based on a single frame
        composite = get_composite_caption(frames_data, composite_model_list, custom_prompt)
        print("Composite caption for image:")
        print(composite)

        # Optionally rewrite (refine) the composite caption to remove markdown wrappers
        if composite.strip():
            try:
                final_caption = rewrite_composite_caption(composite)
                print("Final rewritten caption:")
                print(final_caption)
                composite = final_caption
            except Exception as e:
                print(f"Error rewriting composite caption: {e}")

        # Save the JSON structure and plain text caption
        base_name = os.path.splitext(file_path)[0]
        output_json_filename = base_name + ".json"
        output_txt_filename = base_name + ".txt"

        json_data = {"frames": frames_data, "composite_caption": composite}
        with open(output_json_filename, "w") as out_f:
            json.dump(json_data, out_f, indent=2)
        print(f"Caption JSON saved to {output_json_filename}")

        with open(output_txt_filename, "w") as out_f:
            out_f.write(composite)
        print(f"Caption text saved to {output_txt_filename}")

        # Optionally, move outputs to the output directory.
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            import shutil
            shutil.move(file_path, os.path.join(output_dir, os.path.basename(file_path)))
            shutil.move(output_json_filename, os.path.join(output_dir, os.path.basename(output_json_filename)))
            shutil.move(output_txt_filename, os.path.join(output_dir, os.path.basename(output_txt_filename)))
            print(f"Moved image file and captions to {output_dir}")

    except Exception as e:
        print(f"Error processing image {file_path}: {e}")


def process_file(file_path, args, video_exts, image_exts):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in video_exts:
        process_video(file_path, args.fps, INDIVIDUAL_FALLBACK_MODELS,
                      COMPOSITE_FALLBACK_MODELS, custom_prompt=args.custom_prompt,
                      max_frames=args.max_frames, output_dir=args.output_dir)
    elif ext in image_exts:
        process_image(file_path, INDIVIDUAL_FALLBACK_MODELS, COMPOSITE_FALLBACK_MODELS,
                      custom_prompt=args.custom_prompt, output_dir=args.output_dir)
    else:
        print(f"Skipping unsupported file type: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Caption all videos/images in a directory using the Gemini API with fallback. "
                    "Uses lower-tier models for individual frames and a top-quality model for the final composite caption. "
                    "Optionally, move completed files to an output directory."
    )
    parser.add_argument("--dir", required=True, help="Directory containing video and/or image files")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample from videos (default 1)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum number of frames to caption for each video (optional)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to move source files and captions once captioning succeeds")
    # NEW: custom prompt argument. This can be a string with extra instructions.
    parser.add_argument("--custom_prompt", type=str, default="",
                        help="Custom instructions to include in captions for both individual frames and composite caption")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable.")
        sys.exit(1)
    genai.configure(api_key=api_key)

    directory = args.dir
    video_exts = {".mp4", ".mov", ".m4v", ".avi", ".mpeg", ".wmv", ".flv", ".mpg", ".webm", ".3gpp"}
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}

    # Gather all file paths to process
    file_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    # Use a thread pool to process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_file, fp, args, video_exts, image_exts) for fp in file_paths]
        # Optionally wait for all jobs to complete and handle exceptions
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing a file: {e}")


if __name__ == "__main__":
    main()

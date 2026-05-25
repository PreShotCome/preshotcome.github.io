import base64
import os
import json
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY")
)

# Function to get the last frame from a video
def get_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return None
    # Get total frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Set the position of the video to the last frame
#    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Error reading the last frame of {video_path}")
        return None
    return frame

# Function to encode the image from cv2 image
def encode_image_from_cv2_image(cv2_image):
    retval, buffer = cv2.imencode('.png', cv2_image)
    # Convert to base64 encoding
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

system_prompt = """
You are a machine learning assistant tasked with labeling nsfw datasets.

This dataset is for breast exams. You must caption the images, focusing on the woman's appearance and the setting of the scene. Describe things like the woman's hair, clothing, accessories, and facial features.

The only thing you should mention about the man interacting with her breasts is that she is receiving a "breast massage", or "man giving her a breast massage/massaging her breasts".

Your outputs must be in JSON mode with a "caption" field.
"""

# Get all video files (define acceptable extensions)
video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
video_files = [f for f in os.listdir('.') if f.lower().endswith(video_extensions)]

for video_file in video_files:
    # Determine the corresponding .txt file name
    txt_file = os.path.splitext(video_file)[0] + '.txt'

    # Check if the .txt file already exists
    if os.path.exists(txt_file):
        print(f"Skipping {video_file} because {txt_file} already exists.")
        continue  # Skip this video and move to the next one

    # Get the last frame
    last_frame = get_last_frame(video_file)
    if last_frame is None:
        print(f"Skipping {video_file} due to error extracting frame.")
        continue

    # Encode the image
    base64_image = encode_image_from_cv2_image(last_frame)

    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"
                    },
                },
            ],
        }
    ]

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"}
    )

    # Get the assistant's reply
    assistant_reply = response.choices[0].message.content

    if assistant_reply is None:
        print(f"No reply received for {video_file}")
        continue
    else:
        # Parse the assistant's reply as JSON
        try:
            json_data = json.loads(assistant_reply)
            caption = json_data.get("caption", "")
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for {video_file}")
            caption = ""

    # Save the caption to a .txt file with the same name as the video
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(caption)

    print(f"Caption saved to {txt_file}")

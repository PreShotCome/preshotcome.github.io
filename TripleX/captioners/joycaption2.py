import logging
import os
import sys

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Constants
MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
MODEL_PATH = "models/llama-joycaption-alpha-two-hf-llava"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

# Ensure model is downloaded
if not os.path.exists(MODEL_PATH):
    logging.info("Model not found. Downloading...")
    os.makedirs("models", exist_ok=True)
    snapshot_download(repo_id=MODEL_NAME, local_dir=MODEL_PATH)
    logging.info("Download complete!")

# Load model and processor
logging.info("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
llava_model.eval()
logging.info("Model loaded successfully!")


def describe_image(image_path):
    """
    Generate a descriptive caption for an image using JoyCaption2 and save it to a .txt file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        None
    """
    if not os.path.exists(image_path):
        logging.error(f"Error: Image file '{image_path}' not found!")
        return

    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")

        # Build the conversation prompt
        convo = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {
                "role": "user",
                "content": "Write a long descriptive caption for this image in a formal tone. Include information about lighting. Include information about camera angle. Do NOT mention any text that is in the image.",
            },
        ]

        # Format conversation for LLaVA
        convo_string = processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(convo_string, str)

        # Process the inputs
        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to(
            "cuda"
        )
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        # Ensure tensors are valid
        if (
            torch.isnan(inputs["pixel_values"]).any()
            or torch.isinf(inputs["pixel_values"]).any()
        ):
            logging.error("Error: Input tensor contains NaN or Inf values.")
            return

        # Generate caption
        with torch.no_grad():
            generate_ids = llava_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                suppress_tokens=None,
                use_cache=True,
                temperature=0.6,
                top_k=None,
                top_p=0.9,
            )[0]

        # Trim the prompt from output
        generate_ids = generate_ids[inputs["input_ids"].shape[1] :]

        # Decode and clean up caption
        caption = processor.tokenizer.decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

        # Save caption to a .txt file in the same directory as the image
        output_file = os.path.splitext(image_path)[0] + ".txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(caption)

        logging.info(f"🖼️ Caption saved to: {output_file}")

    except Exception as e:
        logging.error(f"Error generating caption: {e}")
        logging.warning(f"No caption generated for: {image_path}")


def process_directory(directory_path):
    """
    Process all images in a directory and generate captions for each.

    Args:
        directory_path (str): Path to the directory containing images.

    Returns:
        None
    """
    if not os.path.exists(directory_path):
        logging.error(f"Error: Directory '{directory_path}' not found!")
        return

    image_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTENSIONS
    ]

    if not image_files:
        logging.warning(f"No valid images found in '{directory_path}'!")
        return

    logging.info(f"Processing {len(image_files)} images in '{directory_path}'...\n")

    for image_file in image_files:
        describe_image(image_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python utils/describe-image.py <image_path_or_folder>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        process_directory(path)
    else:
        describe_image(path)

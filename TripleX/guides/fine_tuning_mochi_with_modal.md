# Fine-Tuning Mochi with Custom Video Data Using TripleX and Modal

In this article, I'll share how I utilized the **TripleX** toolkit from the NSFW API to create a custom dataset from a single compilation video. This dataset was then used to fine-tune a **Mochi** LoRA (Low-Rank Adaptation) model using **Modal**. The goal was to generate personalized video content by adapting the Mochi video generation model to specific themes present in the compilation video.

Check out the [NSFW API subreddit](https://reddit.com/r/NSFW_API)

Join the [NSFW API Discord](https://discord.gg/bW4Bhkfk)

## Overview of the Tools Used

- **TripleX**: A toolkit designed for downloading and processing videos, including scene detection and frame analysis. https://github.com/NSFW-API/TripleX
- **Mochi**: A state-of-the-art video generation model developed by [Genmo](https://genmo.ai), capable of producing high-fidelity videos based on textual prompts. https://github.com/genmoai/mochi
- **Modal**: A platform that provides GPU resources for running intensive machine learning tasks, such as training large models. https://modal.com/

## Samples

https://drive.google.com/drive/folders/1P9sBX_BnwW3g91OlIQB6dG_GO0rHnQql

## Downloading and Setting Up TripleX

Before starting, you need to download and set up TripleX from its GitHub repository.

**Steps:**

1. **Clone the Repository**

   Open your terminal and clone the TripleX repository:

   ```bash
   git clone https://github.com/NSFW-API/TripleX.git
   cd TripleX
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install FFmpeg**

   FFmpeg is required for video processing. Install it using the following commands:

   - **macOS:**

     ```bash
     brew install ffmpeg
     ```

   - **Ubuntu/Debian:**

     ```bash
     sudo apt-get install ffmpeg
     ```

   - **Windows:**

     - Download FFmpeg from the [official website](https://ffmpeg.org/download.html).
     - Add FFmpeg to your system PATH.

5. **Install TensorFlow and Additional Dependencies**

   ```bash
   pip install tensorflow opencv-python numpy
   ```

6. **Set Up Machine Learning Models**

   Run the setup script to download the necessary machine learning models:

   ```bash
   python setup_models.py
   ```

   - This script downloads the required model files from Google Drive and places them in the `models/` directory.

**Note:** Ensure you have an active internet connection during the installation process.

## Steps Followed

### 1. Downloading and Processing the Video with TripleX

With TripleX set up, I began by downloading a compilation video that contained multiple scenes aligned with the themes I wanted the model to learn.

**TripleX Steps:**

- **Downloading the Video:** Used the `download_xhamster.py` script to download the compilation video from a supported website.

  ```bash
  python downloaders/download_xhamster.py <video_url>
  ```

  - Replace `<video_url>` with the actual URL of the video you wish to download.

- **Splitting the Video into Scenes:** Employed the `split_by_scene.py` utility to split the compilation video into individual scenes. This automated process made it manageable to work with multiple short clips extracted from the original video.

  ```bash
  python utils/split_by_scene.py
  ```

- **Trimming Frames (Optional):** If needed, used `trim_frame_beginning.py` to trim unwanted frames from the beginning of each scene.

  ```bash
  python utils/trim_frame_beginning.py 5  # Trims the first 5 frames
  ```

### 2. Captioning the Scenes

To fine-tune the Mochi model effectively, each video clip needed an accompanying textual description. I used a Visual Language Model (VLM) to generate detailed captions for each scene.

**Captioning Process:**

- **Frame Extraction:** Extracted a representative frame from each video clip using the `extract_sharpest_frame.py` script.

  ```bash
  python utils/extract_sharpest_frame.py --input <video_clip_path>
  ```

- **Generating Descriptions:** Provided the extracted frame to the VLM and prompted it to describe the scene's details, including the actors' appearances and actions.

- **Manual Review:** Reviewed and edited the generated captions to ensure accuracy and appropriateness.

**Example Caption 1:**

> *In a brightly lit room, a woman with long, wavy blonde hair tilts her head back with a joyful expression. Her big blue eyes are closed, and she smiles widely, showcasing her white teeth. As a man's hand approaches from the left, he gently holds his hand towards her. A sense of warmth fills the scene as she appears to revel in the experience.*

**Example Caption 2:**

> *A woman stands in a softly lit bathroom, surrounded by neutral tones that create a relaxed atmosphere. She has shoulder-length auburn hair, slightly tousled, framing her face. Dressed in a white, ribbed crop top that hugs her figure, she grips the hem with both hands, pulling it upwards. As she lifts the top, her toned stomach and delicate tattoos become visible, adding to the intimate setting.*

### 3. Preparing the Dataset for Fine-Tuning

With the video clips and captions ready, I organized them into the structure required for fine-tuning Mochi with LoRA:

- Created a folder containing all the video clips.
- Ensured each video clip had a corresponding `.txt` file with its caption.

  ```
  dataset/
    scene_1.mp4
    scene_1.txt
    scene_2.mp4
    scene_2.txt
    ...
  ```

### 4. Setting Up the Mochi Fine-Tuner

I followed the instructions provided in the Mochi LoRA Fine-Tuner documentation to set up the environment:

- **Cloned the Mochi Models Repository:**

  ```bash
  git clone https://github.com/genmoai/models
  cd models 
  ```

- **Installed Dependencies:**

  ```bash
  pip install uv
  uv venv .venv
  source .venv/bin/activate
  uv pip install setuptools
  uv pip install -e . --no-build-isolation
  ```

- **Downloaded the Weights:**

  ```bash
  python3 ./scripts/download_weights.py weights/
  ```

### 5. Fine-Tuning with Modal

To leverage GPU resources for training, I used Modal's platform. This enabled efficient fine-tuning without the need for local high-end hardware.

**Modal Steps:**

- **Set Up Modal:**

  ```bash
  pip install modal
  modal setup
  ```

- **Downloaded the Dataset to Modal Volume:**

  ```bash
  modal run main::download_videos
  ```

- **Downloaded Model Weights to Modal Volume:**

  ```bash
  modal run -d main::download_weights
  ```

- **Processed Videos and Captions:**

  ```bash
  modal run main::preprocess
  ```

- **Fine-Tuned the Model:**

  ```bash
  modal run -d main::finetune
  ```

  - Configured the `lora.yaml` file to specify training parameters, such as the number of steps and learning rate.
  - Monitored the training process to ensure the model was learning effectively from the custom dataset.

### 6. Generating Videos with the Fine-Tuned Model

After fine-tuning, I used the updated model to generate new videos based on textual prompts.

**Generation Steps:**

- **Ran the Inference Script:**

  ```bash
  python3 ./demos/cli.py --model_dir weights/ --lora_path finetunes/my_mochi_lora/model_2000.lora.safetensors --num_frames 37 --cpu_offload --prompt "A joyful moment between two individuals in a bright setting."
  ```

- **Adjusted Parameters:**
  - Tweaked the `--num_frames` to control the length of the generated video.
  - Experimented with different prompts to explore the model's capabilities.

### 7. Reviewing and Sharing Results

- **Sample Outputs:**
  - The generated videos captured themes similar to those in the training dataset.
  - **Samples from the generations are included as `samples.zip` for reference.**

- **Potential Applications:**
  - Personalized content creation.
  - Exploring model capabilities in understanding and generating specific scenarios.

## Conclusion

By combining the data processing power of TripleX, the advanced video generation of Mochi, and the computational resources provided by Modal, I successfully fine-tuned a video generation model using custom data extracted from a single compilation video. This process highlights the potential for creators to tailor AI models to generate content that aligns with specific themes or styles.

**Note:** When working with sensitive content, it's crucial to adhere to all legal and ethical guidelines, including respecting copyright laws and ensuring that all content is appropriate and consensual.

## Acknowledgments

- **Genmo Team** for developing Mochi and providing comprehensive documentation.
- **NSFW API** for creating TripleX, a tool that simplifies video data processing.
- **Modal** for offering accessible computational resources for model training.

---

*For any questions or further discussion, feel free to reach out or leave a comment below!*
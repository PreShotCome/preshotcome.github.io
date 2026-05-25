# Full Guide to Fine-Tuning Hunyuan Video with finetrainers [Credit: L3n4]

This comprehensive guide details everything you need to fine-tune **Hunyuan Video** using the finetrainers library. From dataset preparation to inference.

---
## **Table of Contents**
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Preparing Your Dataset](#preparing-your-dataset)
5. [Training Hunyuan Video](#training-hunyuan-video)

---

## **Overview**

This comprehensive guide details everything you need to fine-tune **Hunyuan Video** using the **finetrainers** library and datasets prepared with **TripleX**. From dataset preparation using TripleX's video processing utilities to training and inference with finetrainers.

**finetrainers** is a library designed to simplify the training and fine-tuning of video models, such as **CogVideoX**, **LTX Video**, and **Hunyuan Video**. The library supports various methodologies, including **LoRA** fine-tuning. Recently, [Hunyuan Video by Tencent](https://huggingface.co/tencent/HunyuanVideo) was added to the lineup, making it easier for developers to adapt the model for custom datasets.

---

## **2. Prerequisites**

Before getting started, ensure that you meet the following requirements:
- Python version >= 3.10.
- Access to GPUs for training; renting an L40 or equivalent on vast.ai or RunPod is highly recommended.
- Installed dependencies and access to datasets (instructions in the next sections).

---

## **3. Setup**

### **Step 1: Clone the finetrainers Repository**

Clone the finetrainers GitHub repository separately and set up a virtual environment:

```bash
git clone https://github.com/a-r-r-o-w/finetrainers
cd finetrainers
python -m venv venv
source venv/bin/activate
```

### **Step 2: Install Required Dependencies**

1. Install Python dependencies from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

2. Install the Hugging Face `diffusers` library from source for the latest features:

   ```bash
   pip install git+https://github.com/huggingface/diffusers
   ```

3. Additional required packages (if not included):

   ```bash
   pip install huggingface_hub accelerate
   ```

**Note**: Ensure that you have set up **TripleX** as per its installation instructions before proceeding.
   
---

## **4. Preparing Your Dataset**

To fine-tune Hunyuan Video, you need a dataset of videos and corresponding captions organized in a specific format. We will use **TripleX** to download videos, split them into scenes, and prepare the dataset.

### **Step 1: Download and Process Videos with TripleX**

1. **Navigate to the TripleX Directory**:

   ```bash
   cd path/to/TripleX
   ```

2. **Download Videos**: Use the `download_xhamster.py` script to download videos.

   ```bash
   python downloaders/download_xhamster.py <video_url>
   ```

   - Replace `<video_url>` with the actual URL of the xHamster video.
   - The videos will be saved in `data/video`.

3. **Split Videos into Scenes**: Use the `split_by_scene.py` script to split the downloaded videos into scenes.

   ```bash
   python utils/split_by_scene.py
   ```

   - Scenes will be saved in `data/clips`.

### **Step 2: Prepare the Dataset for Hunyuan Video**

We will use the `output_clips_to_hunyuan_dataset.py` script to process the scenes into the required format.

**Script Location**: `utils/training/hunyuan/output_clips_to_hunyuan_dataset.py`

#### **Script: output_clips_to_hunyuan_dataset.py**

This script:

- Processes video scenes from `data/clips`.
- Rescales resolutions to multiples of 32.
- Normalizes frame counts to be divisible by 4 or `4k + 1`.
- Outputs processed videos as `.mp4` files.
- Generates `videos.txt` and `prompt.txt`.

**Usage**:

```bash
python utils/training/hunyuan/output_clips_to_hunyuan_dataset.py \
  --input_videos_dir ./data/clips \
  --output_dataset_dir ./data/hunyuan_dataset
```

- `--input_videos_dir`: Path to the directory containing video scenes (`./data/clips`).
- `--output_dataset_dir`: Directory where the prepared dataset will be saved (`./data/hunyuan_dataset`).

#### **After Running the Script**:

- The output directory (`./data/hunyuan_dataset`) will include:
  - `videos/` (processed videos suitable for training).
  - `prompt.txt` (placeholders for captions).
  - `videos.txt` (relative paths to videos).

**Note**: Manually populate `prompt.txt` with captions before proceeding to training.

---

## **5. Training Hunyuan Video**

Now that your dataset is prepared using TripleX, you can proceed to fine-tune Hunyuan Video using finetrainers.

### **Training Script**

Save the script below as `train_hunyuan_video.sh` inside the `finetrainers` directory:

```bash
#!/bin/bash
# Environment Variables
export WANDB_MODE="offline"  # Use offline mode for W&B logging
export NCCL_P2P_DISABLE=1    # For distributed memory efficiency
export TORCH_NCCL_ENABLE_MONITORING=0  # Disable NCCL monitoring for stability

GPU_IDS="0"  # Specify the GPU ID(s) to use

# Paths
DATA_ROOT="/path/to/TripleX/outputs/hunyuan_dataset"  # Path to the dataset prepared with TripleX
CAPTION_COLUMN="prompt.txt"        # Captions file path
VIDEO_COLUMN="videos.txt"          # Videos file path
OUTPUT_DIR="/path/to/output/dir"   # Destination for trained outputs

# Model Arguments
model_cmd="--model_name hunyuan_video \
  --pretrained_model_name_or_path tencent/HunyuanVideo"

# Dataset Arguments
dataset_cmd="--data_root $DATA_ROOT \
  --video_column $VIDEO_COLUMN \
  --caption_column $CAPTION_COLUMN \
  --id_token afkx \
  --video_resolution_buckets 49x512x768 \
  --caption_dropout_p 0.05"

# DataLoader Arguments
dataloader_cmd="--dataloader_num_workers 0"

# Training Arguments
training_cmd="--training_type lora \
  --seed 42 \
  --mixed_precision bf16 \
  --batch_size 1 \
  --train_steps 500 \
  --rank 128 \
  --lora_alpha 128 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --checkpointing_steps 500 \
  --checkpointing_limit 2 \
  --enable_slicing \
  --enable_tiling \
  --precompute_conditions"

# Optimizer Arguments
optimizer_cmd="--optimizer adamw \
  --lr 2e-5 \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 100 \
  --beta1 0.9 \
  --beta2 0.95 \
  --weight_decay 2e-5 \
  --epsilon 1e-8 \
  --max_grad_norm 1.0"

# Miscellaneous Arguments
miscellaneous_cmd="--tracker_name finetrainers-hunyuan-video \
  --output_dir $OUTPUT_DIR \
  --nccl_timeout 1800 \
  --report_to wandb"

# Compiled Accelerate Configuration
cmd="accelerate launch --config_file accelerate_configs/compiled_1.yaml --gpu_ids $GPU_IDS train.py \
  $model_cmd \
  $dataset_cmd \
  $dataloader_cmd \
  $training_cmd \
  $optimizer_cmd \
  $miscellaneous_cmd"

echo "Running command: $cmd"
eval $cmd
echo -ne "-------------------- Finished executing script --------------------\n\n"
```

**Run the Script**:

Navigate to the `finetrainers` directory and run:

```bash
bash train_hunyuan_video.sh
```

**Note**:

- Ensure that the `DATA_ROOT` variable in the script correctly points to the dataset directory prepared by TripleX (`/path/to/TripleX/data/hunyuan_dataset`).
- Adjust `OUTPUT_DIR` to the desired location for saving the trained model outputs.

---

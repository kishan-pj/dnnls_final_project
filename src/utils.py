import os
import re
import random
import textwrap
from typing import Dict, Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from bs4 import BeautifulSoup
import yaml

# load experiment settig from congfig.yaml
def load_config(path:str = "config.yaml")->Dict[str,Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

# Create folders if thery do not already exist
def ensure_dirs(*paths: str) ->None:
    for path in paths:
        if path:
            os.makedirs(path, exist_ok=True)

# Set random seeds for reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Use GPR if availabe otherwise use CPU
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading and saving data
# CHECKPOINT UTILITIES
"""
This cell handles the persistence of our model.
1. It mounts Google Drive to `/content/gdrive` so we can save files permanently.
2. `save_checkpoint_to_drive`: Saves the model state, optimizer state, and current epoch to a file in Drive.
3. `load_checkpoint_from_drive`: Restores the model and optimizer state from a file in Drive, allowing us to resume training.
"""

CHECKPOINT_FOLDER = "checkpoints"

def save_checkpoint_to_drive(model, optimizer, epoch, loss, filename="autoencoder_checkpoint.pth"):
    checkpoint_folder = CHECKPOINT_FOLDER
    os.makedirs(checkpoint_folder, exist_ok=True)

    checkpoint_filename = os.path.basename(filename)
    full_path = os.path.join(checkpoint_folder,checkpoint_filename)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    torch.save(checkpoint, full_path)
    print(f"Checkpoint saved to local folder: {full_path} at epoch {epoch}")


def load_checkpoint(model, optimizer=None, filename="text_autoencoder.pth"):
    checkpoint_folder =  CHECKPOINT_FOLDER
    
    checkpoint_filename = os.path.basename(filename)
    full_path = os.path.join(checkpoint_folder, checkpoint_filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint file not found: {full_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(full_path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)

    print(f"Checkpoint loaded from {full_path} (epoch {epoch})")
    return model, optimizer, epoch, loss


# IMAGE + TEXT PARSING UTILITIES
# @title Functions to show images and process data
"""
This cell provides essential data processing and visualization tools.
1. `parse_gdi_text`: Parses the raw "GDI" (Graph-Text-Interleaved) format from the dataset into a structured dictionary containing objects, actions, and locations.
2. `show_image`: A helper using matplotlib to display pytorch tensors as images, handling optional de-normalization.
"""

# This function just extracts the tags from the text, don't get distracted by it.
# I changed this function a bit to fix some bugs
def parse_gdi_text(text):
    """Parse GDI formatted text into structured data"""
    soup = BeautifulSoup(text, 'html.parser')
    images = []

    for gdi in soup.find_all('gdi'):
        # Debug: print what BeautifulSoup sees

        # Method 1: Try to get image attribute directly
        image_id = None
        if gdi.attrs:
            # Check for attributes like 'image1', 'image2', etc.
            for attr_name, attr_value in gdi.attrs.items():
                if 'image' in attr_name.lower():
                    image_id = attr_name.replace('image', '')
                    break

        # Method 2: Extract from the tag string using regex
        if not image_id:
            tag_str = str(gdi)
            match = re.search(r'<gdi\s+image(\d+)', tag_str)
            if match:
                image_id = match.group(1)

        # Method 3: Fallback - use sequential numbering
        if not image_id:
            image_id = str(len(images) + 1)

        content = gdi.get_text().strip()

        # Extract tagged elements using BeautifulSoup directly
        objects = [obj.get_text().strip() for obj in gdi.find_all('gdo')]
        actions = [act.get_text().strip() for act in gdi.find_all('gda')]
        locations = [loc.get_text().strip() for loc in gdi.find_all('gdl')]

        images.append({
            'image_id': image_id,
            'description': content,
            'objects': objects,
            'actions': actions,
            'locations': locations,
            'raw_text': str(gdi)
        })

    return images

# This is an utility function to show images.
# Why do we need to do all this?
def show_image(ax, image, de_normalize=False, img_mean=None, img_std=None):
    """
    De-normalize the image (if necessary) and show image
    """
    if de_normalize:
        new_mean = -img_mean/img_std
        new_std = 1/img_std

        image = transforms.Normalize(
            mean=new_mean,
            std=new_std,
        )(image)
    ax.imshow(image.permute(1, 2, 0))


# CoT (CHAIN OF THOUGHT) GROUNDING UTILITIES
"""
This cell contains helper functions to parse the "Chain-of-Thought" (CoT) annotations provided in the dataset.
1. `parse_cot_grounding`: Extracts bounding boxes for characters and objects from the CoT markdown tables.
2. `crop_and_resize`: Crops an image region based on a bounding box and resizes it to a fixed size (60x125).
3. `pick_reid_pair`: Selects a pair of bounding boxes for the same entity across different frames to support Re-Identification (ReID) learning.
4. `extract_cot_text_for_frame`: Extracts the text reasoning corresponding to a specific frame.
"""


def _parse_markdown_table(block: str) -> List[Dict[str, str]]:
    lines = [l.rstrip() for l in block.splitlines()]
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:
        return []
    header_line = table_lines[0]
    data_lines = table_lines[2:]
    headers = [h.strip() for h in header_line.strip("|").split("|")]
    rows = []
    for line in data_lines:
        if not line.strip().startswith("|"):
            break
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) != len(headers):
            continue
        rows.append(dict(zip(headers, cols)))
    return rows


def parse_cot_grounding(chain_of_thought: str) -> Dict[int, Dict[str, Any]]:
    """Parse StoryReasoning-style CoT markdown into per-frame bbox annotations."""
    frames: Dict[int, Dict[str, Any]] = {}
    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought or ""))

    for i, m in enumerate(matches):
        img_idx = int(m.group(1)) - 1
        start = m.end()
        end = matches[i + 1].start() if (i + 1 < len(matches)) else len(chain_of_thought)
        section = (chain_of_thought or "")[start:end]

        frames[img_idx] = {"characters": [], "objects": []}

        char_match = re.search(r"###\s*Characters(.*?)(?=\n###|\n##|$)", section, flags=re.DOTALL)
        if char_match:
            for row in _parse_markdown_table(char_match.group(1)):
                cid = row.get("Character ID", "").strip()
                bbox_str = row.get("Bounding Box", "").strip()
                if cid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["characters"].append({"id": cid, "bbox": [x1, y1, x2, y2]})
                    except Exception:
                        pass

        obj_match = re.search(r"###\s*Objects(.*?)(?=\n###|\n##|$)", section, flags=re.DOTALL)
        if obj_match:
            for row in _parse_markdown_table(obj_match.group(1)):
                oid = row.get("Object ID", "").strip()
                bbox_str = row.get("Bounding Box", "").strip()
                if oid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["objects"].append({"id": oid, "bbox": [x1, y1, x2, y2]})
                    except Exception:
                        pass
    return frames


def _clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def crop_and_resize(pil_img, bbox, out_hw=(60, 125)):
    x1, y1, x2, y2 = bbox
    W, H = pil_img.size
    x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, W, H)
    crop = pil_img.crop((x1, y1, x2, y2))
    crop = transforms.Resize(out_hw)(crop)
    crop = transforms.ToTensor()(crop)
    return crop


def pick_reid_pair(frames_cot: Dict[int, Dict[str, Any]]) -> Optional[Tuple[int, int, List[int], List[int], str]]:
    """Pick two detections of the same entity id across frames."""
    id_to_dets = {}
    for f_idx, content in frames_cot.items():
        for det in content.get("characters", []) + content.get("objects", []):
            ent_id = det.get("id")
            bbox = det.get("bbox")
            if ent_id and bbox:
                id_to_dets.setdefault(ent_id, []).append((f_idx, bbox))

    candidates = [ent_id for ent_id, dets in id_to_dets.items() if len(dets) >= 2]
    if not candidates:
        return None

    ent_id = random.choice(candidates)
    dets = id_to_dets[ent_id]
    (f1, b1), (f2, b2) = random.sample(dets, 2)
    return f1, f2, b1, b2, ent_id


def extract_cot_text_for_frame(chain_of_thought: str, frame_idx: int, max_chars: int = 600) -> str:
    """Option 4 helper: extract the 'Image N' section as plain text (best-effort)."""
    if not chain_of_thought:
        return ""
    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought))
    target = None
    for i, m in enumerate(matches):
        if int(m.group(1)) - 1 == frame_idx:
            start = m.end()
            end = matches[i + 1].start() if (i + 1 < len(matches)) else len(chain_of_thought)
            target = chain_of_thought[start:end]
            break
    if target is None:
        return ""
    # Remove markdown tables (keep only non-table lines)
    lines = []
    for line in target.splitlines():
        if line.strip().startswith("|"):
            continue
        if set(line.strip()) <= set("-|:"):
            continue
        lines.append(line)
    text = " ".join([l.strip() for l in lines if l.strip()])
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]



# @title Training utility functions: To initialize and to visualize the progress
"""
1. `init_weights`: Initializes neural network weights using Kaiming Normal initialization.
2. `validation`: A visualization routine. It runs the model on a validation batch and plots:
   - The 4 input frames and their descriptions.
   - The target (Ground Truth) frame and description.
   - The model's predicted frame and generated text description.
"""


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)


# @title Utility functions for NLP tasks
"""
1. `generate`: An inference function for the text decoder.
   - It performs autoregressive generation: predicting one token at a time and feeding it back as input for the next step.
   - Uses greedy decoding by default for stable metrics, with optional sampling for qualitative examples.
"""


def generate(
    model: nn.Module,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    max_len: int,
    sos_token_id: int,
    eos_token_id: int,
    device: torch.device,
    temperature: float = 0.85,
    sample: bool = True,
    repetition_penalty: float = 1.25,
    no_repeat_ngram_size: int = 2,
    top_k: int = 50,
) -> List[int]:
    """
      This function generates a sequence of tokens using the provided decoder.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # 2. SETUP DECODER INPUT
    # Start with the SOS token, shape (1, 1)
    dec_input = torch.tensor([[sos_token_id]], dtype=torch.long, device=device)
    # hidden = torch.zeros(1, 1, hidden_dim, device=device)
    # cell = torch.zeros(1, 1, hidden_dim, device=device)

    generated_tokens = []

    # 3. AUTOREGRESSIVE LOOP
    for _ in range(max_len):
        with torch.no_grad():
            # Run the decoder one step at a time
            # dec_input is (1, 1) here—it's just the last predicted token
            prediction, hidden, cell = model(dec_input, hidden, cell)

        logits = prediction.squeeze(1)  # Shape (1, vocab_size)
        for blocked_id in {0, sos_token_id}:
            if blocked_id is not None and 0 <= blocked_id < logits.size(-1):
                logits[:, blocked_id] = float("-inf")

        if repetition_penalty > 1.0 and generated_tokens:
            for previous_token in set(generated_tokens):
                if 0 <= previous_token < logits.size(-1):
                    if logits[:, previous_token].item() > 0:
                        logits[:, previous_token] /= repetition_penalty
                    else:
                        logits[:, previous_token] *= repetition_penalty

        if no_repeat_ngram_size > 1 and len(generated_tokens) >= no_repeat_ngram_size - 1:
            prefix = tuple(generated_tokens[-(no_repeat_ngram_size - 1):])
            banned_tokens = set()
            for start in range(len(generated_tokens) - no_repeat_ngram_size + 1):
                ngram = generated_tokens[start:start + no_repeat_ngram_size]
                if tuple(ngram[:-1]) == prefix:
                    banned_tokens.add(ngram[-1])

            for banned_token in banned_tokens:
                if 0 <= banned_token < logits.size(-1):
                    logits[:, banned_token] = float("-inf")

        if sample:
            sample_logits = logits
            if top_k > 0 and top_k < sample_logits.size(-1):
                top_values, top_indices = torch.topk(sample_logits, top_k, dim=-1)
                filtered_logits = torch.full_like(sample_logits, float("-inf"))
                filtered_logits.scatter_(1, top_indices, top_values)
                sample_logits = filtered_logits

            probabilities = torch.softmax(sample_logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = next_token.squeeze().item()

        # Check for the End-of-Sequence token
        if token_id == eos_token_id:
            break

        if token_id == 0 or token_id == sos_token_id:
            dec_input = next_token
            continue

        # Append the predicted token
        generated_tokens.append(token_id)

        # The predicted token becomes the input for the next iteration
        dec_input = next_token

    # Return the list of generated token IDs
    return generated_tokens


def validation(
    model: nn.Module,
    data_loader,
    tokenizer,
    device: torch.device,
    output_dir: Optional[str] = None,
    epoch: Optional[int] = None,
    show: bool = False,
) -> None:
    """Display one validation example in the notebook."""
    model.eval()
    with torch.no_grad():
        # Unpack 9 values (dataset was updated to return CoT info)
        frames, descriptions, image_target, text_target, roi1, roi2, roi_valid, roi_frame, ent_id = next(iter(data_loader))

        descriptions = descriptions.to(device)
        frames = frames.to(device)
        image_target = image_target.to(device)
        text_target = text_target.to(device)

        # Unpack 7 values (model now returns extra latents for grounding)
        predicted_image_k, context_image, _, hidden, cell, _, _ = model(frames, descriptions, text_target)

        figure, ax = plt.subplots(2, 6, figsize=(20, 5), gridspec_kw={"height_ratios": [2, 1.5]})

        for i in range(4):
            im = frames[0, i, :, :, :].cpu()
            show_image(ax[0, i], im)
            ax[0, i].set_aspect("auto")
            ax[0, i].axis("off")
            wrapped_text = textwrap.fill(tokenizer.decode(descriptions[0, i, :], skip_special_tokens=True), width=40)

            ax[1, i].text(
                0.5,
                0.99,
                wrapped_text,
                ha="center",
                va="top",
                fontsize=10,
                wrap=True,
            )

            ax[1, i].axis("off")  # Hide axes for the text subplot

        show_image(ax[0, 4], image_target[0].cpu())
        ax[0, 4].set_title("Target")
        ax[0, 4].set_aspect("auto")
        ax[0, 4].axis("off")
        text_target = text_target.squeeze(1)

        wrapped_text = textwrap.fill(tokenizer.decode(text_target[0], skip_special_tokens=True), width=40)
        ax[1, 4].text(
            0.5,
            0.99,
            wrapped_text,
            ha="center",
            va="top",
            fontsize=10,
            wrap=False,
        )
        ax[1, 4].axis("off")
        output = context_image[0, :, :, :].cpu()
        show_image(ax[0, 5], output)
        ax[0, 5].set_title("Predicted")
        ax[0, 5].set_aspect("auto")
        ax[0, 5].axis("off")

        generated_tokens = generate(
            model.text_decoder,
            hidden[:, 0, :].unsqueeze(1),
            cell[:, 0, :].unsqueeze(1),
            max_len=80,
            sos_token_id=tokenizer.cls_token_id,
            eos_token_id=tokenizer.sep_token_id,
            device=device,
            temperature=0.8,
            sample=True,
            repetition_penalty=1.25,
            no_repeat_ngram_size=2,
        )

        wrapped_text = textwrap.fill(tokenizer.decode(generated_tokens), width=40)

        ax[1, 5].text(
            0.5,
            0.99,
            wrapped_text,
            ha="center",
            va="top",
            fontsize=10,
            wrap=False,
        )
        ax[1, 5].axis("off")
        plt.tight_layout()
        # Validation is displayed in the notebook only. Result images are saved explicitly in experiments.ipynb.
        if show:
            plt.show()
        plt.close(figure)

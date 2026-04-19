# src/utils.py

# Imports
import os
import re
import random
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as FT
from bs4 import BeautifulSoup
from datasets import load_dataset
import numpy as np


# DATASET LOADING (replacing removed data.py)
def get_storyreasoning_dataset(split="train"):
    """
    Load StoryReasoning dataset from HuggingFace.
    """
    if split not in ["train", "test"]:
        raise ValueError("split must be 'train' or 'test'")
    return load_dataset("daniel3303/StoryReasoning", split=split)


# CHECKPOINT UTILITIES
def save_checkpoint_to_drive(model, optimizer, epoch, loss, filename="autoencoder_checkpoint.pth"):
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }

    torch.save(checkpoint, full_path)
    print(f"Checkpoint saved at {full_path} (epoch {epoch})")


def load_checkpoint_from_drive(model, optimizer=None, filename="autoencoder_checkpoint.pth"):
    full_path = os.path.join("./checkpoints", filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint file not found: {full_path}")

    checkpoint = torch.load(full_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)

    print(f"Checkpoint loaded from {full_path} (epoch {epoch})")
    return model, optimizer, epoch, loss


# IMAGE + TEXT PARSING UTILITIES
def parse_gdi_text(text: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(text, "html.parser")
    images = []

    for gdi in soup.find_all("gdi"):
        image_id = None

        if gdi.attrs:
            for attr_name in gdi.attrs:
                if "image" in attr_name.lower():
                    image_id = attr_name.replace("image", "")
                    break

        if not image_id:
            tag_str = str(gdi)
            match = re.search(r"<gdi\s+image(\d+)", tag_str)
            image_id = match.group(1) if match else str(len(images) + 1)

        content = gdi.get_text().strip()
        objects = [obj.get_text().strip() for obj in gdi.find_all("gdo")]
        actions = [act.get_text().strip() for act in gdi.find_all("gda")]
        locations = [loc.get_text().strip() for loc in gdi.find_all("gdl")]

        images.append({
            "image_id": image_id,
            "description": content,
            "objects": objects,
            "actions": actions,
            "locations": locations,
            "raw_text": str(gdi)
        })

    return images


def show_image(ax, image, de_normalize=False, img_mean=None, img_std=None):
    if de_normalize and img_mean is not None and img_std is not None:
        new_mean = -img_mean / img_std
        new_std = 1 / img_std
        image = transforms.Normalize(mean=new_mean, std=new_std)(image)

    ax.imshow(image.permute(1, 2, 0))


# CoT (CHAIN OF THOUGHT) GROUNDING UTILITIES
def _parse_markdown_table(block: str) -> List[Dict[str, str]]:
    lines = [l.rstrip() for l in block.splitlines()]
    table_lines = [l for l in lines if l.strip().startswith("|")]

    if len(table_lines) < 3:
        return []

    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    data_lines = table_lines[2:]

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
    frames: Dict[int, Dict[str, Any]] = {}

    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought or ""))

    for i, m in enumerate(matches):
        img_idx = int(m.group(1)) - 1
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chain_of_thought)
        section = (chain_of_thought or "")[start:end]

        frames[img_idx] = {"characters": [], "objects": []}

        char_match = re.search(
            r"###\s*Characters(.*?)(?=\n###|\n##|$)",
            section,
            flags=re.DOTALL
        )

        if char_match:
            for row in _parse_markdown_table(char_match.group(1)):
                cid = row.get("Character ID", "").strip()
                bbox_str = row.get("Bounding Box", "").strip()

                if cid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["characters"].append({
                            "id": cid,
                            "bbox": [x1, y1, x2, y2]
                        })
                    except Exception:
                        pass

        obj_match = re.search(
            r"###\s*Objects(.*?)(?=\n###|\n##|$)",
            section,
            flags=re.DOTALL
        )

        if obj_match:
            for row in _parse_markdown_table(obj_match.group(1)):
                oid = row.get("Object ID", "").strip()
                bbox_str = row.get("Bounding Box", "").strip()

                if oid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["objects"].append({
                            "id": oid,
                            "bbox": [x1, y1, x2, y2]
                        })
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
    if not chain_of_thought:
        return ""

    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought))

    target = None

    for i, m in enumerate(matches):
        if int(m.group(1)) - 1 == frame_idx:
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(chain_of_thought)
            target = chain_of_thought[start:end]
            break

    if target is None:
        return ""
    
    # Remove markdown tables (keep only non-table lines)

    lines = []
    for line in target.splitlines():
        if line.strip().startswith("|") or set(line.strip()) <= set("-|:"):
            continue
        lines.append(line.strip())

    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()

    return text[:max_chars]


# DATASET CLASSES 
class SequencePredictionDataset(Dataset):
    def __init__(self, original_dataset, tokenizer, K: int = 4, max_len: int = 120, image_hw=(60, 125)):
        super().__init__()
        self.dataset = original_dataset
        self.tokenizer = tokenizer
        self.K = K
        self.max_len = max_len
        self.image_hw = image_hw

        self.transform = transforms.Compose([
            transforms.Resize(image_hw),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frames = self.dataset[idx]["images"]
        image_attributes = parse_gdi_text(self.dataset[idx]["story"])
        cot = self.dataset[idx].get("chain_of_thought", "")
        cot_frames = parse_cot_grounding(cot)

        frame_tensors = []
        description_list = []

        for frame_idx in range(self.K):
            image = FT.equalize(frames[frame_idx])
            frame_tensors.append(self.transform(image))

            description = image_attributes[frame_idx]["description"]
            input_ids = self.tokenizer(
                description,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len
            ).input_ids.squeeze(0)

            description_list.append(input_ids)

        image_target = FT.equalize(frames[self.K])
        image_target = self.transform(image_target)

        target_desc = image_attributes[self.K]["description"]
        target_ids = self.tokenizer(
            target_desc,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        ).input_ids

        roi_valid = torch.tensor(0, dtype=torch.long)
        roi1 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi2 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi_frame = torch.tensor(-1, dtype=torch.long)
        ent_id = ""

        pair = pick_reid_pair(cot_frames)

        if pair:
            f1, f2, b1, b2, ent_id = pair

            if 0 <= f1 < self.K and 0 <= f2 < self.K:
                try:
                    roi1 = crop_and_resize(frames[f1], b1, out_hw=self.image_hw)
                    roi2 = crop_and_resize(frames[f2], b2, out_hw=self.image_hw)
                    roi_valid = torch.tensor(1, dtype=torch.long)
                    roi_frame = torch.tensor(int(f1), dtype=torch.long)
                except Exception:
                    pass

        return (
            torch.stack(frame_tensors),
            torch.stack(description_list),
            image_target,
            target_ids,
            roi1,
            roi2,
            roi_valid,
            roi_frame,
            ent_id
        )


class TextTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_attributes = parse_gdi_text(self.dataset[idx]["story"])
        frame_idx = np.random.randint(0, 5)
        return image_attributes[frame_idx]["description"]


class AutoEncoderTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((240, 500)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frames = self.dataset[idx]["images"]
        frame_idx = torch.randint(0, 5, (1,)).item()
        return self.transform(frames[frame_idx])


# TEXT GENERATION UTILITY
def generate(model, hidden, cell, max_len, sos_token_id, eos_token_id, device="cpu"):
    model.eval()

    dec_input = torch.tensor([[sos_token_id]], dtype=torch.long, device=device)
    generated_tokens = []

    for _ in range(max_len):
        with torch.no_grad():
            prediction, hidden, cell = model(dec_input, hidden, cell)

        logits = prediction.squeeze(1)
        temperature = 0.9
        probabilities = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probabilities, num_samples=1)
        token_id = next_token.squeeze().item()

        if token_id == eos_token_id or token_id == 0 or token_id == sos_token_id:
            if token_id == eos_token_id:
                break
            continue

        generated_tokens.append(token_id)
        dec_input = next_token

    return generated_tokens

# VALIDATION FUNCTION
def validation(sequence_predictor, val_dataloader, device="cpu"):
    import torch
    import torch.nn.functional as F

    sequence_predictor.eval()

    criterion = torch.nn.L1Loss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:

            (
                frames,
                descriptions,
                image_target,
                text_target,
                roi1,
                roi2,
                roi_valid,
                roi_frame,
                ent_id
            ) = batch

            frames = frames.to(device)
            descriptions = descriptions.to(device)
            image_target = image_target.to(device)
            text_target = text_target.to(device)

            roi1 = roi1.to(device)
            roi2 = roi2.to(device)

            outputs = sequence_predictor(frames, descriptions, text_target)
            pred_image = outputs[0]

            loss = criterion(pred_image, image_target)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_dataloader)
    print(f"[Validation] Loss: {avg_loss:.4f}")

    return avg_loss
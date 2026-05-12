import argparse
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer

from .model import (
    DecoderLSTM,
    EncoderLSTM,
    Experiment1SequencePredictor,
    Experiment2SequencePredictor,
    Experiment3SequencePredictor,
    SequencePredictionDataset,
    SequencePredictor,
    Seq2SeqLSTM,
    TextTaskDataset,
    VisualAutoencoder,
)
from .utils import (
    ensure_dirs,
    init_weights,
    load_checkpoint,
    load_config,
    set_seed,
    validation,
)

#  Data Preparation

def build_tokenizer():
    """Create the BERT tokenizer."""
    return BertTokenizer.from_pretrained("google-bert/bert-base-uncased", padding=True, truncation=True)


# @title Loading the dataset
def load_storyreasoning(config):
    """Load the StoryReasoning train and test splits."""
    dataset_config = config["dataset"]
    train_dataset = load_dataset(dataset_config["name"], split=dataset_config["train_split"])
    test_dataset = load_dataset(dataset_config["name"], split=dataset_config["test_split"])
    return train_dataset, test_dataset


# @title For the Sequence prediction task
"""
This cell sets up the data pipeline for the main sequence prediction task.
1. Initializes the BERT tokenizer.
2. Creates `SequencePredictionDataset` instances for training and testing.
3. Splits the training data into `train_subset` and `val_subset`.
4. Creates `DataLoader` objects (`train_dataloader`, `val_dataloader`, `test_dataloader`) to batch and shuffle the data.
"""


def build_sequence_dataloaders(config, tokenizer, train_dataset, test_dataset):
    """Build train, validation, and test DataLoaders."""
    dataset_config = config["dataset"]
    image_hw = (dataset_config["image_height"], dataset_config["image_width"])

    sp_train_dataset = SequencePredictionDataset(
        train_dataset,
        tokenizer,
        K=dataset_config["sequence_length"],
        max_len=dataset_config["max_text_length"],
        image_hw=image_hw,
        use_cot_text=config["cot"]["use_cot_text"],
    )
    sp_test_dataset = SequencePredictionDataset(
        test_dataset,
        tokenizer,
        K=dataset_config["sequence_length"],
        max_len=dataset_config["max_text_length"],
        image_hw=image_hw,
        use_cot_text=config["cot"]["use_cot_text"],
    )

    # Let's do things properly, we will also have a validation split
    # Split the training dataset into training and validation sets
    val_size = int(dataset_config["validation_fraction"] * len(sp_train_dataset))
    train_size = len(sp_train_dataset) - val_size
    train_subset, val_subset = random_split(sp_train_dataset, [train_size, val_size])

    # Instantiate the dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=dataset_config["train_batch_size"],
        shuffle=True,
        num_workers=dataset_config["num_workers"],
    )
    # We will use the validation set to visualize the progress.
    val_loader = DataLoader(
        val_subset,
        batch_size=dataset_config["val_batch_size"],
        shuffle=True,
        num_workers=dataset_config["num_workers"],
    )
    test_loader = DataLoader(
        sp_test_dataset,
        batch_size=dataset_config["test_batch_size"],
        shuffle=False,
        num_workers=dataset_config["num_workers"],
    )
    return train_loader, val_loader, test_loader


# Training Routines

# Initialization and setup

# @title Initializing the NLP models
"""
1. Instantiates the `EncoderLSTM`, `DecoderLSTM`, and `Seq2SeqLSTM`.
2. Loads pre-trained weights from `checkpoints/text_autoencoder.pth`.
3. Freezes the text autoencoder parameters to keep them fixed during the main training phase.
"""


def _build_shared_autoencoders(config, tokenizer, device):
    """Build the text and visual autoencoders used by all runs."""
    model_config = config["model"]

    encoder = EncoderLSTM(
        tokenizer.vocab_size,
        model_config["embedding_dim"],
        model_config["latent_dim"],
        model_config["num_layers"],
        model_config["dropout"],
    ).to(device)
    decoder = DecoderLSTM(
        tokenizer.vocab_size,
        model_config["embedding_dim"],
        model_config["latent_dim"],
        model_config["num_layers"],
        model_config["dropout"],
    ).to(device)
    text_autoencoder = Seq2SeqLSTM(encoder, decoder).to(device)

    checkpoint_path = config["paths"]["text_autoencoder_checkpoint"]
    checkpoint_exists = os.path.exists(checkpoint_path)
    text_autoencoder, _, _, _ = load_checkpoint(
        text_autoencoder,
        optimizer=None,
        filename=checkpoint_path,
    )

    total_params = sum(p.numel() for p in text_autoencoder.parameters())
    print(f"Total parameters (Not trainable): {total_params}")
    if config["training"]["freeze_text_autoencoder"] and checkpoint_exists:
        for param in text_autoencoder.parameters():
            param.requires_grad = False
            
        for param in text_autoencoder.decoder.parameters():
            param.requires_grad = True
    elif config["training"]["freeze_text_autoencoder"] and not checkpoint_exists:
        print("Text autoencoder checkpoint is missing, so the text module will stay trainable.")
    visual_autoencoder = VisualAutoencoder(latent_dim=model_config["latent_dim"]).to(device)
    visual_autoencoder.apply(init_weights)

    total_params = sum(p.numel() for p in visual_autoencoder.parameters() if p.requires_grad)
    print(f"Total trainable parameters in visual autoencoder: {total_params}")

    return visual_autoencoder, text_autoencoder


def _print_sequence_predictor_params(sequence_predictor):
    """Print trainable and total parameter counts."""
    total_params = sum(p.numel() for p in sequence_predictor.parameters() if p.requires_grad)
    print(f"Total trainable parameters in the whole model: {total_params}")

    total_params = sum(p.numel() for p in sequence_predictor.parameters())
    print(f"Total parameters: {total_params}")


# BASELINE MODEL BUILDER.
# Builds the simple concatenation + unidirectional GRU model.
def build_model(config, tokenizer, device):
    """Build the baseline sequence predictor."""
    model_config = config["model"]
    visual_autoencoder, text_autoencoder = _build_shared_autoencoders(config, tokenizer, device)

    sequence_predictor = SequencePredictor(
        visual_autoencoder,
        text_autoencoder,
        latent_dim=model_config["latent_dim"],
        gru_hidden_dim=model_config["latent_dim"],
    ).to(device)

    _print_sequence_predictor_params(sequence_predictor)
    return sequence_predictor

# EXPERIMENT 1 MODEL BUILDER.
# Builds Cross-Modal Attention grounding + unidirectional GRU.
def build_experiment1_model(config, tokenizer, device):
    """Build Experiment 1: Cross-Modal Attention + unidirectional GRU."""
    model_config = config["model"]
    visual_autoencoder, text_autoencoder = _build_shared_autoencoders(config, tokenizer, device)

    sequence_predictor = Experiment1SequencePredictor(
        visual_autoencoder,
        text_autoencoder,
        latent_dim=model_config["latent_dim"],
        gru_hidden_dim=model_config["gru_hidden_dim"],
    ).to(device)

    _print_sequence_predictor_params(sequence_predictor)
    return sequence_predictor

# EXPERIMENT 2 MODEL BUILDER.
# Builds simple concatenation + bidirectional GRU.
def build_experiment2_model(config, tokenizer, device):
    """Build Experiment 2: simple concatenation + bidirectional GRU."""
    model_config = config["model"]
    visual_autoencoder, text_autoencoder = _build_shared_autoencoders(config, tokenizer, device)

    sequence_predictor = Experiment2SequencePredictor(
        visual_autoencoder,
        text_autoencoder,
        latent_dim=model_config["latent_dim"],
        gru_hidden_dim=model_config["gru_hidden_dim"],
    ).to(device)

    _print_sequence_predictor_params(sequence_predictor)
    return sequence_predictor

# EXPERIMENT 3 MODEL BUILDER.
# Builds simple concatenation + bidirectional LSTM.
def build_experiment3_model(config, tokenizer, device):
    """Build Experiment 3: simple concatenation + bidirectional LSTM."""
    model_config = config["model"]
    visual_autoencoder, text_autoencoder = _build_shared_autoencoders(config, tokenizer, device)

    sequence_predictor = Experiment3SequencePredictor(
        visual_autoencoder,
        text_autoencoder,
        latent_dim=model_config["latent_dim"],
        gru_hidden_dim=model_config["gru_hidden_dim"],
    ).to(device)

    _print_sequence_predictor_params(sequence_predictor)
    return sequence_predictor


# Training loops

# @title Training loop for the sequence predictor
"""
The main training loop:
1. Iterates over epochs and batches.
2. Performs the forward pass to get predictions and latent representations.
3. Computes the **Base Losses**: Image L1, Context MSE, Text CrossEntropy.
4. Computes **CoT Grounding Losses** (if data is valid):
   - `loss_reid`: Visual consistency for re-identified entities.
   - `loss_ground_mse`: Embedding alignment between ROI and text.
   - `loss_contrast`: Contrastive loss for ROI-Text alignment.
   - `loss_entity_pool`: Consistency within the batch for the same entity.
5. Backpropagates total loss and updates weights.
6. Runs the validation visualization at the end of each epoch.
"""


# SHARED TRAINING FUNCTION.
# Baseline and all experiment wrappers reuse this loop with different model builders.
def train_sequence_predictor(
    config_path: str,
    show_validation: bool = False,
    model_builder=build_model,
    config_overrides=None,
):
    """Train the selected baseline or experiment model."""
    config = load_config(config_path)
    if config_overrides:
        for section, values in config_overrides.items():
            if isinstance(values, dict):
                config.setdefault(section, {}).update(values)
            else:
                config[section] = values
    set_seed(config.get("seed", 42))
    ensure_dirs(config["paths"]["checkpoint_dir"], config["paths"]["results_dir"])

    # @title Variables and initial setup
    """
    Global setup:
    - Sets the computation device (CUDA/CPU).
    - Defines hyperparameters: `N_EPOCHS`, `emb_dim`, `latent_dim`, etc.
    """
    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = build_tokenizer()
    train_dataset, test_dataset = load_storyreasoning(config)
    train_loader, val_loader, _ = build_sequence_dataloaders(config, tokenizer, train_dataset, test_dataset)
    sequence_predictor = model_builder(config, tokenizer, device)

    # @title Training tools
    """
    Sets up the optimization process:
    1. `criterion_images`: L1 Loss for image reconstruction.
    2. `criterion_ctx`: MSE Loss for context guidance (mean color).
    3. `criterion_text`: CrossEntropy Loss for text generation.
    4. `optimizer`: Adam optimizer for updating model weights.
    """
    criterion_images = nn.L1Loss()
    criterion_ctx = nn.MSELoss()
    criterion_text = nn.CrossEntropyLoss(ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
    optimizer = torch.optim.Adam(
        (param for param in sequence_predictor.parameters() if param.requires_grad),
        lr=config["training"]["learning_rate"],
    )

    # Instantiate the model, define loss and optimizer

    # --- CoT-loss weights (added) ---
    losses = []
    training_log = []

    for epoch in range(config["training"]["epochs"]):
        sequence_predictor.train()
        running_loss = 0.0
        for (frames, descriptions, image_target, text_target,
             roi1, roi2, roi_valid, roi_frame, ent_id) in train_loader:

            # Send images and tokens to the GPU
            descriptions = descriptions.to(device)
            frames = frames.to(device)
            image_target = image_target.to(device)
            text_target = text_target.to(device)

            roi1 = roi1.to(device)
            roi2 = roi2.to(device)
            roi_valid = roi_valid.to(device)
            roi_frame = roi_frame.to(device)

            optimizer.zero_grad()

            # Predictions from our model (+ per-frame latents for CoT grounding)
            pred_image_content, pred_image_context, predicted_text_logits_k, _, _, z_v_seq, z_t_seq = sequence_predictor(
                frames, descriptions, text_target
            )

            # -------------------------
            # Base losses (unchanged)
            # -------------------------
            loss_im = criterion_images(pred_image_content, image_target)

            mu_global = frames.mean(dim=[0, 1])
            mu_global = mu_global.unsqueeze(0).expand_as(pred_image_context)
            loss_context = criterion_ctx(pred_image_context, mu_global)

            prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
            target_labels = text_target.squeeze(1)[:, 1:]  # shift for teacher forcing
            target_flat = target_labels.reshape(-1)
            loss_text = criterion_text(prediction_flat, target_flat)

            # -------------------------
            # CoT-based grounding losses (added)
            # -------------------------
            loss_reid = torch.tensor(0.0, device=device)
            loss_ground_mse = torch.tensor(0.0, device=device)
            loss_contrast = torch.tensor(0.0, device=device)
            loss_entity_pool = torch.tensor(0.0, device=device)

            if roi_valid.any():
                mask = roi_valid.bool()
                if mask.sum() > 0:
                    z_r1 = sequence_predictor.image_encoder(roi1[mask])  # [M,D]
                    z_r2 = sequence_predictor.image_encoder(roi2[mask])  # [M,D]

                    # Simplest grounding: same entity across frames -> close in embedding
                    loss_reid = F.mse_loss(z_r1, z_r2)

                    z_t_match = None
                    # Option 2: frame-aware grounding MSE (ROI aligned to the description embedding of its frame)
                    if config["cot"]["use_frame_aware_grounding"]:
                        f = roi_frame[mask].clamp(min=0, max=z_t_seq.size(1) - 1)  # [M]
                        z_t_match = z_t_seq[mask].gather(
                            1, f.view(-1, 1, 1).expand(-1, 1, z_t_seq.size(-1))
                        ).squeeze(1)  # [M,D]
                        loss_ground_mse = F.mse_loss(z_r1, z_t_match)

                    # Option 1: contrastive ROI↔text grounding (InfoNCE with batch negatives)
                    if config["cot"]["use_contrastive_roi"] and z_t_match is not None:
                        # Normalize for cosine similarity
                        z_img = F.normalize(z_r1, dim=-1)
                        z_txt = F.normalize(z_t_match, dim=-1)
                        logits = (z_img @ z_txt.t()) / config["cot"]["contrastive_tau"]  # [M,M]
                        labels = torch.arange(logits.size(0), device=device)
                        loss_contrast = F.cross_entropy(logits, labels)

                    # Option 3: entity-specific pooling/consistency across batch
                    if config["cot"]["use_entity_pooling"]:
                        # ent_id comes from the DataLoader as a list of strings
                        ent_list = [ent_id[i] for i, m in enumerate(mask.detach().cpu().tolist()) if m]
                        # group embeddings by entity id and pull to group mean (within-batch)
                        uniq = {}
                        for i_e, eid in enumerate(ent_list):
                            if not eid:
                                continue
                            uniq.setdefault(eid, []).append(i_e)

                        if len(uniq) > 0:
                            pool_losses = []
                            for eid, idxs in uniq.items():
                                if len(idxs) < 2:
                                    continue
                                group = z_r1[idxs]  # [G,D]
                                mean = group.mean(dim=0, keepdim=True)
                                pool_losses.append(F.mse_loss(group, mean.expand_as(group)))
                            if len(pool_losses) > 0:
                                loss_entity_pool = torch.stack(pool_losses).mean()

            # Total loss (base + optional improvements)
            weights = config["loss_weights"]
            loss = loss_im + loss_context + loss_text
            loss = loss + weights["reid"] * loss_reid
            loss = loss + weights["ground_mse"] * loss_ground_mse
            loss = loss + weights["contrastive"] * loss_contrast
            loss = loss + weights["entity_pool"] * loss_entity_pool

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)

        log_line = (
            f"Epoch [{epoch + 1}/{config['training']['epochs']}] Loss: {epoch_loss:.4f}  "
            f"(im={loss_im.item():.3f}, ctx={loss_context.item():.3f}, txt={loss_text.item():.3f}, "
            f"reid={float(loss_reid):.3f}, g_mse={float(loss_ground_mse):.3f}, "
            f"nce={float(loss_contrast):.3f}, entpool={float(loss_entity_pool):.3f})"
        )
        print(log_line)
        training_log.append(log_line)

        # Validation step
        validation(
            sequence_predictor,
            val_loader,
            tokenizer,
            device,
            show=show_validation,
        )
        sequence_predictor.train()  # Set back to train mode

        # Checkpoint saving is not used for the notebook experiments.
        # The required outputs are saved in results/baseline, results/Experiment_1, results/Experiment_2, and results/Experiment_3.

    print(f"Finished training. Final loss: {losses[-1]:.4f}")
    return sequence_predictor, tokenizer, val_loader, losses, training_log

# EXPERIMENT 1 TRAINING FUNCTION.
def train_experiment1(config_path: str, show_validation: bool = False):
    """Train Experiment 1 and write outputs under results/Experiment_1."""
    return train_sequence_predictor(
        config_path,
        show_validation=show_validation,
        model_builder=build_experiment1_model,
        config_overrides={
            "paths": {
                "results_dir": "results/Experiment_1",
            }
        },
    )

# EXPERIMENT 2 TRAINING FUNCTION.
def train_experiment2(config_path: str, show_validation: bool = False):
    """Train Experiment 2 and write outputs under results/Experiment_2."""
    return train_sequence_predictor(
        config_path,
        show_validation=show_validation,
        model_builder=build_experiment2_model,
        config_overrides={
            "paths": {
                "results_dir": "results/Experiment_2",
            }
        },
    )

# EXPERIMENT 3 TRAINING FUNCTION.
def train_experiment3(config_path: str, show_validation: bool = False):
    """Train Experiment 3 and write outputs under results/Experiment_3."""
    return train_sequence_predictor(
        config_path,
        show_validation=show_validation,
        model_builder=build_experiment3_model,
        config_overrides={
            "paths": {
                "results_dir": "results/Experiment_3",
            }
        },
    )

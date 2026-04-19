import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import yaml
from model import SequencePredictor

# Load config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Flags section (NEW)
flags = config.get("flags", {})

USE_FRAME_AWARE_GROUNDING = flags.get("use_frame_aware_grounding", False)
USE_CONTRASTIVE_ROI = flags.get("use_contrastive_roi", False)
USE_ENTITY_POOLING = flags.get("use_entity_pooling", False)
USE_COT_TEXT = flags.get("use_cot_text", False)

# Loss section
CONTRASTIVE_TAU = config["loss"]["contrastive_tau"]


# Weight initialization
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0)


# TRAINING LOOP
def train_sequence_predictor(
    sequence_predictor,
    train_dataloader,
    val_dataloader,
    optimizer,
    tokenizer,
    device,
    N_EPOCHS,
    validation_fn=None,
):

    criterion_images = nn.L1Loss()
    criterion_ctx = nn.MSELoss()
    criterion_text = nn.CrossEntropyLoss(
        ignore_index=tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    )

    LAMBDA_REID = 0.10
    LAMBDA_GROUND_MSE = 0.10
    LAMBDA_CONTRAST = 0.10
    LAMBDA_ENTITY_POOL = 0.05

    sequence_predictor.to(device)
    sequence_predictor.train()

    losses = []

    for epoch in range(N_EPOCHS):

        running_loss = 0.0

        for (
            frames,
            descriptions,
            image_target,
            text_target,
            roi1,
            roi2,
            roi_valid,
            roi_frame,
            ent_id
        ) in train_dataloader:

            frames = frames.to(device)
            descriptions = descriptions.to(device)
            image_target = image_target.to(device)
            text_target = text_target.to(device)

            roi1 = roi1.to(device)
            roi2 = roi2.to(device)
            roi_valid = roi_valid.to(device)
            roi_frame = roi_frame.to(device)

            optimizer.zero_grad()

            # Forward pass
            (
                pred_image_content,
                pred_image_context,
                predicted_text_logits_k,
                _,
                _,
                z_v_seq,
                z_t_seq
            ) = sequence_predictor(frames, descriptions, text_target)

            # Base losses
            loss_im = criterion_images(pred_image_content, image_target)

            mu_global = frames.mean(dim=[0, 1])
            mu_global = mu_global.unsqueeze(0).expand_as(pred_image_context)
            loss_context = criterion_ctx(pred_image_context, mu_global)

            prediction_flat = predicted_text_logits_k.reshape(-1, tokenizer.vocab_size)
            target_labels = text_target.squeeze(1)[:, 1:]
            target_flat = target_labels.reshape(-1)

            loss_text = criterion_text(prediction_flat, target_flat)

            # CoT losses
            loss_reid = torch.tensor(0.0, device=device)
            loss_ground_mse = torch.tensor(0.0, device=device)
            loss_contrast = torch.tensor(0.0, device=device)
            loss_entity_pool = torch.tensor(0.0, device=device)

            if roi_valid.any():
                mask = roi_valid.bool()

                if mask.sum() > 0:
                    z_r1 = sequence_predictor.image_encoder(roi1[mask])
                    z_r2 = sequence_predictor.image_encoder(roi2[mask])

                    loss_reid = F.mse_loss(z_r1, z_r2)

                    if USE_FRAME_AWARE_GROUNDING:
                        f = roi_frame[mask].clamp(0, z_t_seq.size(1) - 1)

                        z_t_match = z_t_seq[mask].gather(
                            1,
                            f.view(-1, 1, 1).expand(-1, 1, z_t_seq.size(-1))
                        ).squeeze(1)

                        loss_ground_mse = F.mse_loss(z_r1, z_t_match)

                    if USE_CONTRASTIVE_ROI and USE_FRAME_AWARE_GROUNDING:
                        z_img = F.normalize(z_r1, dim=-1)
                        z_txt = F.normalize(z_t_match, dim=-1)

                        logits = (z_img @ z_txt.t()) / CONTRASTIVE_TAU
                        labels = torch.arange(logits.size(0), device=device)

                        loss_contrast = F.cross_entropy(logits, labels)

                    if USE_ENTITY_POOLING:
                        ent_list = [
                            ent_id[i]
                            for i, m in enumerate(mask.detach().cpu().tolist())
                            if m
                        ]

                        uniq = {}
                        for i_e, eid in enumerate(ent_list):
                            if not eid:
                                continue
                            uniq.setdefault(eid, []).append(i_e)

                        pool_losses = []
                        for eid, idxs in uniq.items():
                            if len(idxs) < 2:
                                continue

                            group = z_r1[idxs]
                            mean = group.mean(dim=0, keepdim=True)

                            pool_losses.append(
                                F.mse_loss(group, mean.expand_as(group))
                            )

                        if len(pool_losses) > 0:
                            loss_entity_pool = torch.stack(pool_losses).mean()

            # TOTAL LOSS
            loss = (
                loss_im
                + loss_context
                + loss_text
                + LAMBDA_REID * loss_reid
                + LAMBDA_GROUND_MSE * loss_ground_mse
                + LAMBDA_CONTRAST * loss_contrast
                + LAMBDA_ENTITY_POOL * loss_entity_pool
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        losses.append(epoch_loss)

        print(
            f"Epoch [{epoch+1}/{N_EPOCHS}] Loss: {epoch_loss:.4f} "
            f"(im={loss_im.item():.3f}, ctx={loss_context.item():.3f}, txt={loss_text.item():.3f})"
        )

        # Validation (FIXED)
        if validation_fn is not None:
            validation_fn(sequence_predictor, val_dataloader,device)
            sequence_predictor.train()

    return losses


# Utility
def set_seed(seed=42):
    import random
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
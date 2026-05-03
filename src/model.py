# Imports
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset

from .utils import (
    crop_and_resize,
    extract_cot_text_for_frame,
    parse_cot_grounding,
    parse_gdi_text,
    pick_reid_pair,
)


# @title Main dataset
"""
Defines the `SequencePredictionDataset` class, which is the core data provider for the main task.
1. `__getitem__`:
   - Loads 5 frames (4 context + 1 target).
   - Parses text descriptions and optionally appends CoT text.
   - Extracts bounding box crops (ROIs) for grounding tasks if CoT data is available.
   - Returns a tuple containing: sequence images, descriptions, target image, target text, ROI crops, and validity flags.
"""


class SequencePredictionDataset(Dataset):
    def __init__(
        self,
        original_dataset,
        tokenizer,
        K=4,
        max_len=120,
        image_hw=(60, 125),
        use_cot_text=True,
    ):
        super(SequencePredictionDataset, self).__init__()
        self.dataset = original_dataset
        self.tokenizer = tokenizer
        self.K = K
        self.max_len = max_len
        self.image_hw = image_hw
        self.use_cot_text = use_cot_text

        # Potential experiments: Try other transforms!
        self.transform = transforms.Compose([
            transforms.Resize(image_hw),  # Reasonable size based on our previous analysis
            transforms.ToTensor(),        # HxWxC -> CxHxW
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Selects a 5 frame sequence from the dataset. Sets 4 for training and the last one
        as a target.

        Returns:
          frames:        [K, C, H, W]
          descriptions:  [K, T]
          image_target:  [C, H, W]
          target_ids:    [1, T]
          roi1, roi2:    [C, H, W] (cropped from CoT bboxes, if available)
          roi_valid:     0/1
          roi_frame:     frame index for roi1 (0..K-1) if available else -1
          ent_id:        string id for the ROI entity (empty if none)
        """
        frames = self.dataset[idx]["images"]
        image_attributes = parse_gdi_text(self.dataset[idx]["story"])

        # CoT grounding annotations (may be missing / unparseable)
        cot = self.dataset[idx].get("chain_of_thought", "")
        cot_frames = parse_cot_grounding(cot)

        frame_tensors = []
        description_list = []

        for frame_idx in range(self.K):
            image = FT.equalize(frames[frame_idx])
            input_frame = self.transform(image)
            frame_tensors.append(input_frame)

            description = image_attributes[frame_idx]["description"]

            # Option 4: include CoT text snippet for this frame (best-effort)
            if self.use_cot_text:
                cot_txt = extract_cot_text_for_frame(cot, frame_idx)
                if cot_txt:
                    description = description + " [COT] " + cot_txt

            input_ids = self.tokenizer(
                description,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
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
            max_length=self.max_len,
        ).input_ids  # [1, T]

        # ---- CoT ROI pair (Options 1-3 need these) ----
        roi_valid = torch.tensor(0, dtype=torch.long)
        roi1 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi2 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi_frame = torch.tensor(-1, dtype=torch.long)
        ent_id = ""

        pair = pick_reid_pair(cot_frames)
        if pair is not None:
            f1, f2, b1, b2, ent_id = pair
            # We only use ROIs that fall within the input window (0..K-1)
            if (0 <= f1 < self.K) and (0 <= f2 < self.K):
                try:
                    roi1 = crop_and_resize(frames[f1], b1, out_hw=self.image_hw)
                    roi2 = crop_and_resize(frames[f2], b2, out_hw=self.image_hw)
                    roi_valid = torch.tensor(1, dtype=torch.long)
                    roi_frame = torch.tensor(int(f1), dtype=torch.long)
                except Exception:
                    pass

        sequence_tensor = torch.stack(frame_tensors)          # [K, C, H, W]
        description_tensor = torch.stack(description_list)    # [K, T]

        return (
            sequence_tensor,
            description_tensor,
            image_target,
            target_ids,
            roi1,
            roi2,
            roi_valid,
            roi_frame,
            ent_id,
        )

# @title Text task dataset (text autoencoding)
"""
Defines `TextTaskDataset` for pre-training or fine-tuning the text encoder separately.
It simply pulls a random text description from a story to perform text-to-text autoencoding.
"""

class TextTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        num_frames = self.dataset[idx]["frame_count"]
        self.image_attributes = parse_gdi_text(self.dataset[idx]["story"])

        # Pick
        frame_idx = np.random.randint(0, 5)
        description = self.image_attributes[frame_idx]["description"]

        return description  # Returning the whole description

# @title Dataset for image autoencoder task
"""
Defines `AutoEncoderTaskDataset` for pre-training the visual autoencoder.
It retrieves a single random frame from the dataset to learn image reconstruction (Image -> Latent -> Image).
"""

class AutoEncoderTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((240, 500)),  # Reasonable size based on our previous analysis
            transforms.ToTensor(),  # HxWxC -> CxHxW
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        num_frames = self.dataset[idx]["frame_count"]
        frames = self.dataset[idx]["images"]

        # Pick a frame at random
        frame_idx = torch.randint(0, 5, (1,)).item()
        input_frame = self.transform(frames[frame_idx])  # Input to the autoencoder

        return input_frame,  # Returning the image


# @title The text autoencoder (Seq2Seq)
"""
Defines the neural network modules for processing text:
1. `EncoderLSTM`: Encodes text tokens into a hidden state vector using an LSTM.
2. `DecoderLSTM`: Decodes a hidden state back into text tokens using an LSTM.
3. `Seq2SeqLSTM`: Combines the encoder and decoder into a full autoencoder architecture.
"""

class EncoderLSTM(nn.Module):
    """
      Encodes a sequence of tokens into a latent space representation.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell

class DecoderLSTM(nn.Module):
    """
      Decodes a latent space representation into a sequence of tokens.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(hidden_dim, vocab_size) # Should be hidden_dim

    def forward(self, input_seq, hidden, cell):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell

# We create the basic text autoencoder (a special case of a sequence to sequence model)
class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, target_seq):
        # input_seq and target_seq are both your 'input_ids'
        # Encode the input sequence
        _enc_out, hidden, cell = self.encoder(input_seq)

        # Create the "shifted" decoder input for teacher forcing.
        # We want to predict target_seq[:, 1:]
        # So, we feed in target_seq[:, :-1]
        # (i.e., feed "[SOS], hello, world" to predict "hello, world, [EOS]")
        decoder_input = target_seq[:, :-1]

        # Run the decoder *once* on the entire sequence.
        # It takes the encoder's final state (hidden, cell)
        # and the full "teacher" sequence (decoder_input).
        predictions, _hidden, _cell = self.decoder(decoder_input, hidden, cell)

        # predictions shape will be (batch_size, seq_len-1, vocab_size)
        return predictions


# The visual autoencoder
"""
Defines the computer vision modules:
1. `Backbone`: A CNN that processes input images into feature maps.
2. `VisualEncoder`: Uses two backbones to separate 'content' and 'context' features, projecting them to a latent space.
3. `VisualDecoder`: Reconstructs images from the latent representation using Transposed Convolutions.
4. `VisualAutoencoder`: The container class for the encoder and decoder.
"""

class Backbone(nn.Module):
    """
      Main convolutional blocks for our CNN
    """
    def __init__(self, latent_dim=16, output_w = 8, output_h = 16):
        super(Backbone, self).__init__()
        # Encoder convolutional layers
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3),
            nn.GroupNorm(8, 16),
            nn.LeakyReLU(0.1),

            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(0.1),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1),
        )

        # Calculate flattened dimension for linear layer
        self.flatten_dim = 64 * output_w * output_h
        # Latent space layers
        self.fc1 = nn.Sequential(nn.Linear(self.flatten_dim, latent_dim), nn.ReLU())


    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, self.flatten_dim)  # flatten for linear layer
        z = self.fc1(x)
        return z


# Visual Encoder

class VisualEncoder(nn.Module):
    """
      Encodes an image into a latent space representation. Note the two pathways
      to try to disentangle the mean pattern from the image
    """
    def __init__(self, latent_dim=16, output_w = 8, output_h = 16):
        super(VisualEncoder, self).__init__()

        self.context_backbone = Backbone(latent_dim, output_w, output_h)
        self.content_backbone = Backbone(latent_dim, output_w, output_h)

        self.projection = nn.Linear(2*latent_dim, latent_dim)
    def forward(self, x):
        z_context = self.context_backbone(x)
        z_content = self.content_backbone(x)
        z = torch.cat((z_content, z_context), dim=1)
        z = self.projection(z)
        return z


# Visual Decoder

class VisualDecoder(nn.Module):
    """
      Decodes a latent representation into a content image and a context image
    """
    def __init__(self, latent_dim=16, output_w = 8, output_h = 16):
        super(VisualDecoder, self).__init__()
        self.imh = 60
        self.imw = 125
        self.flatten_dim = 64 * output_w * output_h
        self.output_w = output_w
        self.output_h = output_h

        self.fc1 = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder_conv = nn.Sequential(
          nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1,1)),
          nn.GroupNorm(8, 32),
          nn.LeakyReLU(0.1),

          nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
          nn.GroupNorm(8, 16),
          nn.LeakyReLU(0.1),

          nn.ConvTranspose2d(16, 3, kernel_size=7, stride=2, padding=3, output_padding=(1, 1)),
          nn.Sigmoid() # Use nn.Tanh() if your data is normalized to [-1, 1]
      )

    def forward(self, z):
      x = self.fc1(z)
      x_content = self.decode_image(x)
      x_context = self.decode_image(x)

      return x_content, x_context

    def decode_image(self, x):
      x = x.view(-1, 64, self.output_w, self.output_h)      # reshape to conv feature map
      x = self.decoder_conv(x)
      x = x[:, :, :self.imh, :self.imw]          # crop to original size if needed
      return x


# Visual Autoencoder

class VisualAutoencoder( nn.Module):
    def __init__(self, latent_dim=16, output_w = 8, output_h = 16):
        super(VisualAutoencoder, self).__init__()
        self.encoder = VisualEncoder(latent_dim, output_w, output_h)
        self.decoder = VisualDecoder(latent_dim, output_w, output_h)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# ATTENTION MODULE


# @title A simple attention architecture
"""
Defines an `Attention` module.
It computes attention weights over a sequence of RNN outputs to create a context vector, helping the model focus on relevant parts of the input sequence.
"""


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # This "attention" layer learns a query vector
        self.attn = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)  # Over the sequence length

    def forward(self, rnn_outputs):
        # rnn_outputs shape: [batch, seq_len, hidden_dim]

        # Pass through linear layer to get "energy" scores
        energy = self.attn(rnn_outputs).squeeze(2)  # Shape: [batch, seq_len]

        # Get attention weights
        attn_weights = self.softmax(energy)  # Shape: [batch, seq_len]

        # Apply weights
        # attn_weights.unsqueeze(1) -> [batch, 1, seq_len]
        # bmm with rnn_outputs -> [batch, 1, hidden_dim]
        context = torch.bmm(attn_weights.unsqueeze(1), rnn_outputs)

        # Squeeze to get final context vector
        return context.squeeze(1)  # Shape: [batch, hidden_dim]


# @title The main sequence predictor model
"""
This is the core architecture `SequencePredictor`.
1. **Encoders**: Uses the `image_encoder` and `text_encoder` to process the sequence of 4 input frames and descriptions.
2. **Temporal Fusion**: A GRU processes the sequence of fused (image+text) embeddings to capture temporal dynamics.
3. **Attention**: Applies attention over the sequence to summarize context.
4. **Decoders**: Predicts the *next* (5th) frame's image and text using `image_decoder` and `text_decoder`.
"""

# BASELINE ARCHITECTURE.

class SequencePredictor(nn.Module):
    def __init__(self, visual_autoencoder, text_autoencoder, latent_dim, gru_hidden_dim):
        super(SequencePredictor, self).__init__()

        # --- 1. Static Encoders ---
        # (These process one pair at a time)
        self.image_encoder = visual_autoencoder.encoder
        self.text_encoder = text_autoencoder.encoder

        # --- 2. Temporal Encoder ---
        fusion_dim = latent_dim * 2  # z_visual + z_text
        self.temporal_rnn = nn.GRU(fusion_dim, latent_dim, batch_first=True)

        # --- 3. Attention ---
        self.attention = Attention(gru_hidden_dim)

        # --- 4. Final Projection ---
        # cat(h, context) -> gru_hidden_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, latent_dim),
            nn.ReLU(),
        )

        # --- 5. Decoders ---
        # (These predict the *next* item)
        self.image_decoder = visual_autoencoder.decoder
        self.text_decoder = text_autoencoder.decoder

        self.fused_to_h0 = nn.Linear(latent_dim, 16)
        self.fused_to_c0 = nn.Linear(latent_dim, 16)

    def forward(self, image_seq, text_seq, target_seq):
        # image_seq shape: [batch, seq_len, C, H, W]
        # text_seq shape:  [batch, seq_len, text_len]
        # target_text_for_teacher_forcing: [batch, text_len] (This is the last text)

        batch_size, seq_len, C, H, W = image_seq.shape

        # --- 1 & 2: Run Static Encoders over the sequence ---
        # We can't pass a 5D/4D tensor to the encoders.
        # We "flatten" the batch and sequence dimensions.

        # Reshape for image_encoder
        img_flat = image_seq.view(batch_size * seq_len, C, H, W)
        # Reshape for text_encoder
        txt_flat = text_seq.view(batch_size * seq_len, -1)  # -1 infers text_len

        # Run encoders
        z_v_flat = self.image_encoder(img_flat)  # Shape: [b*s, latent]
        _, hidden, cell = self.text_encoder(txt_flat)  # Shape: [b*s, latent]

        # Keep per-frame latents for optional grounding losses
        z_v_seq = z_v_flat.view(batch_size, seq_len, -1)                 # [b, s, latent]
        z_t_seq = hidden.squeeze(0).view(batch_size, seq_len, -1)        # [b, s, latent]

        # Combine
        z_fusion_flat = torch.cat((z_v_flat, hidden.squeeze(0)), dim=1)  # Shape: [b*s, fusion_dim]

        # "Un-flatten" back into a sequence
        z_fusion_seq = z_fusion_flat.view(batch_size, seq_len, -1)  # Shape: [b, s, fusion_dim]

        # --- 3. Run Temporal Encoder ---
        # zseq shape: [b, s, gru_hidden]
        # h    shape: [1, b, gru_hidden]
        zseq, h = self.temporal_rnn(z_fusion_seq)
        h = h.squeeze(0)  # Shape: [b, gru_hidden]

        # --- 4. Attention ---
        context = self.attention(zseq)  # Shape: [b, gru_hidden]

        # --- 5. Final Prediction Vector (z) ---
        z = self.projection(torch.cat((h, context), dim=1))  # Shape: [b, joint_latent_dim]

        # --- 6. Decode (Predict pk) ---
        pred_image_content, pred_image_context = self.image_decoder(z)

        h0 = self.fused_to_h0(z).unsqueeze(0)
        c0 = self.fused_to_c0(z).unsqueeze(0)

        decoder_input = target_seq[:, :, :-1].squeeze(1)

        # 3. Run the decoder *once* on the entire sequence.
        # It takes the encoder's final state (hidden, cell)
        # and the full "teacher" sequence (decoder_input).
        predicted_text_logits_k, _hidden, _cell = self.text_decoder(decoder_input, h0, c0)

        return pred_image_content, pred_image_context, predicted_text_logits_k, h0, c0, z_v_seq, z_t_seq


# Experiment architecture variants
# EXPERIMENT 1 GROUNDING MODULE - Cross-Modal Attention.
class CrossModalAttention(nn.Module):
    """
    Experiment 1 grounding module.

    Text token embeddings act as queries, and visual spatial feature-map
    locations act as keys/values. The output is a text representation grounded
    in the image regions.
    """
    def __init__(self, text_dim=16, visual_channels=64, latent_dim=16):
        super().__init__()
        self.query_proj = nn.Linear(text_dim, latent_dim)
        self.key_proj = nn.Linear(visual_channels, latent_dim)
        self.value_proj = nn.Linear(visual_channels, latent_dim)
        self.output_proj = nn.Linear(latent_dim, latent_dim)
        self.scale = math.sqrt(latent_dim)

    def forward(self, token_embeddings, visual_feature_map, token_ids=None):
        """Compute token-to-region attention and return grounded text features."""
        visual_tokens = visual_feature_map.flatten(2).transpose(1, 2)

        queries = self.query_proj(token_embeddings)
        keys = self.key_proj(visual_tokens)
        values = self.value_proj(visual_tokens)

        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attended_visual_tokens = torch.bmm(attention_weights, values)

        if token_ids is not None:
            token_mask = (token_ids != 0).float().unsqueeze(-1)
            grounded_text = (attended_visual_tokens * token_mask).sum(dim=1)
            grounded_text = grounded_text / token_mask.sum(dim=1).clamp_min(1.0)
        else:
            grounded_text = attended_visual_tokens.mean(dim=1)

        grounded_text = self.output_proj(grounded_text)
        return grounded_text, attention_weights


# EXPERIMENT 1 ARCHITECTURE.
# Cross-Modal Attention grounding + unidirectional GRU.
class Experiment1SequencePredictor(nn.Module):
    """
    Experiment 1 model: Cross-Modal Attention + baseline unidirectional GRU.
    """
    def __init__(self, visual_autoencoder, text_autoencoder, latent_dim, gru_hidden_dim):
        super().__init__()

        # --- 1. Static Encoders ---
        # (These process one pair at a time)
        self.image_encoder = visual_autoencoder.encoder
        self.text_encoder = text_autoencoder.encoder

        self.cross_modal_attention = CrossModalAttention(
            text_dim=text_autoencoder.encoder.embedding_dim,
            visual_channels=64,
            latent_dim=latent_dim,
        )

        # --- 2. Temporal Encoder ---
        # Sequence predictor remains the baseline unidirectional GRU.
        fusion_dim = latent_dim * 2  # z_visual + grounded z_text
        self.temporal_rnn = nn.GRU(fusion_dim, gru_hidden_dim, batch_first=True)

        # --- 3. Attention ---
        self.attention = Attention(gru_hidden_dim)

        # --- 4. Final Projection ---
        # cat(h, context) -> gru_hidden_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, latent_dim),
            nn.ReLU(),
        )

        # --- 5. Decoders ---
        # (These predict the *next* item)
        self.image_decoder = visual_autoencoder.decoder
        self.text_decoder = text_autoencoder.decoder

        self.fused_to_h0 = nn.Linear(latent_dim, latent_dim)
        self.fused_to_c0 = nn.Linear(latent_dim, latent_dim)
        self.last_cross_attention = None
        self.last_cross_attention_hw = None

    def forward(self, image_seq, text_seq, target_seq):
        """Run Experiment 1 forward pass and store attention weights for heatmaps."""
        # image_seq shape: [batch, seq_len, C, H, W]
        # text_seq shape:  [batch, seq_len, text_len]
        # target_text_for_teacher_forcing: [batch, text_len] (This is the last text)

        batch_size, seq_len, channels, height, width = image_seq.shape

        # --- 1 & 2: Run Static Encoders over the sequence ---
        # We can't pass a 5D/4D tensor to the encoders.
        # We "flatten" the batch and sequence dimensions.

        # Reshape for image_encoder
        img_flat = image_seq.view(batch_size * seq_len, channels, height, width)
        # Reshape for text_encoder
        txt_flat = text_seq.view(batch_size * seq_len, -1)  # -1 infers text_len

        # Run encoders
        z_v_flat = self.image_encoder(img_flat)  # Shape: [b*s, latent]

        token_embeddings = self.text_encoder.embedding(txt_flat)
        _, (hidden, _cell) = self.text_encoder.lstm(token_embeddings)
        text_latent = hidden[-1]

        visual_feature_map = self.image_encoder.content_backbone.encoder_conv(img_flat)
        grounded_text_flat, cross_attention = self.cross_modal_attention(
            token_embeddings,
            visual_feature_map,
            token_ids=txt_flat,
        )

        self.last_cross_attention = cross_attention.detach().view(batch_size, seq_len, txt_flat.size(1), -1)
        self.last_cross_attention_hw = visual_feature_map.shape[-2:]

        # Keep per-frame latents for optional grounding losses
        z_v_seq = z_v_flat.view(batch_size, seq_len, -1)          # [b, s, latent]
        z_t_seq = grounded_text_flat.view(batch_size, seq_len, -1)       # [b, s, latent]

        # Experiment 1 fusion: replace baseline text latent with cross-modal grounded text.
        # Baseline reference: z_fusion_flat = torch.cat((z_v_flat, text_latent), dim=1)
        z_fusion_flat = torch.cat((z_v_flat, grounded_text_flat), dim=1)  # Shape: [b*s, fusion_dim]

        # "Un-flatten" back into a sequence
        z_fusion_seq = z_fusion_flat.view(batch_size, seq_len, -1)  # Shape: [b, s, fusion_dim]

        # --- 3. Run Temporal Encoder ---
        # zseq shape: [b, s, gru_hidden]
        # h    shape: [1, b, gru_hidden]
        zseq, h = self.temporal_rnn(z_fusion_seq)
        h = h.squeeze(0)  # Shape: [b, gru_hidden]

        # --- 4. Attention ---
        context = self.attention(zseq)  # Shape: [b, gru_hidden]

        # --- 5. Final Prediction Vector (z) ---
        z = self.projection(torch.cat((h, context), dim=1))  # Shape: [b, joint_latent_dim]

        # --- 6. Decode (Predict pk) ---
        pred_image_content, pred_image_context = self.image_decoder(z)

        h0 = self.fused_to_h0(z).unsqueeze(0)
        c0 = self.fused_to_c0(z).unsqueeze(0)

        decoder_input = target_seq[:, :, :-1].squeeze(1)

        # 3. Run the decoder *once* on the entire sequence.
        # It takes the encoder's final state (hidden, cell)
        # and the full "teacher" sequence (decoder_input).
        predicted_text_logits_k, _hidden, _cell = self.text_decoder(decoder_input, h0, c0)

        return pred_image_content, pred_image_context, predicted_text_logits_k, h0, c0, z_v_seq, z_t_seq

# -*- coding: utf-8 -*-
# START OF FILE model.py
# V2 - Aligning AE parts closer to gen.py description

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
from scipy.interpolate import CubicSpline, interp1d # Added interp1d
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder # Added OneHotEncoder
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from tqdm import tqdm
import warnings
from sklearn.manifold import TSNE
import random # Added random
import time # Added time
import math # Added math
from collections import Counter # Added Counter
import traceback # Added traceback
from itertools import combinations
warnings.filterwarnings('ignore')


# --- Configuration & Constants ---
# Match original model.py settings where applicable, but use AE-specific ones
SEGMENT_LENGTH = 1024
OVERLAP = 0.5
STEP = int(SEGMENT_LENGTH * (1 - OVERLAP))
if STEP < 1: STEP = 1
DEFAULT_WAVELET = 'db4' # As used in gen.py preprocess description
DEFAULT_WAVELET_LEVEL = 3 # As used in gen.py preprocess description

# AE Specific Config (closer to gen.py defaults if possible)
AE_LATENT_DIM = 64 # Keep from original model.py latent_dim setting for AE
AE_EPOCHS = 100# Epochs specifically for AE training (adjust as needed)
AE_LR = 0.001 # Learning rate specifically for AE training (gen.py used 0.001 * 2 = 0.002?) -> Let's try 0.001 first.
AE_BATCH_SIZE = 64 # Batch size specifically for AE training
AE_CONTRASTIVE_WEIGHT = 1.2 # From gen.py description
AE_NOISE_STD = 0.05 # Noise added inside AE training loop, from gen.py description

# CNN/SEN Config
CNN_EPOCHS = 1
CNN_LR = 0.0005
SEN_EPOCHS = 1
SEN_LR = 0.001
CNN_FEATURE_DIM = 512 # Default output feature dimension for CNN fusion & SEN target
DEFAULT_BATCH_SIZE = 64


# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed) # Added random
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


# --- Helper Function create_segments (Ensure finite check) ---
def create_segments(signal_data, segment_length, step):
    """Creates overlapping segments from a signal, ensuring segments are finite."""
    segments = []
    num_samples = len(signal_data)
    if num_samples >= segment_length:
        for i in range(0, num_samples - segment_length + 1, step):
            segment = signal_data[i: i + segment_length]
            if len(segment) == segment_length:
                # Ensure segment contains finite values before adding
                if np.all(np.isfinite(segment)):
                    segments.append(segment)
                # else: # Optional warning
                    # print(f"W: Segment starting at index {i} contains non-finite values, skipping.")
                    # pass
    # Return np.ndarray, even if empty
    return np.array(segments, dtype=np.float32) if segments else np.empty((0, segment_length), dtype=np.float32)

# 1. 数据预处理模块 (MODIFIED to closer match gen.py's preprocess_data logic)
class DataPreprocessor:
    def __init__(self, sample_length=SEGMENT_LENGTH, overlap=OVERLAP, augment=True): # Use constants
        self.sample_length = sample_length
        self.overlap = overlap
        self.augment = augment
        self.stride = int(sample_length * (1 - overlap)) # Recalculate stride here
        if self.stride < 1 : self.stride = 1 # Ensure stride is at least 1
        # Initialize the single scaler used at the end
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    # cubic_spline_interpolation remains the same as the improved V1

    def cubic_spline_interpolation(self, signal_with_missing):
        """使用三次样条插值法处理缺失值 (Improved fallback)"""
        mask = np.isnan(signal_with_missing)
        if np.any(mask):
            x = np.arange(len(signal_with_missing))
            x_known = x[~mask]
            y_known = signal_with_missing[~mask]
            if len(x_known) >= 2: # Need at least 2 points for linear, 4 for cubic
                if len(x_known) >= 4:
                    kind = 'cubic'
                else:
                    kind = 'linear'
                try:
                    # Use interp1d which handles both linear and cubic (via spline)
                    f = interp1d(x_known, y_known, kind=kind, fill_value="extrapolate", bounds_error=False)
                    signal_with_missing[mask] = f(x[mask])
                except ValueError as e:
                    print(f"W: Interpolation failed ({kind}): {e}. Using mean fill.")
                    mean_val = np.nanmean(y_known) if len(y_known) > 0 else 0
                    signal_with_missing[mask] = mean_val # Fallback to mean
            elif len(x_known) == 1:
                signal_with_missing[mask] = y_known[0] # Fill with the single known value
            else:
                signal_with_missing[mask] = 0.0 # Fill with zero if no known points
        # Ensure no NaNs remain
        return np.nan_to_num(signal_with_missing, nan=0.0)


    def remove_outliers_sigma(self, signal_data, sigma_threshold=5):
        """Simpler outlier removal using sigma-clipping (like gen.py's intent)"""
        signal_data_clean = np.nan_to_num(signal_data, nan=np.nanmedian(signal_data))
        mean = np.mean(signal_data_clean)
        std = np.std(signal_data_clean)
        if std < 1e-9: # Avoid division by zero std
             return signal_data # Return original if no variation
        lower_bound = mean - sigma_threshold * std
        upper_bound = mean + sigma_threshold * std
        # Clip the original data using these bounds
        return np.clip(signal_data, lower_bound, upper_bound)

    def wavelet_denoising_universal(self, signal_data, wavelet=DEFAULT_WAVELET, level=DEFAULT_WAVELET_LEVEL):
        """Wavelet denoising using Universal Threshold (closer to gen.py's description)"""
        signal_data = np.nan_to_num(signal_data, nan=0.0)
        data_len = len(signal_data)
        if data_len == 0: return signal_data

        try:
            # Use the correct function for 1D signals: dwt_max_level
            max_level = pywt.dwt_max_level(data_len, pywt.Wavelet(wavelet))  # <<< FIXED HERE
            actual_level = min(level, max_level)
            # Ensure level is valid (>=1) before proceeding
            if actual_level < 1:
                # print(f"W: Cannot perform wavelet decomposition with level {actual_level} for data length {data_len}. Skipping denoising.")
                return signal_data  # Return original signal if level is invalid

            coeffs = pywt.wavedec(signal_data, wavelet, level=actual_level)
            detail_coeffs = coeffs[1:]

            # Noise sigma estimation (using MAD)
            sigma = 0.0
            if len(detail_coeffs) > 0 and detail_coeffs[-1] is not None and len(detail_coeffs[-1]) > 0:
                median_abs_dev = np.median(np.abs(detail_coeffs[-1] - np.median(detail_coeffs[-1])))  # More robust MAD
                sigma = median_abs_dev / 0.6745 if median_abs_dev > 1e-9 else 0.0
            else:  # Fallback if last level is unusable
                valid_coeffs = [c for c in detail_coeffs if c is not None and len(c) > 1]
                if valid_coeffs:
                    median_abs_dev = np.median([np.median(np.abs(c - np.median(c))) for c in valid_coeffs])
                    sigma = median_abs_dev / 0.6745 if median_abs_dev > 1e-9 else 0.0

            # Universal Threshold calculation
            thr = sigma * np.sqrt(2 * np.log(max(data_len, 1))) if sigma > 1e-9 and data_len > 1 else 0.0

            # Apply thresholding
            coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, thr, mode='soft') if c is not None else None for c in
                                           coeffs[1:]]

            # Reconstruction
            denoised_signal = pywt.waverec(coeffs_thresh, wavelet)

            # Ensure output length matches input
            if len(denoised_signal) != data_len:
                if len(denoised_signal) > data_len:
                    denoised_signal = denoised_signal[:data_len]
                else:
                    padding = data_len - len(denoised_signal)
                    denoised_signal = np.pad(denoised_signal, (0, padding), 'edge')

        except Exception as e:
            print(f"W: Wavelet denoising failed unexpectedly: {e}. Returning original signal.")
            traceback.print_exc()
            denoised_signal = signal_data  # Fallback

        # Final check for NaNs introduced during processing
        if not np.all(np.isfinite(denoised_signal)):
            print("W: NaN detected after wavelet denoising. Replacing with 0.")
            denoised_signal = np.nan_to_num(denoised_signal, nan=0.0)

        return denoised_signal

    # Butterworth filter is removed as it's not in gen.py's core preprocessing

    def add_gaussian_noise(self, signal_data, std_dev_factor=0.01):
        """Adds Gaussian noise based on signal's std dev (more relative)."""
        signal_data = np.nan_to_num(signal_data, nan=0.0)
        signal_std = np.std(signal_data)
        # Use a small base std dev if signal_std is zero
        noise_std = signal_std * std_dev_factor if signal_std > 1e-9 else 1e-6 * std_dev_factor
        noise = np.random.normal(0, noise_std, len(signal_data))
        return signal_data + noise

    def time_augmentation(self, signal_data, max_shift_ratio=0.02, scale_range=(0.98, 1.02)):
        """Simpler time augmentation: shift and scale."""
        signal_data = np.nan_to_num(signal_data, nan=0.0)
        data_len = len(signal_data)
        if data_len == 0: return signal_data

        # 1. Random Shift
        max_shift = max(1, int(max_shift_ratio * data_len))
        shift = np.random.randint(-max_shift, max_shift + 1)
        shifted_signal = np.roll(signal_data, shift)

        # 2. Random Scaling
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        scaled_signal = shifted_signal * scale_factor

        return scaled_signal

    def segment_signal(self, signal_data):
        """Uses the helper create_segments function."""
        return create_segments(signal_data, self.sample_length, self.stride)

    def preprocess(self, signal_data, augmentation=False):
        """Complete preprocessing flow, closer to gen.py."""
        # Initial NaN check
        if not np.all(np.isfinite(signal_data)):
             print("W: Input signal contains non-finite values. Attempting initial cleanup.")
             signal_data = np.nan_to_num(signal_data, nan=np.nanmean(signal_data), posinf=np.nanmax(signal_data), neginf=np.nanmin(signal_data))
             if not np.all(np.isfinite(signal_data)):
                  print("E: Could not clean non-finite values. Returning empty result.")
                  return np.empty((0, self.sample_length))

        # 1. Missing value interpolation (e.g., linear/cubic)
        signal_data = self.cubic_spline_interpolation(signal_data)

        # 2. Outlier removal (Sigma clipping)
        signal_data = self.remove_outliers_sigma(signal_data, sigma_threshold=5)

        # 3. Wavelet Denoising (Universal Threshold)
        signal_data = self.wavelet_denoising_universal(signal_data)

        # 4. Normalization (MinMax Scaler applied ONCE at the end of signal processing)
        # Needs reshape for scaler
        signal_data_normalized = self.scaler.fit_transform(signal_data.reshape(-1, 1)).flatten()

        # Ensure final normalized data is finite
        if not np.all(np.isfinite(signal_data_normalized)):
            print("W: Non-finite values after normalization. Replacing with 0.")
            signal_data_normalized = np.nan_to_num(signal_data_normalized, nan=0.0)

        # --- Augmentation (Applied to Normalized Signal) ---
        if augmentation and self.augment:
            processed_signals_for_segmentation = []
            # Include the original normalized signal
            processed_signals_for_segmentation.append(signal_data_normalized)

            # a) Add Gaussian Noise
            for _ in range(2): # Add noise a couple of times
                noisy_sig = self.add_gaussian_noise(signal_data_normalized, std_dev_factor=0.02) # Smaller noise factor?
                noisy_sig_clamped = np.clip(noisy_sig, -1.0, 1.0) # CLAMP augmented signal
                if np.all(np.isfinite(noisy_sig_clamped)):
                    processed_signals_for_segmentation.append(noisy_sig_clamped)

            # b) Time Augmentation
            for _ in range(2):
                time_aug_sig = self.time_augmentation(signal_data_normalized)
                time_aug_sig_clamped = np.clip(time_aug_sig, -1.0, 1.0) # CLAMP augmented signal
                if np.all(np.isfinite(time_aug_sig_clamped)):
                    processed_signals_for_segmentation.append(time_aug_sig_clamped)

            # --- Segmentation (Applied AFTER processing and augmentation) ---
            all_segments = []
            for sig in processed_signals_for_segmentation:
                if np.all(np.isfinite(sig)):
                    segments = self.segment_signal(sig)
                    if segments.shape[0] > 0:
                        all_segments.append(segments)
                else:
                     print("W: Augmented signal became non-finite before segmentation. Skipping.")

            if not all_segments:
                 print("W: No valid segments generated after augmentation.")
                 return np.empty((0, self.sample_length))

            return np.vstack(all_segments)
        else:
            # --- Segmentation (Applied AFTER processing, no augmentation) ---
            return self.segment_signal(signal_data_normalized)


# --- Autoencoder Definition (Exact match to gen.py architecture) ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim=SEGMENT_LENGTH, latent_dim=AE_LATENT_DIM):
        super(Autoencoder, self).__init__()
        # Hidden dimension calculation identical to gen.py
        h1, h2 = 256, 128
        if input_dim < h1: h1 = max(latent_dim, (input_dim + latent_dim) // 2)
        if h1 < h2: h2 = max(latent_dim, (h1 + latent_dim) // 2)
        if h2 < latent_dim: h2 = latent_dim
        h1 = max(h1, h2)
        latent_dim = min(latent_dim, h2)

        # Ensure dimensions are positive
        if not (input_dim > 0 and h1 > 0 and h2 > 0 and latent_dim > 0):
             raise ValueError(f"Invalid AE dimensions: In={input_dim}, H1={h1}, H2={h2}, Latent={latent_dim}")

        print(f"AE Arch (gen.py style): {input_dim} -> {h1} -> {h2} -> {latent_dim} -> {h2} -> {h1} -> {input_dim}")

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            # No BatchNorm
            nn.Linear(h1, h2),
            nn.ReLU(),
            # No BatchNorm
            nn.Linear(h2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            # No BatchNorm
            nn.Linear(h2, h1),
            nn.ReLU(),
            # No BatchNorm
            nn.Linear(h1, input_dim),
            nn.Tanh() # Use Tanh since input is scaled to [-1, 1]
        )
        # Store actual latent dim determined by calculation
        self.actual_latent_dim = latent_dim

    def forward(self, x):
        # Input check (optional, but good practice)
        # if not torch.all(torch.isfinite(x)): ...
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded # Return only decoded for standard AE interface

    # encode method needed for extracting features
    def encode(self, x):
        if not torch.all(torch.isfinite(x)):
             print("W: AE encode input contains non-finite values. Clamping.")
             x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        latent = self.encoder(x)
        if not torch.all(torch.isfinite(latent)):
             print("W: AE encode output contains non-finite values. Clamping.")
             latent = torch.nan_to_num(latent, nan=0.0) # Clamp latent space?
        return latent


# 2. 知识语义和数据语义构建模块 (Modified to use gen.py's AE training details)
class FaultSemanticBuilder:
    # 现有初始化方法保持不变
    def __init__(self, latent_dim=64, hidden_dim=128):
        self.latent_dim_config = latent_dim
        self.actual_latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = None
        self.preprocessor = DataPreprocessor(sample_length=1024)
        self.knowledge_dim = 0
        self.data_semantics = {}
        self.idx_to_fault = {}


    # build_knowledge_semantics remains unchanged

    def build_knowledge_semantics(self):
        """构建基于轴承故障位置和尺寸的知识语义 (Unchanged from original)"""
        # Defines fault locations and bearing parameters...
        fault_location = {
            'normal': [0, 0, 0],'inner': [1, 0, 0],'outer': [0, 1, 0],'ball': [0, 0, 1],
            'inner_outer': [1, 1, 0],'inner_ball': [1, 0, 1],'outer_ball': [0, 1, 1],
            'inner_outer_ball': [1, 1, 1]
        }
        bearing_params = {
            'inner_diameter': 17 / 40,'outer_diameter': 1.0,'width': 12 / 40,
            'ball_diameter': 6.75 / 40,'ball_number': 9 / 20
        }
        knowledge_semantics = {}
        first_vec = None
        for fault_type, location_encoding in fault_location.items():
            semantics = np.array(location_encoding + list(bearing_params.values()), dtype=np.float32)
            knowledge_semantics[fault_type] = semantics
            if first_vec is None: first_vec = semantics
        self.knowledge_dim = len(first_vec) if first_vec is not None else 0
        return knowledge_semantics

    # Contrastive loss matching gen.py logic
    def _ae_contrastive_loss(self, latent, latent_aug, labels, temperature=0.2):
        """计算对比损失 (logic matching gen.py _contrastive_loss)."""
        batch_size = latent.size(0)
        if batch_size <= 1: return torch.tensor(0.0, device=latent.device)

        latent_norm = nn.functional.normalize(latent, p=2, dim=1)
        latent_aug_norm = nn.functional.normalize(latent_aug, p=2, dim=1)

        # Check for NaNs after normalization
        if torch.isnan(latent_norm).any() or torch.isnan(latent_aug_norm).any():
            print("W: NaN found after L2 norm in contrastive loss. Using 0 loss for batch.")
            latent_norm = torch.nan_to_num(latent_norm, nan=0.0)
            latent_aug_norm = torch.nan_to_num(latent_aug_norm, nan=0.0)
            # Optionally return 0 loss immediately
            # return torch.tensor(0.0, device=latent.device)


        # Similarity matrix (original vs augmented)
        sim_matrix = torch.matmul(latent_norm, latent_aug_norm.t()) / temperature
        sim_matrix = torch.clamp(sim_matrix, min=-30.0, max=30.0)

        # Positive pair mask (original_i vs augmented_j where labels match)
        labels_eq = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().to(self.device)
        # Diagonal IS included here, comparing original[i] vs augmented[i] (should be positive)

        # InfoNCE Loss calculation
        exp_sim = torch.exp(sim_matrix)
        # Sum exp(sim) for positive pairs for each anchor i
        pos_sum = (exp_sim * labels_eq).sum(dim=1)
        # Sum exp(sim) for all pairs for each anchor i
        total_sum = exp_sim.sum(dim=1) # Denominator includes positive and negative pairs

        contrastive_loss_terms = -torch.log((pos_sum + 1e-8) / (total_sum + 1e-8))

        # Check for NaNs/Infs in the loss terms themselves
        if torch.isnan(contrastive_loss_terms).any() or torch.isinf(contrastive_loss_terms).any():
            print("W: NaN/Inf detected in contrastive loss terms. Using mean of finite terms.")
            finite_loss_terms = contrastive_loss_terms[torch.isfinite(contrastive_loss_terms)]
            if len(finite_loss_terms) > 0:
                mean_loss = finite_loss_terms.mean()
            else:
                print("W: All contrastive loss terms are non-finite. Returning 0 loss.")
                mean_loss = torch.tensor(0.0, device=latent.device)
        else:
             mean_loss = contrastive_loss_terms.mean()


        # Final check on the mean loss
        if torch.isnan(mean_loss) or torch.isinf(mean_loss):
             print("W: AE contrastive loss resulted in NaN/Inf mean. Returning 0.")
             # Optional: Add more diagnostics here if needed
             return torch.tensor(0.0, device=latent.device)

        return mean_loss

    # AE Training loop adjusted to match gen.py
    def train_autoencoder(self, X_train, labels, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, lr=AE_LR):
        """训练自编码器 (Closer alignement with gen.py training logic)"""
        print("Training Autoencoder for data semantics (gen.py aligned)...")
        input_dim = X_train.shape[1]

        # 1. Initialize AE model
        self.autoencoder = Autoencoder(input_dim=input_dim, latent_dim=self.latent_dim_config).to(self.device)
        # Store the actual latent dim used by the initialized AE
        self.actual_latent_dim = self.autoencoder.actual_latent_dim
        if self.actual_latent_dim != self.latent_dim_config:
             print(f"W: AE latent dim adjusted by architecture: {self.actual_latent_dim}")


        # 2. Prepare Data Loaders / Tensors
        if labels is None: raise ValueError("Labels are required for gen.py style contrastive AE training.")

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(labels))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # Ensure shuffle & drop_last

        all_labels_np = labels # Keep for centroid calculation later
        all_data_tensor = torch.FloatTensor(X_train).to(self.device) # Keep all data on device for centroid calc


        # 3. Optimizer and Scheduler (Matching gen.py description)
        # Gen.py used AdamW? Or Adam? Let's use Adam as default. And wd=1e-5.
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        # Gen.py used StepLR? Let's use that for closer replication.
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)


        # 4. Loss Function
        criterion_recon = nn.MSELoss()

        # 5. Training Loop
        self.autoencoder.train()
        num_samples = len(X_train)
        best_loss = float('inf')
        patience_counter = 0
        patience = 15 # Early stopping patience


        for epoch in range(epochs):
            epoch_recon_loss = 0.0
            epoch_contrastive_loss = 0.0
            samples_processed = 0

            for data in train_loader:
                batch_data, batch_labels = data # Labels guaranteed by initial check
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                if batch_data.shape[0] < 2: continue # Need >= 2 for contrastive

                # Augmentation inside loop (noise) - Matches gen.py
                noise = torch.randn_like(batch_data) * AE_NOISE_STD
                batch_aug = torch.clamp(batch_data + noise, -1.0, 1.0) # Clamp needed as input is [-1,1]


                optimizer.zero_grad()

                # Forward Pass
                decoded = self.autoencoder(batch_data)
                decoded_aug = self.autoencoder(batch_aug)
                latent = self.autoencoder.encode(batch_data)
                latent_aug = self.autoencoder.encode(batch_aug)


                # Basic check for NaNs before loss calculation
                if not torch.all(torch.isfinite(decoded)) or not torch.all(torch.isfinite(latent)) or \
                   not torch.all(torch.isfinite(decoded_aug)) or not torch.all(torch.isfinite(latent_aug)):
                    print(f"W: NaN/Inf detected in AE outputs epoch {epoch+1}. Skipping batch loss.")
                    continue

                # Reconstruction Loss (Original vs Decoded AND Augmented vs DecodedAugmented)
                # Should probably reconstruct original from original, augmented from augmented?
                # Let's stick to reconstructing the *original* input for both terms for simplicity.
                # recon_loss = criterion_recon(decoded, batch_data) + criterion_recon(decoded_aug, batch_data) # Reconstruct original?
                # Or reconstruct corresponding input?
                recon_loss = criterion_recon(decoded, batch_data) + criterion_recon(decoded_aug, batch_aug) # Reconstruct input
                recon_loss = recon_loss / 2.0

                # Contrastive Loss (using the specific _ae_contrastive_loss)
                contrastive_loss = self._ae_contrastive_loss(latent, latent_aug, batch_labels)

                # Total Loss (weighted as per gen.py)
                total_loss = recon_loss + AE_CONTRASTIVE_WEIGHT * contrastive_loss

                # Backward and Optimize
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"W: Total AE loss NaN/Inf epoch {epoch+1}. Skipping backward.")
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0) # Clip gradients
                optimizer.step()

                epoch_recon_loss += recon_loss.item() * batch_data.size(0)
                epoch_contrastive_loss += contrastive_loss.item() * batch_data.size(0)
                samples_processed += batch_data.size(0)

            # Scheduler Step (after epoch)
            scheduler.step()

            # Logging and Early Stopping
            if samples_processed > 0:
                avg_recon_loss = epoch_recon_loss / samples_processed
                avg_contrastive_loss = epoch_contrastive_loss / samples_processed
                avg_total_loss = avg_recon_loss + AE_CONTRASTIVE_WEIGHT * avg_contrastive_loss
                print(f"Epoch [{epoch + 1}/{epochs}] - AE Total Loss: {avg_total_loss:.6f} (Recon: {avg_recon_loss:.6f}, Contr: {avg_contrastive_loss:.6f})")

                if avg_total_loss < best_loss:
                     best_loss = avg_total_loss
                     patience_counter = 0
                     torch.save(self.autoencoder.state_dict(), 'best_autoencoder_gen_aligned.pth')
                else:
                     patience_counter += 1
                     if patience_counter >= patience:
                         print(f"AE Early stopping triggered at epoch {epoch + 1}")
                         break
            else:
                 print(f"Epoch [{epoch + 1}/{epochs}] - No samples processed.")


        # --- Post-Training: Load best model & Calculate Centroids ---
        if os.path.exists('best_autoencoder_gen_aligned.pth'):
            self.autoencoder.load_state_dict(torch.load('best_autoencoder_gen_aligned.pth'))
            print("Loaded best AE model state (gen aligned).")
        else:
            print("W: No best AE model state file found. Using the last state.")
        self.autoencoder.eval()

        # --- Calculate Centroids directly from latent features (NO SCALING) ---
        print("Calculating data semantic centroids (no scaling)...")
        self.data_semantics = {}
        all_latent_list = []
        inference_batch_size = batch_size * 4

        if all_data_tensor is None:
             print("E: Full data tensor not available for latent feature extraction.")
             return

        with torch.no_grad():
            for i in range(0, num_samples, inference_batch_size):
                 batch = all_data_tensor[i:i + inference_batch_size]
                 if not torch.all(torch.isfinite(batch)): continue # Skip bad batches
                 latent_batch = self.autoencoder.encode(batch)
                 if torch.all(torch.isfinite(latent_batch)):
                     all_latent_list.append(latent_batch.cpu().numpy())
                 else: print(f"W: NaN/Inf in latent vectors during centroid calc index {i}")


        if not all_latent_list:
             print("E: No valid latent features extracted for centroids.")
             return

        all_latent_features = np.vstack(all_latent_list)
        labels_array = all_labels_np

        # Filter non-finite latent vectors and corresponding labels
        finite_mask = np.all(np.isfinite(all_latent_features), axis=1)
        if not np.all(finite_mask):
             num_non_finite = np.sum(~finite_mask)
             print(f"W: Filtering {num_non_finite} non-finite latent vectors before centroid calc.")
             all_latent_features = all_latent_features[finite_mask]
             labels_array = labels_array[finite_mask]

        if all_latent_features.shape[0] == 0:
             print("E: No finite latent features remaining after filtering.")
             return

        # Calculate centroids per class
        unique_label_indices = np.unique(labels_array)
        for label_idx in unique_label_indices:
            type_mask = (labels_array == label_idx)
            if not np.any(type_mask): continue # Should not happen if unique_label_indices came from labels_array

            type_features = all_latent_features[type_mask]
            centroid = np.mean(type_features, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)  # 增加L2归一化


            if np.all(np.isfinite(centroid)):
                fault_type = self.idx_to_fault.get(label_idx, f"UnknownLabel_{label_idx}")
                if fault_type == f"UnknownLabel_{label_idx}":
                     print(f"W: Cannot map label index {label_idx} back to fault type name.")
                self.data_semantics[fault_type] = centroid
            else:
                fault_type = self.idx_to_fault.get(label_idx, f"UnknownLabel_{label_idx}")
                print(f"W: Centroid calculation for '{fault_type}' (Label {label_idx}) resulted in non-finite values. Setting zero.")
                self.data_semantics[fault_type] = np.zeros(self.actual_latent_dim, dtype=np.float32)

        print(f"AE training & centroid calculation complete. Found {len(self.data_semantics)} centroids.")


    # extract_data_semantics: Use the trained AE (unchanged conceptually)
    def extract_data_semantics(self, X, fault_labels=None):
        """Extract data semantics using the trained AE (spectral clustering part retained)."""
        if self.autoencoder is None:
            raise ValueError("Autoencoder has not been trained. Call train_autoencoder first.")
        if X is None or X.shape[0] == 0:
             return np.empty((0, self.actual_latent_dim)), {} if fault_labels is not None else np.empty((0, self.actual_latent_dim))

        self.autoencoder.eval()
        all_data_semantics_list = []
        inference_batch_size = AE_BATCH_SIZE * 4

        X_tensor = torch.FloatTensor(X).to(self.device)
        original_indices = np.arange(len(X)) # Keep track of original indices before filtering
        filtered_indices = []

        with torch.no_grad():
            for i in range(0, X_tensor.size(0), inference_batch_size):
                batch_indices = original_indices[i : i + inference_batch_size]
                batch_x = X_tensor[batch_indices] # Index tensor directly

                batch_valid_mask = torch.all(torch.isfinite(batch_x), dim=1) # Check which rows in batch are finite
                if not batch_valid_mask.all():
                     print(f"W: Non-finite data in batch for AE encoding (extract), index {i}. Processing valid rows only.")
                     batch_x = batch_x[batch_valid_mask]
                     batch_indices = batch_indices[batch_valid_mask.cpu().numpy()] # Filter indices

                if batch_x.shape[0] == 0: continue # Skip if batch became empty

                z_batch = self.autoencoder.encode(batch_x)

                batch_z_valid_mask = torch.all(torch.isfinite(z_batch), dim=1)
                if not batch_z_valid_mask.all():
                     print(f"W: Non-finite latent vectors from AE encoder (extract), index {i}. Filtering.")
                     z_batch = z_batch[batch_z_valid_mask]
                     batch_indices = batch_indices[batch_z_valid_mask.cpu().numpy()] # Filter indices again

                if z_batch.shape[0] > 0:
                    all_data_semantics_list.append(z_batch.cpu().numpy())
                    filtered_indices.extend(batch_indices) # Store indices of successful encodings

        if not all_data_semantics_list:
             print("E: No valid data semantics extracted.")
             return np.empty((0, self.actual_latent_dim)), {} if fault_labels is not None else np.empty((0, self.actual_latent_dim))

        data_semantics = np.vstack(all_data_semantics_list)
        # Filter the original labels according to the filtered indices
        if fault_labels is not None:
             fault_labels_filtered = fault_labels[np.array(filtered_indices, dtype=int)]
        else:
             fault_labels_filtered = None

        # --- Prototype Calculation (Spectral Clustering part - applied to current data) ---
        # This part is kept from original model.py
        prototype_semantics = {}
        if fault_labels_filtered is not None:
             unique_faults = np.unique(fault_labels_filtered)
             for fault in unique_faults:
                 indices = np.where(fault_labels_filtered == fault)[0]
                 if len(indices) > 0:
                     fault_type_semantics = data_semantics[indices] # Use filtered semantics
                     if len(fault_type_semantics) > 10:
                         n_clusters = max(2, min(5, len(fault_type_semantics) // 3))
                         try:
                             spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                                                          assign_labels='kmeans', random_state=42,
                                                          n_neighbors=max(5, min(10, len(fault_type_semantics)-1)) )
                             cluster_labels = spectral.fit_predict(fault_type_semantics)

                             if cluster_labels is None or len(cluster_labels) != len(fault_type_semantics):
                                  prototype = np.mean(fault_type_semantics, axis=0)
                             else:
                                  counts = np.bincount(cluster_labels[cluster_labels>=0]) # Ignore potential negative labels
                                  if len(counts) > 0:
                                      largest_cluster = np.argmax(counts)
                                      prototype_idx = np.where(cluster_labels == largest_cluster)[0]
                                      if len(prototype_idx) > 0:
                                          prototype = np.mean(fault_type_semantics[prototype_idx], axis=0)
                                      else: prototype = np.mean(fault_type_semantics, axis=0) # Fallback
                                  else: prototype = np.mean(fault_type_semantics, axis=0) # Fallback
                         except Exception as e:
                             print(f"E: Spectral clustering exception for fault {fault}: {e}. Using mean.")
                             prototype = np.mean(fault_type_semantics, axis=0)
                     else:
                         prototype = np.mean(fault_type_semantics, axis=0)

                     if np.all(np.isfinite(prototype)):
                         fault_name = self.idx_to_fault.get(fault, f"UnknownLabel_{fault}")
                         prototype_semantics[fault_name] = prototype

        if fault_labels_filtered is not None:
            return data_semantics, prototype_semantics # Return filtered semantics & prototypes
        else:
            return data_semantics # Return only filtered semantics


    # synthesize_compound_semantics unchanged
    def synthesize_compound_semantics(self, single_fault_prototypes):
        """使用Dempster-Shafer理论合成复合故障语义，替换max操作"""
        if self.actual_latent_dim <= 0:
            print("E: Cannot synthesize, actual latent dim not set.")
            return {}

        compound_semantics = {}
        compound_combinations = {
            'inner_outer': ['inner', 'outer'],
            'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'],
            'inner_outer_ball': ['inner', 'outer', 'ball']
        }

        def compute_bpa(semantic_vector, epsilon=0.01):
            """从语义向量生成BPA"""
            norm = np.linalg.norm(semantic_vector) + 1e-8
            bpa_focal = semantic_vector / norm if norm > 1e-8 else np.zeros_like(semantic_vector)
            bpa_uncertainty = epsilon / (norm + epsilon)
            bpa_focal = (1 - bpa_uncertainty) * bpa_focal
            return bpa_focal, bpa_uncertainty

        def ds_combination(bpa_list, components):
            """Dempster组合规则合并BPA"""
            combined_bpa = np.zeros(self.actual_latent_dim)
            conflict = 0.0
            for subset in combinations(range(len(bpa_list)), 2):
                bpa1, bpa2 = bpa_list[subset[0]][0], bpa_list[subset[1]][0]
                intersection = np.minimum(bpa1, bpa2)
                combined_bpa += intersection
                conflict += np.sum(np.maximum(bpa1, bpa2) - intersection)
            if conflict < 1.0:
                combined_bpa /= (1.0 - conflict)
            return combined_bpa

        for compound_type, components in compound_combinations.items():
            component_semantics = []
            bpa_list = []
            all_valid = True

            # 收集单一故障语义并生成BPA
            for comp in components:
                proto = single_fault_prototypes.get(comp)
                if proto is not None and np.all(np.isfinite(proto)) and len(proto) == self.actual_latent_dim:
                    component_semantics.append(proto)
                    bpa_focal, bpa_uncertainty = compute_bpa(proto)
                    bpa_list.append((bpa_focal, bpa_uncertainty))
                else:
                    all_valid = False
                    break

            if all_valid and component_semantics:
                # 使用DS理论合并BPA
                synthesized_bpa = ds_combination(bpa_list, components)

                # 映射回语义向量（加权平均）
                weights = [1.0 / len(component_semantics)] * len(component_semantics)  # 简单均等权重
                synthesized = np.average(component_semantics, axis=0, weights=weights)

                # 应用BPA置信度调整
                synthesized = synthesized * np.linalg.norm(synthesized_bpa)

                if np.all(np.isfinite(synthesized)):
                    compound_semantics[compound_type] = synthesized
                else:
                    print(f"W: Synthesized semantics for {compound_type} contains non-finite values.")

        return compound_semantics


# 3. DualChannelCNN (Input dims adjusted)
class DualChannelCNN(nn.Module):
    # Needs input_length=SEGMENT_LENGTH, semantic_dim=AE_LATENT_DIM, num_classes
    def __init__(self, input_length=SEGMENT_LENGTH, semantic_dim=AE_LATENT_DIM, num_classes=8, feature_dim = CNN_FEATURE_DIM):
        super(DualChannelCNN, self).__init__()
        if input_length <= 0 or semantic_dim <= 0 or num_classes <= 0 or feature_dim <= 0:
             raise ValueError("Invalid dimensions provided to DualChannelCNN")
        self.feature_dim = feature_dim
        self.input_length = input_length
        self.semantic_dim = semantic_dim # This is AE Latent Dim

        # --- Channel 1: Signal Processing ---
        c1, c2, c3, c4 = 64, 128, 256, 512
        self.conv1 = nn.Conv1d(1, c1, kernel_size=64, stride=2, padding=31) # Adjust padding slightly if needed
        self.bn1 = nn.BatchNorm1d(c1); self.relu1 = nn.LeakyReLU(0.2); self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=32, stride=1, padding='same', dilation=2)
        self.bn2 = nn.BatchNorm1d(c2); self.relu2 = nn.LeakyReLU(0.2); self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=16, stride=1, padding='same', dilation=4)
        self.bn3 = nn.BatchNorm1d(c3); self.relu3 = nn.LeakyReLU(0.2); self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.conv4 = nn.Conv1d(c3, c4, kernel_size=8, stride=1, padding='same')
        self.bn4 = nn.BatchNorm1d(c4); self.relu4 = nn.LeakyReLU(0.2)
        # Adaptive pool to get fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        self.channel1_flattened = c4 * 8

        # --- Channel 2: Data Semantic Processing (Input: semantic_dim = AE latent dim) ---
        sem_hid1 = 256
        sem_hid2 = 512 # Target intermediate dim
        self.channel2_fc = nn.Sequential(
            nn.Linear(self.semantic_dim, sem_hid1), nn.BatchNorm1d(sem_hid1), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(sem_hid1, sem_hid2), nn.BatchNorm1d(sem_hid2), nn.LeakyReLU(0.2)
        )
        self.channel2_out_dim = sem_hid2

        # --- Channel Attention (on semantic features) ---
        attn_hid = 128
        self.channel_attention = nn.Sequential(
            nn.Linear(self.channel2_out_dim, attn_hid), nn.BatchNorm1d(attn_hid), nn.LeakyReLU(0.2),
            nn.Linear(attn_hid, self.channel2_out_dim), nn.Sigmoid()
        )

        # --- Feature Fusion ---
        fusion_input_dim = self.channel1_flattened + self.channel2_out_dim
        fus_hid1 = 1024
        # Ensure output is the desired feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fus_hid1), nn.BatchNorm1d(fus_hid1), nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.Linear(fus_hid1, self.feature_dim), nn.BatchNorm1d(self.feature_dim), nn.LeakyReLU(0.2), nn.Dropout(0.3),
        )

        # --- Classifier ---
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, semantic=None, return_features=False):
         if x.dim() == 2: x = x.unsqueeze(1)
         if not torch.all(torch.isfinite(x)): x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

         # Channel 1
         c1_out = self.pool1(self.relu1(self.bn1(self.conv1(x))))
         c1_out = self.pool2(self.relu2(self.bn2(self.conv2(c1_out))))
         c1_out = self.pool3(self.relu3(self.bn3(self.conv3(c1_out))))
         c1_out = self.relu4(self.bn4(self.conv4(c1_out)))
         c1_out = self.adaptive_pool(c1_out)
         channel1_out = c1_out.reshape(c1_out.size(0), -1)

         if not torch.all(torch.isfinite(channel1_out)):
             print("W: CNN Ch1 output non-finite. Clamping.")
             channel1_out = torch.nan_to_num(channel1_out, nan=0.0)

         # If no semantic provided (used during SEN training / ZSL eval)
         if semantic is None:
             # Simple projection from C1 to feature_dim if needed for consistent output?
             # This path should primarily return features of self.feature_dim if 'return_features' is True.
             # Let's just return the channel1 features, and expect the caller to handle projection if needed.
             # For classification: Add a simple classifier head?
             if return_features:
                 # How to return C1 features of the right dimension? Need projection.
                 # Create a dedicated projection if needed, or assume fusion handles it?
                 # For simplicity, let's assume this path isn't used for final logits.
                 # It *is* used in SEN training and ZSL eval where we need features.
                 # We need channel1_out mapped to feature_dim. Let's add a quick FC layer.
                 if not hasattr(self, 'c1_to_feature'):
                      self.c1_to_feature = nn.Linear(self.channel1_flattened, self.feature_dim).to(x.device)
                 features_c1 = self.c1_to_feature(channel1_out)
                 features_c1 = torch.nan_to_num(features_c1, nan=0.0) # Clamp output
                 return features_c1 # Just return features of expected dim

             else: # If we absolutely need logits without semantics...
                  # Use the C1-only features to classify
                   if not hasattr(self, 'c1_to_feature'):
                       self.c1_to_feature = nn.Linear(self.channel1_flattened, self.feature_dim).to(x.device)
                   features_c1 = self.c1_to_feature(channel1_out)
                   logits_c1 = self.classifier(F.leaky_relu(features_c1)) # Apply activation?
                   features_c1 = torch.nan_to_num(features_c1, nan=0.0)
                   logits_c1 = torch.nan_to_num(logits_c1, nan=0.0)
                   return logits_c1, features_c1


         # --- Semantic provided path ---
         if not torch.all(torch.isfinite(semantic)):
             print("W: CNN semantic input non-finite. Clamping.")
             semantic = torch.nan_to_num(semantic, nan=0.0)
         if semantic.dim() != 2 or semantic.shape[1] != self.semantic_dim:
              raise ValueError(f"CNN Semantic input wrong shape. Expected [B, {self.semantic_dim}], got {semantic.shape}")


         # Channel 2
         channel2_out = self.channel2_fc(semantic)
         if not torch.all(torch.isfinite(channel2_out)):
             channel2_out = torch.nan_to_num(channel2_out, nan=0.0)

         # Attention
         channel_attn = self.channel_attention(channel2_out)
         if not torch.all(torch.isfinite(channel_attn)): channel_attn = torch.full_like(channel2_out, 0.5)
         attended_semantic = channel2_out * channel_attn

         # Fusion
         concatenated = torch.cat([channel1_out, attended_semantic], dim=1)
         if not torch.all(torch.isfinite(concatenated)): concatenated = torch.nan_to_num(concatenated, nan=0.0)

         fusion_out = self.fusion(concatenated) # Output is final feature vector
         if not torch.all(torch.isfinite(fusion_out)): fusion_out = torch.nan_to_num(fusion_out, nan=0.0)

         if return_features:
             return fusion_out # Return the final fused features

         # Classification
         logits = self.classifier(fusion_out)
         if not torch.all(torch.isfinite(logits)): logits = torch.nan_to_num(logits, nan=0.0)

         return logits, fusion_out # Return logits and final features


# 4. SemanticEmbeddingNetwork (Input/Output dims adjusted)
class SemanticEmbeddingNetwork(nn.Module):
    # Input: semantic_dim (FUSED = knowledge + AE data)
    # Output: feature_dim (CNN Feature Dim)
    def __init__(self, semantic_dim, feature_dim=CNN_FEATURE_DIM):
        super(SemanticEmbeddingNetwork, self).__init__()
        if semantic_dim <= 0 or feature_dim <= 0:
             raise ValueError("Invalid dimensions for SEN")
        self.input_dim = semantic_dim
        self.output_dim = feature_dim

        # Dynamic hidden layers based on input/output dims
        hid1 = max(feature_dim * 2, (semantic_dim + feature_dim) // 2)
        hid1 = min(hid1, 512) # Cap hidden size
        hid2 = max(feature_dim, (hid1 + feature_dim) // 2)
        hid2 = min(hid2, max(feature_dim, 256)) # Cap size
        hid1 = max(hid1, hid2) # Ensure h1>=h2
        hid2 = max(hid2, feature_dim) # Ensure h2>=feature_dim

        print(f"SEN Arch: {semantic_dim} -> {hid1} -> {hid2} -> {feature_dim}")

        self.embedding = nn.Sequential(
            nn.Linear(semantic_dim, hid1), nn.BatchNorm1d(hid1), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hid1, hid2), nn.BatchNorm1d(hid2), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(hid2, feature_dim)
        )
        # Residual connection
        self.residual = nn.Linear(semantic_dim, feature_dim) if semantic_dim != feature_dim else nn.Identity()

    def forward(self, semantic):
         if not torch.all(torch.isfinite(semantic)): semantic = torch.nan_to_num(semantic, nan=0.0)
         embedded = self.embedding(semantic)
         residual = self.residual(semantic)
         if not torch.all(torch.isfinite(embedded)): embedded = torch.zeros_like(embedded)
         if not torch.all(torch.isfinite(residual)): residual = torch.zeros_like(residual)
         output = embedded + residual
         if not torch.all(torch.isfinite(output)): output = torch.nan_to_num(output, nan=0.0)
         return output


# 5. Loss Functions (ContrastiveLoss, FeatureSemanticConsistencyLoss) - Kept as in V1 (Aligned with original model.py)
class ContrastiveLoss(nn.Module):
    """ For CNN Training - Encourages features of same class to be closer """
    def __init__(self, temperature=0.1): # Adjusted default temp based on prev run
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        if not torch.all(torch.isfinite(features)): return torch.tensor(0.0, device=features.device)
        batch_size = features.size(0)
        if batch_size < 2: return torch.tensor(0.0, device=features.device)

        features = F.normalize(features, p=2, dim=1)
        if torch.isnan(features).any(): features = torch.nan_to_num(features, nan=0.0)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, min=-30.0, max=30.0)

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        identity_mask = torch.eye(batch_size, device=features.device)
        mask = mask - identity_mask # Positive pairs mask (exclude self)

        # InfoNCE loss formulation
        exp_logits = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) - exp_logits.diag().unsqueeze(1) + 1e-8) # Denominator excludes self (k!=i)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        # Handle cases where a sample might not have any positive pairs
        valid_loss_terms = mean_log_prob_pos[mask.sum(dim=1) > 0] # Only average over anchors with positive pairs
        if len(valid_loss_terms) > 0:
             loss = -valid_loss_terms.mean()
        else: # If no positive pairs in the batch (e.g., all unique labels)
             loss = torch.tensor(0.0, device=features.device)


        if torch.isnan(loss) or torch.isinf(loss): return torch.tensor(0.0, device=features.device)
        return loss


class FeatureSemanticConsistencyLoss(nn.Module):
    """ For CNN Training - Aligns CNN features with projected *data* semantics """

    def __init__(self, beta=0.01, semantic_dim_input=AE_LATENT_DIM, feature_dim_output=CNN_FEATURE_DIM,
                 temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        # Create the projection layer immediately
        self.projection = nn.Linear(semantic_dim_input, feature_dim_output)
        # Optional: Initialize projection weights
        try:
             nn.init.xavier_uniform_(self.projection.weight)
             if self.projection.bias is not None: nn.init.zeros_(self.projection.bias)
        except: pass # Ignore init errors silently
        print(f"Initialized ConsistencyLoss projection: {semantic_dim_input} -> {feature_dim_output}")


    def forward(self, features, data_semantics): # Input is *data* semantics
         # Input Checks
         if not torch.all(torch.isfinite(features)): return torch.tensor(0.0, device=features.device)
         if not torch.all(torch.isfinite(data_semantics)): return torch.tensor(0.0, device=features.device)
         if features.shape[0] != data_semantics.shape[0]: return torch.tensor(0.0, device=features.device)
         if data_semantics.shape[1] != self.projection.in_features: # Check input dim
              print(f"E: ConsistencyLoss dim mismatch! Expected {self.projection.in_features}, got {data_semantics.shape[1]}")
              return torch.tensor(0.0, device=features.device)

         # Ensure projection layer is on the correct device
         self.projection.to(features.device)

         # Project data semantics to feature space
         projected_data_semantics = self.projection(data_semantics)

         if not torch.all(torch.isfinite(projected_data_semantics)):
              print("W: ConsistencyLoss projection output non-finite. Using zeros.")
              projected_data_semantics = torch.zeros_like(projected_data_semantics)


         # Normalize vectors
         features_norm = F.normalize(features, p=2, dim=1)
         projected_data_semantics_norm = F.normalize(projected_data_semantics, p=2, dim=1)
         if torch.isnan(features_norm).any(): features_norm = torch.nan_to_num(features_norm, nan=0.0)
         if torch.isnan(projected_data_semantics_norm).any(): projected_data_semantics_norm = torch.nan_to_num(projected_data_semantics_norm, nan=0.0)

         # Cosine Similarity Loss
         similarity = torch.sum(features_norm * projected_data_semantics_norm, dim=1) / self.temperature  # 加入温度缩放
         consistency_loss = 1.0 - torch.mean(similarity)

         # L2 Regularization on features
         l2_reg = torch.mean(torch.norm(features, p=2, dim=1)**2) if self.beta > 0 else 0.0
         loss = consistency_loss + self.beta * l2_reg

         if torch.isnan(loss) or torch.isinf(loss): return torch.tensor(0.0, device=features.device)

         return loss
class SemanticRelationContrastiveLoss(nn.Module):
    """考虑语义关系的对比损失（仅使用单一故障数据）"""

    def __init__(self, temperature=0.07, fault_relationships=None):
        super().__init__()
        self.temperature = temperature
        self.fault_relationships = fault_relationships or {}

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # 计算所有样本对之间的余弦相似度
        features_norm = nn.functional.normalize(features, p=2, dim=1)
        if torch.isnan(features_norm).any():
            features_norm = torch.nan_to_num(features_norm, nan=0.0)

        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        similarity_matrix = torch.clamp(similarity_matrix, min=-30.0, max=30.0)

        # 创建标签匹配矩阵：如果两个样本有相同标签则为1，否则为0
        labels_expand_row = labels.unsqueeze(1).expand(batch_size, batch_size)
        labels_expand_col = labels.unsqueeze(0).expand(batch_size, batch_size)
        mask_same_class = (labels_expand_row == labels_expand_col).float()

        # 创建不是同一样本的掩码
        mask_not_same = 1 - torch.eye(batch_size, device=device)

        # 构建语义关系权重矩阵
        # 不使用合成数据，而是利用先验知识调整相似度权重
        relation_weights = torch.ones_like(similarity_matrix)

        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and labels[i] != labels[j]:
                    # 获取故障类型名称
                    fault_i = self.fault_relationships.get('idx_to_fault', {}).get(labels[i].item())
                    fault_j = self.fault_relationships.get('idx_to_fault', {}).get(labels[j].item())

                    if fault_i and fault_j:
                        # 检查是否存在潜在的复合关系
                        if self._check_potential_compound(fault_i, fault_j):
                            # 如果两个故障可能共同出现在复合故障中，减少它们之间的排斥
                            relation_weights[i, j] = 0.5  # 降低负样本权重

        # 正样本掩码（同类但不同样本）
        mask_positives = mask_same_class * mask_not_same

        # 对比损失计算
        exp_logits = torch.exp(similarity_matrix) * mask_not_same
        # 应用关系权重到负样本
        exp_logits = exp_logits * (mask_positives + (1 - mask_positives) * relation_weights)

        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 计算正样本对的平均对数概率
        mean_log_prob = (mask_positives * log_prob).sum(1) / (mask_positives.sum(1) + 1e-12)

        # 处理没有正样本的情况
        valid_terms = mean_log_prob[mask_positives.sum(1) > 0]
        if len(valid_terms) == 0:
            return torch.tensor(0.0, device=device)

        loss = -valid_terms.mean()

        # 最后检查以防NaN
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=device)

        return loss

    def _check_potential_compound(self, fault1, fault2):
        """检查两种单一故障是否可能共同出现在复合故障中"""
        # 使用先验知识而非合成数据
        common_combinations = {
            ('inner', 'outer'), ('inner', 'ball'), ('outer', 'ball'),
            ('outer', 'inner'), ('ball', 'inner'), ('ball', 'outer')
        }
        return (fault1, fault2) in common_combinations


def balanced_clustering_loss(features, labels, device, model, margin=1.0):
        """
        改进的聚类损失，保持故障之间的关系结构

        参数:
            features: 特征向量 [batch_size, feature_dim]
            labels: 类别标签 [batch_size]
            device: 计算设备
            model: 模型对象，用于访问idx_to_fault映射
            margin: 类间距离的最小值

        返回:
            聚类损失值
        """
        batch_size = features.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)

        # 计算特征的类中心
        unique_labels = torch.unique(labels)
        centers = {}

        for label in unique_labels:
            mask = (labels == label)
            if torch.sum(mask) > 0:
                centers[label.item()] = torch.mean(features[mask], dim=0)

        # 计算类内损失（到中心的距离）
        intra_loss = torch.tensor(0.0, device=device)
        num_intra = 0

        for label in unique_labels:
            mask = (labels == label)
            if torch.sum(mask) <= 1:
                continue

            center = centers[label.item()]
            dist_to_center = torch.sum((features[mask] - center.unsqueeze(0)) ** 2, dim=1)
            intra_loss += torch.mean(dist_to_center)
            num_intra += 1

        if num_intra > 0:
            intra_loss = intra_loss / num_intra

        # 检查故障类型之间的关系
        def _check_fault_relation(fault1, fault2):
            """检查两种故障之间是否有语义关联"""
            if fault1 is None or fault2 is None:
                return False

            # 如果是相同故障，返回True
            if fault1 == fault2:
                return True

            # 分解复合故障名称
            components1 = fault1.split('_')
            components2 = fault2.split('_')

            # 计算组件重叠度
            common = set(components1) & set(components2)
            return len(common) > 0

        # 计算类间损失（中心之间的距离）
        inter_loss = torch.tensor(0.0, device=device)
        num_inter = 0

        center_labels = list(centers.keys())
        for i in range(len(center_labels)):
            for j in range(i + 1, len(center_labels)):
                label_i = center_labels[i]
                label_j = center_labels[j]

                # 获取故障类型名称
                fault_i = model.idx_to_fault.get(label_i, f"Unknown_{label_i}")
                fault_j = model.idx_to_fault.get(label_j, f"Unknown_{label_j}")

                # 检查两个故障是否有语义关联
                related = _check_fault_relation(fault_i, fault_j)

                center_i = centers[label_i]
                center_j = centers[label_j]

                # 计算中心之间的距离
                dist = torch.sum((center_i - center_j) ** 2)

                # 对于相关故障，使用较小的margin；对于不相关故障，使用更大的margin
                actual_margin = margin * (0.5 if related else 1.5)

                # 如果距离小于margin，产生损失
                margin_loss = torch.max(torch.tensor(0.0, device=device), actual_margin - dist)

                # 如果两个故障相关，给予较小的惩罚权重
                weight = 0.5 if related else 1.0

                inter_loss += weight * margin_loss
                num_inter += 1

        if num_inter > 0:
            inter_loss = inter_loss / num_inter

        # 平衡类内和类间损失
        # 类内权重更大以确保同类特征聚集，但不要过分惩罚相关故障类型
        intra_weight = 0.7
        inter_weight = 0.3

        total_loss = intra_weight * intra_loss + inter_weight * inter_loss

        return total_loss

# 6. 主程序：模型训练和评估 (Pipeline using the modified components)
class ZeroShotCompoundFaultDiagnosis:
    def __init__(self, data_path, sample_length=SEGMENT_LENGTH, latent_dim=AE_LATENT_DIM, batch_size=DEFAULT_BATCH_SIZE):
        self.data_path = data_path
        self.sample_length = sample_length
        self.latent_dim_config = latent_dim
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize components
        self.preprocessor = DataPreprocessor(sample_length=self.sample_length)
        self.semantic_builder = FaultSemanticBuilder(latent_dim=self.latent_dim_config)

        # Fault types and mapping
        self.fault_types = {
            'normal': 0, 'inner': 1, 'outer': 2, 'ball': 3,
            'inner_outer': 4, 'inner_ball': 5, 'outer_ball': 6, 'inner_outer_ball': 7
        }
        self.idx_to_fault = {v: k for k, v in self.fault_types.items()}
        self.semantic_builder.idx_to_fault = self.idx_to_fault # Provide mapping to builder
        self.compound_fault_types = ['inner_outer', 'inner_ball', 'outer_ball', 'inner_outer_ball']
        self.num_classes = len(self.fault_types)

        # Models - initialized later
        self.cnn_model = None
        self.embedding_net = None
        # Need placeholders for actual dims determined during pipeline
        self.actual_latent_dim = -1
        self.cnn_feature_dim = -1
        self.fused_semantic_dim = -1
        # Loss for CNN training needs careful handling of its projection layer params
        self.consistency_loss_fn = None



    def joint_train_cnn_sen(self, data_dict, semantic_dict, epochs=40, lr=0.0005):
        """联合训练CNN和SEN，考虑语义关系以改进零样本学习性能"""
        print("开始联合训练CNN和SEN...")

        # 1. 准备数据和语义
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']

        data_only_semantics = semantic_dict.get('data_only_semantics')
        fused_semantics = semantic_dict.get('fused_semantics')

        if not data_only_semantics or not fused_semantics:
            print("错误: 语义数据缺失，无法进行联合训练")
            return

        # 2. 创建数据加载器
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if len(X_val) > 0:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)

        # 3. 初始化模型
        if self.cnn_model is None:
            self.cnn_model = DualChannelCNN(
                input_length=self.sample_length,
                semantic_dim=self.actual_latent_dim,
                num_classes=len(set(y_train)),
                feature_dim=CNN_FEATURE_DIM
            ).to(self.device)
            self.cnn_feature_dim = self.cnn_model.feature_dim

        if self.embedding_net is None:
            self.embedding_net = SemanticEmbeddingNetwork(
                semantic_dim=self.fused_semantic_dim,
                feature_dim=self.cnn_feature_dim
            ).to(self.device)

        # 4. 初始化损失函数
        criterion_ce = nn.CrossEntropyLoss()
        criterion_contrastive = SemanticRelationContrastiveLoss(
            temperature=0.1,
            fault_relationships={'idx_to_fault': self.idx_to_fault}
        ).to(self.device)
        criterion_mse = nn.MSELoss()
        criterion_cosine = nn.CosineEmbeddingLoss(margin=0.2)

        # 5. 准备语义向量字典
        semantic_vectors_cnn = {
            idx: data_only_semantics.get(name, np.zeros(self.actual_latent_dim, dtype=np.float32))
            for name, idx in self.fault_types.items()
        }
        fused_vectors = {
            idx: fused_semantics.get(name, np.zeros(self.fused_semantic_dim, dtype=np.float32))
            for name, idx in self.fault_types.items()
        }

        # 6. 优化器
        cnn_params = list(self.cnn_model.parameters())
        sen_params = list(self.embedding_net.parameters())
        all_params = cnn_params + sen_params

        optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

        # 7. 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        print(f"开始联合训练 ({epochs} 轮)...")
        for epoch in range(epochs):
            self.cnn_model.train()
            self.embedding_net.train()

            total_loss = 0.0
            ce_loss_sum = 0.0
            contr_loss_sum = 0.0
            cluster_loss_sum = 0.0
            reverse_mse_loss_sum = 0.0
            correct = 0
            total = 0

            for inputs, batch_y in train_loader:
                inputs, batch_y = inputs.to(self.device), batch_y.to(self.device)
                batch_size = inputs.size(0)

                if batch_size <= 1:
                    continue

                # 获取数据语义
                batch_semantics = torch.stack(
                    [torch.from_numpy(semantic_vectors_cnn[lbl.item()])
                     for lbl in batch_y]
                ).to(self.device)

                # 获取融合语义
                batch_fused_semantics = torch.stack(
                    [torch.from_numpy(fused_vectors[lbl.item()])
                     for lbl in batch_y]
                ).to(self.device)

                optimizer.zero_grad()

                # 1. CNN 前向传播
                logits, cnn_features = self.cnn_model(inputs, batch_semantics)

                # 2. SEN 前向传播
                sen_features = self.embedding_net(batch_fused_semantics)

                # 3. 计算损失
                # 3.1 分类损失
                ce_loss = criterion_ce(logits, batch_y)

                # 3.2 对比损失（基于语义关系的改进版）
                contr_loss = criterion_contrastive(cnn_features, batch_y)

                # 3.3 平衡聚类损失（考虑故障关系）
                cluster_loss = balanced_clustering_loss(cnn_features, batch_y, self.device, self, margin=1.0)

                # 3.4 反向投影损失（CNN特征与SEN特征对齐）
                reverse_mse_loss = criterion_mse(cnn_features, sen_features)

                # 3.5 总损失
                cnn_loss = ce_loss + 0.1 * contr_loss + 0.2 * cluster_loss + 0.1 * reverse_mse_loss

                # 反向传播和优化
                cnn_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

                # 统计
                total_loss += cnn_loss.item() * batch_size
                ce_loss_sum += ce_loss.item() * batch_size
                contr_loss_sum += contr_loss.item() * batch_size
                cluster_loss_sum += cluster_loss.item() * batch_size
                reverse_mse_loss_sum += reverse_mse_loss.item() * batch_size

                _, predicted = torch.max(logits, 1)
                total += batch_size
                correct += (predicted == batch_y).sum().item()

            # 更新学习率
            scheduler.step()

            # 计算训练指标
            avg_loss = total_loss / total if total > 0 else 0
            avg_ce_loss = ce_loss_sum / total if total > 0 else 0
            avg_contr_loss = contr_loss_sum / total if total > 0 else 0
            avg_cluster_loss = cluster_loss_sum / total if total > 0 else 0
            avg_reverse_loss = reverse_mse_loss_sum / total if total > 0 else 0
            accuracy = 100.0 * correct / total if total > 0 else 0

            # 验证
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                self.cnn_model.eval()
                self.embedding_net.eval()
                val_correct, val_total = 0, 0

                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x, val_y = val_x.to(self.device), val_y.to(self.device)
                        val_semantics = torch.stack(
                            [torch.from_numpy(semantic_vectors_cnn[lbl.item()])
                             for lbl in val_y]
                        ).to(self.device)

                        val_logits, _ = self.cnn_model(val_x, val_semantics)
                        val_loss += criterion_ce(val_logits, val_y).item() * val_y.size(0)

                        _, val_pred = torch.max(val_logits, 1)
                        val_total += val_y.size(0)
                        val_correct += (val_pred == val_y).sum().item()

                val_loss = val_loss / val_total if val_total > 0 else 0
                val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0

            print(
                f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f} (CNN: {avg_ce_loss:.4f}, Contr: {avg_contr_loss:.4f}, Clust: {avg_cluster_loss:.4f}, Rev: {avg_reverse_loss:.4f}) Acc: {accuracy:.2f}%")

            if val_loader:
                print(f"  Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        'cnn_model': self.cnn_model.state_dict(),
                        'embedding_net': self.embedding_net.state_dict(),
                        'epoch': epoch + 1,
                        'val_acc': val_acc,
                        'val_loss': val_loss
                    }, 'best_joint_model.pth')
                    print(f"  模型保存！当前最佳平均损失: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"提前停止训练，{patience}轮内验证损失未改善")
                        break

        # 加载最佳模型
        if os.path.exists('best_joint_model.pth'):
            checkpoint = torch.load('best_joint_model.pth')
            self.cnn_model.load_state_dict(checkpoint['cnn_model'])
            self.embedding_net.load_state_dict(checkpoint['embedding_net'])
            print(f"加载最佳联合模型（轮次：{checkpoint['epoch']}，验证准确率：{checkpoint['val_acc']:.2f}%）")

        self.cnn_model.eval()
        self.embedding_net.eval()
        print("联合训练完成！")
    # visualize_data_semantics_distribution (Calls AE extraction) - Modified for robustness in V1
    def visualize_data_semantics_distribution(self, data_dict):
        """Visualize the distribution of data semantics using t-SNE (AE output)."""
        print("Visualizing data semantics distribution (using AE output)...")
        if self.semantic_builder.autoencoder is None:
            print("W: Autoencoder not trained. Cannot visualize.")
            return
        if not hasattr(self.semantic_builder, 'actual_latent_dim') or self.semantic_builder.actual_latent_dim <= 0:
             print("W: AE Latent dim not set. Cannot visualize.")
             return

        # Extract features *from training data* for visualization
        X_vis = data_dict.get('X_train')
        y_vis = data_dict.get('y_train')
        if X_vis is None or y_vis is None or len(X_vis)==0:
            print("W: Training data not available for visualization.")
            return

        print(f"  Extracting semantics from {len(X_vis)} training samples...")
        vis_data_semantics, _ = self.semantic_builder.extract_data_semantics(X_vis, y_vis) # Extract current semantics

        if vis_data_semantics is None or vis_data_semantics.shape[0] == 0:
            print("E: Failed to extract data semantics for visualization.")
            return
        print(f"  Extracted {vis_data_semantics.shape[0]} semantic vectors for visualization.")
        # Use labels corresponding to the *successfully extracted* semantics
        # If extract_data_semantics filtered samples, we need the filtered labels.
        # Let's assume extract_data_semantics returns correctly aligned semantics and labels were handled internally.
        # For simplicity, assume y_vis corresponds to vis_data_semantics rows now. Need to check extract_data_semantics return.
        # If dimensions mismatch, plot without labels for safety.
        if len(y_vis) != vis_data_semantics.shape[0]:
             print(f"W: Label mismatch after extraction. Plotting t-SNE without distinct labels.")
             vis_labels = np.zeros(vis_data_semantics.shape[0]) # Assign dummy label 0
        else:
             vis_labels = y_vis


        # t-SNE
        tsne_start_time = time.time()
        perplexity_val = min(30.0, max(5.0, vis_data_semantics.shape[0] / 5.0 - 1))
        n_components = 2
        if vis_data_semantics.shape[0] <= n_components:
            print(f"E: Not enough samples ({vis_data_semantics.shape[0]}) for t-SNE.")
            return

        try:
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity_val,
                        learning_rate='auto', init='pca', n_iter=1000, metric='cosine')
            semantics_2d = tsne.fit_transform(vis_data_semantics)
            if not np.all(np.isfinite(semantics_2d)): raise ValueError("t-SNE resulted in non-finite values")
            print(f"t-SNE completed in {time.time() - tsne_start_time:.2f}s.")
        except Exception as e_tsne:
             print(f"E: t-SNE failed: {e_tsne}")
             return

        # Plotting
        fault_labels_str = [self.idx_to_fault.get(label, f"Unknown_{int(label)}") for label in vis_labels]
        df = pd.DataFrame({'x': semantics_2d[:, 0], 'y': semantics_2d[:, 1], 'fault_type': fault_labels_str})
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df, x='x', y='y', hue='fault_type', palette='viridis', s=50, alpha=0.6)
        plt.title('t-SNE Visualization of AE Data Semantics')
        plt.xlabel('t-SNE Dimension 1'); plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Fault Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig('data_semantics_distribution_ae.png'); plt.close()
        print("Data semantics distribution saved to 'data_semantics_distribution_ae.png'")


    # load_data (uses the modified preprocessor)
    def load_data(self):
        """Loads and preprocesses data using the modified DataPreprocessor."""
        print("Loading and preprocessing data...")
        all_processed_data = {}
        all_labels_list = []
        all_features_list = []
        single_fault_keys = ['normal', 'inner', 'outer', 'ball']
        fault_files = { # Assuming .mat files based on original
             'normal': 'normal.mat', 'inner': 'inner.mat', 'outer': 'outer.mat', 'ball': 'ball.mat',
             'inner_outer': 'inner_outer.mat', 'inner_ball': 'inner_ball.mat',
             'outer_ball': 'outer_ball.mat', 'inner_outer_ball': 'inner_outer_ball.mat'
        }

        for fault_type, file_name in fault_files.items():
             file_path = os.path.join(self.data_path, file_name)
             print(f"Processing {fault_type} from {file_name}...")
             if not os.path.exists(file_path):
                 print(f"W: File not found: {file_path}. Skipping."); continue

             is_single_fault = fault_type in single_fault_keys
             use_augmentation = is_single_fault # Augment only single faults

             try:
                 mat_data = sio.loadmat(file_path)
                 signal_data_raw = None
                 # Simplified key finding logic
                 potential_keys = [k for k in mat_data if not k.startswith('__') and isinstance(mat_data[k], np.ndarray) and mat_data[k].size > 500]
                 if potential_keys:
                    signal_data_raw = mat_data[potential_keys[-1]] # Often the last large array
                 if signal_data_raw is None:
                    print(f"E: No suitable data in {file_name}. Skip."); continue

                 signal_data_flat = signal_data_raw.ravel().astype(np.float64)
                 max_len = 300000 # Increased max length slightly
                 if len(signal_data_flat) > max_len: signal_data_flat = signal_data_flat[:max_len]

                 # Use the MODIFIED preprocessor -> returns segments
                 processed_segments = self.preprocessor.preprocess(signal_data_flat, augmentation=use_augmentation)

                 if processed_segments is None or processed_segments.shape[0] == 0:
                     print(f"W: Preprocessing yielded no segments for {fault_type}."); continue

                 num_segments = len(processed_segments)
                 label_idx = self.fault_types[fault_type]
                 all_features_list.append(processed_segments)
                 all_labels_list.extend([label_idx] * num_segments)
                 # Store processed segments keyed by name if needed later
                 # all_processed_data[fault_type] = processed_segments
                 print(f"  Processed {fault_type}: {num_segments} segments.")

             except Exception as e: print(f"E: Error processing {file_name}: {e}"); traceback.print_exc()

        if not all_features_list: print("E: No features generated."); return None

        all_features_np = np.vstack(all_features_list); all_labels_np = np.array(all_labels_list)
        single_mask = np.isin(all_labels_np, [self.fault_types[k] for k in single_fault_keys])
        compound_mask = ~single_mask
        X_single, y_single = all_features_np[single_mask], all_labels_np[single_mask]
        X_compound, y_compound = all_features_np[compound_mask], all_labels_np[compound_mask]

        X_train, X_val, y_train, y_val = [], [], [], []
        if len(X_single) > 0:
            try: # Use stratification if possible
                X_train, X_val, y_train, y_val = train_test_split(X_single, y_single, test_size=0.3, random_state=42, stratify=y_single)
            except ValueError: # Fallback if stratification fails
                print("W: Stratification failed, splitting without it.")
                X_train, X_val, y_train, y_val = train_test_split(X_single, y_single, test_size=0.3, random_state=42)
            print(f"Train/Val split: {len(X_train)} train, {len(X_val)} val samples.")
        else: print("E: No single fault data for training."); return None

        return {
            'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val,
            'X_test': X_compound, 'y_test': y_compound
            # 'all_processed_data': all_processed_data # Optional: return raw segments
        }

    # build_semantics (uses the modified AE training)
    def build_semantics(self, data_dict):
        """Builds knowledge and data semantics (AE training aligned with gen.py)."""
        print("Building fault semantics...")
        # 1. Knowledge Semantics (unchanged)
        knowledge_semantics = self.semantic_builder.build_knowledge_semantics()
        if not knowledge_semantics: print("E: Failed build knowledge semantics."); return None
        print(f"  Knowledge semantics built. Dimension: {self.semantic_builder.knowledge_dim}")

        # 2. Train AE for Data Semantics (using aligned logic)
        print("  Training autoencoder for data semantics...")
        X_train_ae = data_dict.get('X_train')
        y_train_ae = data_dict.get('y_train')
        if X_train_ae is None or y_train_ae is None or len(X_train_ae) < AE_BATCH_SIZE:
             print("E: Not enough training data for AE."); return None

        try: # Train AE - populates semantic_builder.data_semantics
            self.semantic_builder.train_autoencoder( X_train_ae, labels=y_train_ae, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, lr=AE_LR)
        except Exception as e: print(f"E: AE training failed: {e}"); traceback.print_exc(); return None

        # Store actual latent dim and check if semantics were created
        self.actual_latent_dim = self.semantic_builder.actual_latent_dim
        single_fault_prototypes = self.semantic_builder.data_semantics
        if not single_fault_prototypes: print("E: AE training produced no data semantics."); return None
        print(f"  Data semantic prototypes learned. Dimension: {self.actual_latent_dim}")

        # 3. Synthesize Compound Data Semantics (unchanged)
        compound_data_semantics = self.semantic_builder.synthesize_compound_semantics(single_fault_prototypes)
        print(f"  Compound data semantics synthesized for {len(compound_data_semantics)} types.")

        # 4. Prepare Output Dictionaries (data_only, fused)
        data_only_semantics = {**single_fault_prototypes, **compound_data_semantics}
        # Fused semantics calculation
        fused_semantics = {}
        self.fused_semantic_dim = self.semantic_builder.knowledge_dim + self.actual_latent_dim
        for ft, k_vec in knowledge_semantics.items():
             d_vec = data_only_semantics.get(ft) # Get corresponding data semantic
             if d_vec is not None and np.all(np.isfinite(k_vec)) and np.all(np.isfinite(d_vec)) and \
                len(k_vec)==self.semantic_builder.knowledge_dim and len(d_vec)==self.actual_latent_dim:
                  fused_vec = np.concatenate([k_vec, d_vec]).astype(np.float32)
                  if np.all(np.isfinite(fused_vec)) and len(fused_vec) == self.fused_semantic_dim:
                       fused_semantics[ft] = fused_vec
        print(f"  Fused semantics prepared for {len(fused_semantics)} types. Dimension: {self.fused_semantic_dim}")

        # Check if critical dimensions are set
        if self.actual_latent_dim <=0 or self.fused_semantic_dim <= 0:
             print("E: Invalid semantic dimensions calculated. Aborting.")
             return None

        return {
            'knowledge_semantics': knowledge_semantics, 'data_prototypes': single_fault_prototypes,
            'compound_data_semantics': compound_data_semantics, 'data_only_semantics': data_only_semantics,
            'fused_semantics': fused_semantics
        }


    # visualize_semantics unchanged

    def visualize_semantics(self, semantic_dict):
        """Visualizes semantic similarity matrices."""
        print("Visualizing semantic similarity matrices...")
        def compute_similarity_matrix(semantics_dict):
            fault_types = []
            vectors = []
            if not semantics_dict: return None, [] # Handle empty dict
            # Determine expected dimension from first valid vector
            first_valid_vec = next((v for v in semantics_dict.values() if v is not None and np.all(np.isfinite(v))), None)
            if first_valid_vec is None: return None, [] # No valid vectors
            expected_dim = len(first_valid_vec)

            for ft, vec in semantics_dict.items():
                 if vec is not None and np.all(np.isfinite(vec)) and len(vec)==expected_dim:
                     norm = np.linalg.norm(vec)
                     if norm > 1e-9: vectors.append(vec / norm); fault_types.append(ft)
                 # else: print(f"W: Skipping invalid/dim-mismatched vector for '{ft}' in sim matrix.") # Optional warning

            if not vectors: return None, []
            n = len(vectors)
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n): # Optimized calculation
                    sim = np.dot(vectors[i], vectors[j])
                    sim_matrix[i, j] = sim; sim_matrix[j, i] = sim # Symmetric matrix
            return sim_matrix, fault_types

        plt.rc('font', size=8) # Smaller font for annotations

        # Data-Only Semantics Plot
        d_sim_mat, d_labels = compute_similarity_matrix(semantic_dict.get('data_only_semantics'))
        if d_sim_mat is not None:
            plt.figure(figsize=(9, 7)); sns.heatmap(d_sim_mat, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=d_labels, yticklabels=d_labels, annot_kws={"size": 7})
            plt.title('Data-Only Semantics Similarity (AE Output)'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            plt.tight_layout(); plt.savefig('data_only_semantics_similarity.png'); plt.close(); print("Data-only sim matrix saved.")
        else: print("W: Could not compute/plot data-only similarity matrix.")

        # Fused Semantics Plot
        f_sim_mat, f_labels = compute_similarity_matrix(semantic_dict.get('fused_semantics'))
        if f_sim_mat is not None:
            plt.figure(figsize=(9, 7)); sns.heatmap(f_sim_mat, annot=True, fmt=".2f", cmap="viridis", xticklabels=f_labels, yticklabels=f_labels, annot_kws={"size": 7})
            plt.title('Fused Semantics Similarity (Knowledge + AE Data)'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            plt.tight_layout(); plt.savefig('fused_semantics_similarity.png'); plt.close(); print("Fused sim matrix saved.")
        else: print("W: Could not compute/plot fused similarity matrix.")
        plt.rcdefaults() # Reset font size


    # train_dual_channel_cnn (updated initialization of consistency loss)
    def train_dual_channel_cnn(self, data_dict, semantic_dict, epochs=CNN_EPOCHS, lr=CNN_LR):
        print("Training dual channel CNN model...")
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']
        if len(X_train) == 0:
            print("E: No training data for CNN.");
            return
        has_val_data = len(X_val) > 0
        data_only_semantics = semantic_dict.get('data_only_semantics')
        if not data_only_semantics:
            print("E: Data-only semantics missing.");
            return
        if self.actual_latent_dim <= 0:
            print("E: AE latent dim not set.");
            return

        semantic_dim_data = self.actual_latent_dim
        default_semantic = np.zeros(semantic_dim_data, dtype=np.float32)
        semantic_vectors_cnn = {idx: data_only_semantics.get(name, default_semantic)
                                for name, idx in self.fault_types.items()}

        try:
            self.cnn_model = DualChannelCNN(input_length=self.sample_length, semantic_dim=semantic_dim_data,
                                            num_classes=self.num_classes, feature_dim=CNN_FEATURE_DIM).to(self.device)
            self.cnn_feature_dim = self.cnn_model.feature_dim
        except Exception as e:
            print(f"E: CNN init failed: {e}");
            return

        criterion_ce = nn.CrossEntropyLoss()
        criterion_contrastive = ContrastiveLoss(temperature=0.1).to(self.device)
        try:
            self.consistency_loss_fn = FeatureSemanticConsistencyLoss(
                beta=0.01,
                semantic_dim_input=self.actual_latent_dim,
                feature_dim_output=self.cnn_feature_dim
            ).to(self.device)
        except Exception as e:
            print(f"E: Consistency loss init failed: {e}");
            return

        cnn_params = list(self.cnn_model.parameters())
        if hasattr(self.consistency_loss_fn, 'projection') and self.consistency_loss_fn.projection is not None:
            consist_params = list(self.consistency_loss_fn.projection.parameters())
            if consist_params:
                print("Adding ConsistencyLoss projection parameters to CNN optimizer.")
                all_params = cnn_params + consist_params
            else:
                all_params = cnn_params
        else:
            all_params = cnn_params
        optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = None
        if has_val_data:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)

        # 初始化保存模型的变量
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15

        print(f"Starting CNN training ({epochs} epochs)...")
        for epoch in range(epochs):
            self.cnn_model.train()
            if self.consistency_loss_fn:
                self.consistency_loss_fn.train()

            train_loss, correct, total = 0.0, 0, 0
            train_loss_ce, train_loss_contr, train_loss_consist = 0.0, 0.0, 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                current_bs = inputs.size(0)
                if current_bs < 2:
                    continue

                batch_semantics_data = torch.stack(
                    [torch.from_numpy(semantic_vectors_cnn[lbl.item()]) for lbl in labels]).to(self.device)

                try:
                    logits, features = self.cnn_model(inputs, batch_semantics_data)
                    if not torch.all(torch.isfinite(logits)) or not torch.all(torch.isfinite(features)):
                        print(f"W: NaN/Inf CNN outputs epoch {epoch + 1}. Skip batch.");
                        continue
                except Exception as e:
                    print(f"E: CNN Fwd fail: {e}");
                    continue

                ce_loss = criterion_ce(logits, labels)
                contr_loss = criterion_contrastive(features, labels)
                consist_loss = self.consistency_loss_fn(features,
                                                        batch_semantics_data) if self.consistency_loss_fn else torch.tensor(
                    0.0)

                w_ce, w_contr, w_consist = 1.0, 0.1, 0.1
                total_batch_loss = w_ce * ce_loss + w_contr * contr_loss + w_consist * consist_loss

                if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                    print(f"W: NaN/Inf Loss epoch {epoch + 1}. Skip batch backprop.");
                    continue

                optimizer.zero_grad();
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

                train_loss += total_batch_loss.item() * current_bs
                train_loss_ce += ce_loss.item() * current_bs
                train_loss_contr += contr_loss.item() * current_bs
                train_loss_consist += consist_loss.item() * current_bs
                _, predicted = torch.max(logits, 1)
                total += current_bs
                correct += (predicted == labels).sum().item()

            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            avg_train_loss = train_loss / total if total > 0 else 0
            train_accuracy = 100.0 * correct / total if total > 0 else 0
            avg_train_ce = train_loss_ce / total if total > 0 else 0
            avg_train_contr = train_loss_contr / total if total > 0 else 0
            avg_train_consist = train_loss_consist / total if total > 0 else 0

            avg_val_loss, val_accuracy = 0.0, 0.0
            if val_loader is not None:
                self.cnn_model.eval()
                if self.consistency_loss_fn:
                    self.consistency_loss_fn.eval()
                val_loss, val_correct, val_total = 0.0, 0, 0
                with torch.no_grad():
                    for inputs_val, labels_val in val_loader:
                        inputs_val, labels_val = inputs_val.to(self.device), labels_val.to(self.device)
                        bs_val = inputs_val.size(0)
                        batch_semantics_val = torch.stack(
                            [torch.from_numpy(semantic_vectors_cnn[lbl.item()]) for lbl in labels_val]).to(self.device)
                        logits_val, _ = self.cnn_model(inputs_val, batch_semantics_val)
                        loss_val = criterion_ce(logits_val, labels_val)
                        if torch.isfinite(loss_val):
                            val_loss += loss_val.item() * bs_val
                        _, predicted_val = torch.max(logits_val, 1)
                        val_total += bs_val
                        val_correct += (predicted_val == labels_val).sum().item()
                avg_val_loss = val_loss / val_total if val_total > 0 else 0
                val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0

            print(
                f"E[{epoch + 1}/{epochs}] LR={current_lr:.6f} TrLss={avg_train_loss:.4f} (CE:{avg_train_ce:.4f},Ctr:{avg_train_contr:.4f},Cns:{avg_train_consist:.4f}) TrAcc={train_accuracy:.2f}% | VlLss={avg_val_loss:.4f} VlAcc={val_accuracy:.2f}%")

            # 修改后的模型保存逻辑
            save_model = False

            # 判断是否需要保存模型
            if val_accuracy > best_val_acc + 0.001:  # 验证准确率显著提高时保存
                save_model = True
                print(f"  发现更高的验证准确率: {val_accuracy:.2f}% > {best_val_acc:.2f}%")
                best_val_acc = val_accuracy
                best_val_loss = avg_val_loss
                patience_counter = 0
            elif abs(val_accuracy - best_val_acc) <= 0.001:  # 验证准确率相近或相同
                if avg_val_loss < best_val_loss:  # 在准确率相同的情况下，损失更低
                    save_model = True
                    print(
                        f"  验证准确率相似: {val_accuracy:.2f}% ≈ {best_val_acc:.2f}%，但损失更低: {avg_val_loss:.4f} < {best_val_loss:.4f}")
                    best_val_acc = val_accuracy
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    print(f"  验证准确率相似，但损失较高：不保存模型")
                    patience_counter += 1
            else:
                # 验证准确率下降
                patience_counter += 1
                print(f"  验证准确率较低: {val_accuracy:.2f}% < {best_val_acc:.2f}%")

            # 保存模型
            if save_model:
                torch.save({
                    'cnn_model_state': self.cnn_model.state_dict(),
                    'consistency_loss_state': self.consistency_loss_fn.state_dict() if self.consistency_loss_fn else None,
                    'epoch': epoch + 1,
                    'val_accuracy': val_accuracy,
                    'val_loss': avg_val_loss
                }, 'best_cnn_model.pth')
                print(
                    f"  保存最佳CNN模型于第 {epoch + 1} 轮 (验证准确率: {val_accuracy:.2f}%, 验证损失: {avg_val_loss:.4f})")

            # 提前停止检查
            if patience_counter >= patience:
                print(f"CNN训练提前停止于第 {epoch + 1} 轮")
                break

        if os.path.exists('best_cnn_model.pth'):
            checkpoint = torch.load('best_cnn_model.pth')
            self.cnn_model.load_state_dict(checkpoint['cnn_model_state'])
            if self.consistency_loss_fn and 'consistency_loss_state' in checkpoint and checkpoint[
                'consistency_loss_state']:
                self.consistency_loss_fn.load_state_dict(checkpoint['consistency_loss_state'])
            print(
                f"加载最佳CNN模型状态 (轮次: {checkpoint.get('epoch', '未知')}, 验证准确率: {checkpoint.get('val_accuracy', 0.0):.2f}%, 验证损失: {checkpoint.get('val_loss', 0.0):.4f})")
        else:
            print("W: 未找到最佳CNN模型状态文件。使用最后一轮的状态。")

        self.cnn_model.eval()
        if self.consistency_loss_fn:
            self.consistency_loss_fn.eval()
        print("CNN训练完成。")

    # train_semantic_embedding (Inputs adjusted)
    def train_semantic_embedding(self, semantic_dict, data_dict):
        """Trains SEN: Fused Semantics -> CNN Feature Space"""
        print("Training semantic embedding network (SEN)...")
        # 1. Check dependencies
        if self.cnn_model is None or self.cnn_feature_dim <= 0 : print("E: CNN model/feature dim not ready."); return
        if self.fused_semantic_dim <= 0: print("E: Fused semantic dim not ready."); return
        fused_semantics = semantic_dict.get('fused_semantics')
        if not fused_semantics: print("E: Fused semantics missing."); return
        X_train_single, y_train_single = data_dict['X_train'], data_dict['y_train']
        if len(X_train_single) == 0: print("E: No single fault data for SEN train."); return

        # 2. Initialize SEN
        try:
            self.embedding_net = SemanticEmbeddingNetwork(semantic_dim=self.fused_semantic_dim, feature_dim=self.cnn_feature_dim).to(self.device)
        except Exception as e: print(f"E: SEN Init Error: {e}"); return

        # 3. Optimizer & Loss
        criterion_mse = nn.MSELoss(); criterion_cosine = nn.CosineEmbeddingLoss(margin=0.0)
        optimizer_sen = optim.AdamW(self.embedding_net.parameters(), lr=SEN_LR, weight_decay=1e-4)
        scheduler_sen = optim.lr_scheduler.CosineAnnealingLR(optimizer_sen, T_max=SEN_EPOCHS, eta_min=SEN_LR / 20)

        # 4. SEN Training Loop
        epochs_sen = SEN_EPOCHS; batch_size_sen = self.batch_size
        num_samples_sen = len(X_train_single); best_sen_loss = float('inf')
        sen_patience_counter, sen_patience = 0, 15
        print(f"  Starting SEN training ({epochs_sen} epochs)...")
        for epoch in range(epochs_sen):
            self.embedding_net.train(); self.cnn_model.eval()
            total_sen_loss, processed_samples = 0.0, 0
            indices = np.random.permutation(num_samples_sen)

            for i in range(0, num_samples_sen, batch_size_sen):
                 batch_indices = indices[i : i + batch_size_sen]; bs = len(batch_indices)
                 if bs == 0: continue
                 batch_x = torch.FloatTensor(X_train_single[batch_indices]).to(self.device)
                 batch_y = y_train_single[batch_indices] # Numpy labels

                 # Get Target CNN Features
                 with torch.no_grad():
                     # Use None semantic for CNN's feature extraction path
                     try: cnn_features_target = self.cnn_model(batch_x, semantic=None, return_features=True)
                     except Exception as e: print(f"E: CNN FeatFail SENTr E{epoch+1}: {e}"); continue
                     if not torch.all(torch.isfinite(cnn_features_target)): print(f"W: NaN Target CNN Feat E{epoch+1}. Skip."); continue

                 # Get Source Fused Semantics & Filter Batch
                 batch_fused_semantics, valid_indices_in_batch = [], []
                 for idx_in_batch, label_idx in enumerate(batch_y):
                      fault_type = self.idx_to_fault.get(label_idx); sem_vec = fused_semantics.get(fault_type)
                      if sem_vec is not None and np.all(np.isfinite(sem_vec)) and len(sem_vec) == self.fused_semantic_dim:
                          batch_fused_semantics.append(sem_vec); valid_indices_in_batch.append(idx_in_batch)

                 if not valid_indices_in_batch: continue # Skip if no valid semantics this batch
                 batch_fused_sem_tensor = torch.FloatTensor(np.array(batch_fused_semantics)).to(self.device)
                 cnn_features_target_filtered = cnn_features_target[valid_indices_in_batch]
                 current_valid_bs = len(valid_indices_in_batch)

                 # SEN Forward & Loss Calc
                 optimizer_sen.zero_grad()
                 try: embedded_semantics = self.embedding_net(batch_fused_sem_tensor)
                 except Exception as e: print(f"E: SEN FwdFail E{epoch+1}: {e}"); continue
                 if not torch.all(torch.isfinite(embedded_semantics)): print(f"W: NaN SEN Out E{epoch+1}. Skip."); continue

                 loss_mse = criterion_mse(embedded_semantics, cnn_features_target_filtered)
                 loss_cos = criterion_cosine(embedded_semantics, cnn_features_target_filtered, torch.ones(current_valid_bs, device=self.device))
                 total_batch_sen_loss = 1.0 * loss_mse + 0.1 * loss_cos # Weights

                 # SEN Backward & Optimize
                 if torch.isnan(total_batch_sen_loss) or torch.isinf(total_batch_sen_loss): print(f"W: NaN SEN Loss E{epoch+1}"); continue
                 total_batch_sen_loss.backward(); torch.nn.utils.clip_grad_norm_(self.embedding_net.parameters(), 1.0); optimizer_sen.step()
                 total_sen_loss += total_batch_sen_loss.item() * current_valid_bs; processed_samples += current_valid_bs

            # --- End of SEN Epoch ---
            scheduler_sen.step()
            if processed_samples > 0:
                 avg_sen_loss = total_sen_loss / processed_samples
                 print(f"Epoch [{epoch+1}/{epochs_sen}] - SEN Loss: {avg_sen_loss:.6f}")
                 if avg_sen_loss < best_sen_loss: best_sen_loss = avg_sen_loss; sen_patience_counter = 0; torch.save(self.embedding_net.state_dict(), 'best_semantic_embedding_net.pth')
                 else: sen_patience_counter += 1;
                 if sen_patience_counter >= sen_patience: print("SEN Early stopping."); break
            else: print(f"Epoch [{epoch+1}/{epochs_sen}] - SEN: No samples processed.")

        # Load best SEN and set to eval
        if os.path.exists('best_semantic_embedding_net.pth'): self.embedding_net.load_state_dict(torch.load('best_semantic_embedding_net.pth')); print("Loaded best SEN state.")
        self.embedding_net.eval()
        print("SEN Training Complete.")

    # generate_compound_fault_projections unchanged
    def generate_compound_fault_projections(self, semantic_dict):
        """Generates compound fault projections using FUSED semantics and SEN."""
        print("Generating compound fault projections...")
        if self.embedding_net is None: print("E: SEN not trained."); return None
        fused_semantics = semantic_dict.get('fused_semantics')
        if not fused_semantics or self.fused_semantic_dim <= 0: print("E: Fused semantics invalid."); return None

        compound_fused_semantics = {ft: fused_semantics[ft] for ft in self.compound_fault_types if ft in fused_semantics and fused_semantics[ft] is not None and len(fused_semantics[ft])==self.fused_semantic_dim and np.all(np.isfinite(fused_semantics[ft]))}
        if not compound_fused_semantics: print("E: No valid compound fused semantics."); return None

        self.embedding_net.eval(); compound_projections = {}
        with torch.no_grad():
             for fault_type, semantic_vec in compound_fused_semantics.items():
                 semantic_tensor = torch.FloatTensor(semantic_vec).unsqueeze(0).to(self.device)
                 try:
                      projected_feature = self.embedding_net(semantic_tensor)
                      if torch.all(torch.isfinite(projected_feature)): compound_projections[fault_type] = projected_feature.cpu().numpy().squeeze(axis=0)
                      else: print(f"W: NaN projection for {fault_type}.")
                 except Exception as e: print(f"E: SEN projection error for {fault_type}: {e}")
        print(f"Generated projections for {len(compound_projections)} compound types.")
        return compound_projections


    # evaluate_zero_shot unchanged
    def evaluate_zero_shot(self, data_dict, compound_projections):
        """Evaluates zero-shot performance."""
        print("Evaluating zero-shot compound fault classification...")
        # 1. Checks
        if self.cnn_model is None or self.cnn_feature_dim <=0: print("E: CNN not ready for eval."); return 0.0, None
        if compound_projections is None or not compound_projections: print("E: No projections for eval."); return 0.0, None

        # 2. Get & Filter Test Data
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        if len(X_test) == 0: print("W: No compound test data."); return 0.0, None
        finite_mask_test = np.all(np.isfinite(X_test), axis=1); X_test, y_test = X_test[finite_mask_test], y_test[finite_mask_test]
        if len(X_test)==0: print("E: No finite test data left."); return 0.0, None

        available_projection_labels = list(compound_projections.keys())
        available_projection_indices = [self.fault_types[label] for label in available_projection_labels if label in self.fault_types]
        test_mask = np.isin(y_test, available_projection_indices)
        X_compound_test, y_compound_test = X_test[test_mask], y_test[test_mask] # Numerical labels
        if len(X_compound_test)==0: print("E: No test samples match available projections."); return 0.0, None
        print(f"  Eval on {len(X_compound_test)} samples for {len(available_projection_labels)} compound types.")

        # 3. Prepare Projections & CNN
        candidate_labels = available_projection_labels
        candidate_projections_tensor = torch.stack([torch.from_numpy(compound_projections[label]) for label in candidate_labels]).to(self.device)
        candidate_projections_norm = F.normalize(candidate_projections_tensor, p=2, dim=1)
        if torch.isnan(candidate_projections_norm).any(): candidate_projections_norm = torch.nan_to_num(candidate_projections_norm, nan=0.0)
        self.cnn_model.eval()

        # 4. Predict on Test Data
        y_pred_indices = []
        inference_batch_size = self.batch_size * 2
        with torch.no_grad():
             num_test = len(X_compound_test)
             for i in range(0, num_test, inference_batch_size):
                 batch_x_test = torch.FloatTensor(X_compound_test[i : i + inference_batch_size]).to(self.device)
                 try: # Use semantic=None to get CNN features
                      batch_features_test = self.cnn_model(batch_x_test, semantic=None, return_features=True)
                 except Exception as e: print(f"E: CNN FeatFail ZSL Eval: {e}"); y_pred_indices.extend([0]*len(batch_x_test)); continue # Predict class 0 on error
                 if not torch.all(torch.isfinite(batch_features_test)): batch_features_test = torch.nan_to_num(batch_features_test, nan=0.0) # Clamp NaNs

                 batch_features_norm = F.normalize(batch_features_test, p=2, dim=1)
                 if torch.isnan(batch_features_norm).any(): batch_features_norm = torch.nan_to_num(batch_features_norm, nan=0.0)
                 similarity_matrix = torch.matmul(batch_features_norm, candidate_projections_norm.t())
                 batch_pred = torch.argmax(similarity_matrix, dim=1).cpu().numpy()
                 y_pred_indices.extend(batch_pred)

        # 5. Calculate Metrics
        y_pred_numerical = [self.fault_types.get(candidate_labels[idx], -1) for idx in y_pred_indices]
        y_true = y_compound_test
        accuracy = accuracy_score(y_true, y_pred_numerical) * 100
        true_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_true]
        pred_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_pred_numerical]
        display_labels = sorted(list(set(true_labels_str))) # Use true labels present for matrix rows/cols

        try: conf_matrix = confusion_matrix(true_labels_str, pred_labels_str, labels=display_labels)
        except Exception as e: print(f"W: CM Error: {e}"); conf_matrix = None

        print(f"ZSL Accuracy: {accuracy:.2f}%")
        # 6. Visualize
        if conf_matrix is not None:
            plt.figure(figsize=(8, 6)); sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels, yticklabels=display_labels)
            plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'ZSL Confusion Matrix (Acc: {accuracy:.2f}%)')
            plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
            plt.savefig('compound_fault_confusion_matrix_zsl.png'); plt.close(); print("ZSL CM saved.")
        return accuracy, conf_matrix


    # run_pipeline unchanged
    def run_pipeline(self):
        """Runs the full ZSL pipeline with joint training."""
        start_time = time.time();
        accuracy = 0.0
        try:
            print("\n--- Step 1: Load Data ---")
            data_dict = self.load_data()
            if data_dict is None: raise RuntimeError("Data loading failed")

            print("\n--- Step 2: Build Semantics ---")
            semantic_dict = self.build_semantics(data_dict)
            if semantic_dict is None: raise RuntimeError("Semantic building failed")

            print("\n--- Step 3: Visualize ---")
            self.visualize_semantics(semantic_dict)
            self.visualize_data_semantics_distribution(data_dict)

            print("\n--- Step 4: Joint Train CNN+SEN ---")
            # 使用联合训练替代单独训练
            self.joint_train_cnn_sen(data_dict, semantic_dict, epochs=40, lr=0.0005)
            if self.cnn_model is None or self.embedding_net is None:
                raise RuntimeError("Joint training failed")

            print("\n--- Step 5: Generate Projections ---")
            compound_projections = self.generate_compound_fault_projections(semantic_dict)
            if compound_projections is None: raise RuntimeError("Projection generation failed")

            print("\n--- Step 6: Evaluate ZSL ---")
            accuracy, _ = self.evaluate_zero_shot(data_dict, compound_projections)

        except Exception as e:
            print(f"E: Pipeline failed: {e}")
            traceback.print_exc()
            accuracy = 0.0

        end_time = time.time()
        print(f"\n--- Pipeline Finished in {(end_time - start_time) / 60:.2f} minutes ---")
        return accuracy

if __name__ == "__main__":
    set_seed(42)
    data_path = "E:/研究生/CNN/HDU1000" # <--- !!! MODIFY THIS PATH !!!
    if not os.path.isdir(data_path):
        print(f"E: Data directory not found: {data_path}")
    else:
        fault_diagnosis = ZeroShotCompoundFaultDiagnosis(
            data_path=data_path, sample_length=SEGMENT_LENGTH,
            latent_dim=AE_LATENT_DIM, batch_size=DEFAULT_BATCH_SIZE )
        final_accuracy = fault_diagnosis.run_pipeline()
        print(f"\n>>> Final ZSL Accuracy: {final_accuracy:.2f}% <<<")
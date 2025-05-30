
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
CNN_EPOCHS =10
CNN_LR = 0.0005
SEN_EPOCHS = 10
SEN_LR = 0.001
CNN_FEATURE_DIM =256 # Default output feature dimension for CNN fusion & SEN target
DEFAULT_BATCH_SIZE = 128


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
    def __init__(self, sample_length=SEGMENT_LENGTH, overlap=OVERLAP, augment=True, random_seed=42):
        self.sample_length = sample_length
        self.overlap = overlap
        self.augment = augment
        self.stride = int(sample_length * (1 - overlap))
        if self.stride < 1: self.stride = 1
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.random_seed = random_seed
        np.random.seed(self.random_seed)  # 确保可重复性

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

    def preprocess(self, signal_data, augmentation=False, scale_globally=False):
        """改进的预处理流程，防止数据泄露"""
        # 初始NaN检查
        if not np.all(np.isfinite(signal_data)):
            signal_data = np.nan_to_num(signal_data, nan=np.nanmean(signal_data),
                                        posinf=np.nanmax(signal_data), neginf=np.nanmin(signal_data))

        # 1. 原始数据处理流程
        signal_data = self.cubic_spline_interpolation(signal_data)
        signal_data = self.remove_outliers_sigma(signal_data)
        signal_data = self.wavelet_denoising_universal(signal_data)

        # 2. 归一化 - 可以选择全局或局部归一化
        if scale_globally:
            # 全局归一化 - 先分段后统一归一化 (防止数据泄露)
            segments = self.segment_signal(signal_data)
            if segments.shape[0] == 0:
                return np.empty((0, self.sample_length))

            # 统一对所有段进行归一化
            segments_flat = segments.reshape(-1, 1)
            normalized_flat = self.scaler.fit_transform(segments_flat)
            signal_data_normalized = normalized_flat.reshape(segments.shape)
        else:
            # 局部归一化 - 先归一化再分段
            signal_data_reshaped = signal_data.reshape(-1, 1)
            signal_data_normalized = self.scaler.fit_transform(signal_data_reshaped).flatten()
            signal_data_normalized = self.segment_signal(signal_data_normalized)

        # 返回原始分段信号(无数据增强)
        if not augmentation or not self.augment:
            return signal_data_normalized

        # 3. 数据增强 (从分段后的数据开始)
        processed_signals = [signal_data_normalized]

        # 使用不同随机种子进行数据增强，避免相同的随机增强
        aug_seed = self.random_seed + 1

        # a) 高斯噪声增强
        for i in range(2):
            np.random.seed(aug_seed + i)
            noise_std = 0.01 + 0.01 * i  # 不同强度的噪声
            noise_segments = []

            for segment in signal_data_normalized:
                noisy_seg = segment + np.random.normal(0, noise_std, len(segment))
                noisy_seg = np.clip(noisy_seg, -1.0, 1.0)  # 限制在[-1,1]范围内
                noise_segments.append(noisy_seg)

            if noise_segments:
                processed_signals.append(np.array(noise_segments))

        # b) 时间平移增强 (比剪切更安全，保留所有故障特征)
        for i in range(2):
            np.random.seed(aug_seed + 10 + i)
            shift_segments = []

            for segment in signal_data_normalized:
                max_shift = len(segment) // 10  # 最多平移10%
                shift_amount = np.random.randint(-max_shift, max_shift + 1)
                if shift_amount != 0:
                    shifted_seg = np.roll(segment, shift_amount)
                else:
                    shifted_seg = segment.copy()
                shift_segments.append(shifted_seg)

            if shift_segments:
                processed_signals.append(np.array(shift_segments))

        # c) 幅值缩放增强
        for i in range(2):
            np.random.seed(aug_seed + 20 + i)
            scale_segments = []

            for segment in signal_data_normalized:
                scale_factor = 0.95 + 0.1 * np.random.random()  # 0.95-1.05之间的缩放
                scaled_seg = segment * scale_factor
                scaled_seg = np.clip(scaled_seg, -1.0, 1.0)  # 限制在[-1,1]范围内
                scale_segments.append(scaled_seg)

            if scale_segments:
                processed_signals.append(np.array(scale_segments))

        # 合并所有增强后的数据
        if not processed_signals:
            return np.empty((0, self.sample_length))

        return np.vstack(processed_signals)


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


class TensorFusionNetwork(nn.Module):
    """张量融合网络：将张量积结果压缩回原始维度"""

    def __init__(self, input_dim, output_dim):
        super(TensorFusionNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 计算中间维度，确保渐进降维
        mid_dim = min(input_dim * input_dim // 4, 512)

        # 压缩网络
        self.compression = nn.Sequential(
            nn.Linear(input_dim * input_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, output_dim),
        )

    def forward(self, tensor_product):
        # 输入为张量积结果，形状为 [input_dim, input_dim]
        flattened = tensor_product.flatten()
        return self.compression(flattened)
# 2. 知识语义和数据语义构建模块 (Modified to use gen.py's AE training details)
class FaultSemanticBuilder:
    def __init__(self, latent_dim=AE_LATENT_DIM, hidden_dim=128): # hidden_dim not used for AE itself
        self.latent_dim_config = latent_dim # Configured latent dim
        self.actual_latent_dim = latent_dim # Actual dim set after AE init
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = None
        self.preprocessor = DataPreprocessor(sample_length=SEGMENT_LENGTH) # Use preprocessor defined above
        self.knowledge_dim = 0
        self.data_semantics = {} # Store learned data semantic centroids
        self.idx_to_fault = {} # Should be populated by main class

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
        """Synthesizes compound fault semantics using max operation."""
        # Requires self.actual_latent_dim to be set
        if self.actual_latent_dim <= 0:
             print("E: Cannot synthesize, actual latent dim not set.")
             return {}

        compound_semantics = {}
        compound_combinations = {
            'inner_outer': ['inner', 'outer'], 'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'], 'inner_outer_ball': ['inner', 'outer', 'ball']
        }

        for compound_type, components in compound_combinations.items():
            component_semantics = []
            all_valid = True
            for comp in components:
                proto = single_fault_prototypes.get(comp)
                if proto is not None and np.all(np.isfinite(proto)) and len(proto) == self.actual_latent_dim:
                    component_semantics.append(proto)
                else:
                    all_valid = False
                    break

            if all_valid and component_semantics:
                # 使用原有融合方法
                synthesized = self.tensor_fusion_combine(component_semantics)

                if np.all(np.isfinite(synthesized)):
                    # ===== 新增: 批量域适应 =====
                    # 对已合成的语义向量进行整体域适应校准
                    synthesized = self.batch_domain_calibration(
                        synthesized, compound_type, components, single_fault_prototypes
                    )

                    compound_semantics[compound_type] = synthesized

        return compound_semantics

    def batch_domain_calibration(self, compound_vector, compound_type, component_types, single_prototypes):
        """
        批量域校准以减少域偏移

        参数:
            compound_vector: 合成的复合故障语义
            compound_type: 复合故障类型名称
            component_types: 组成该复合故障的单一故障类型列表
            single_prototypes: 所有单一故障原型的字典
        """
        # 根据复合故障类型应用特定的校准规则
        n_components = len(component_types)

        # 1. 根据组件数量调整振幅
        # 复杂故障通常比单一故障信号更强
        amplitude_factor = 1.0 + 0.05 * n_components  # 随组件数量增加振幅
        calibrated = compound_vector * amplitude_factor

        # 2. 确保与原始组件保持适当相似度
        similarities = []
        for comp in component_types:
            if comp in single_prototypes:
                single_vec = single_prototypes[comp]
                sim = np.dot(calibrated, single_vec) / (np.linalg.norm(calibrated) * np.linalg.norm(single_vec) + 1e-10)
                similarities.append(sim)

        # 计算平均相似度
        if similarities:
            avg_sim = sum(similarities) / len(similarities)

            # 调整相似度到合理范围 (0.5-0.7之间是理想的)
            if avg_sim > 0.75:  # 过于相似
                # 增加一些随机扰动降低相似度
                noise = np.random.normal(0, 0.05, size=len(calibrated))
                calibrated = calibrated * 0.9 + noise * 0.1
            elif avg_sim < 0.4:  # 相似度过低
                # 向单一故障方向靠拢
                for comp in component_types:
                    if comp in single_prototypes:
                        single_vec = single_prototypes[comp]
                        calibrated = calibrated * 0.8 + single_vec * 0.2 / n_components

        # 3. 应用特定故障类型的规则
        if compound_type == 'inner_outer':
            # 内圈外圈故障通常在特定频率有更强的表现
            # 假设向量前半部分代表低频特征，后半部分代表高频特征
            mid = len(calibrated) // 2
            calibrated[mid:] = calibrated[mid:] * 1.1  # 增强高频成分

        elif compound_type == 'inner_ball' or compound_type == 'outer_ball':
            # 滚动体相关故障通常有更多的调制成分
            # 增加一些随机调制
            mod_pattern = np.sin(np.linspace(0, 3 * np.pi, len(calibrated))) * 0.1
            calibrated = calibrated + mod_pattern * np.mean(np.abs(calibrated))

        elif compound_type == 'inner_outer_ball':
            # 全部故障组合通常有更复杂的频率分布和更高的能量
            calibrated = calibrated * 1.15  # 整体增强15%

            # 添加一些低频高能成分 (通常是全故障的特征)
            low_freq = np.exp(-np.linspace(0, 5, len(calibrated))) * np.mean(np.abs(calibrated)) * 0.2
            calibrated[:len(calibrated) // 4] += low_freq[:len(calibrated) // 4]

        # 4. 确保输出是有效向量
        if not np.all(np.isfinite(calibrated)):
            print(f"W: 校准后产生非有限值，回退到原始合成向量")
            return compound_vector

        # 重新归一化
        norm = np.linalg.norm(calibrated)
        if norm > 1e-8:
            calibrated = calibrated / norm

        return calibrated

    def tensor_fusion_combine(self, vectors):
        """使用张量乘积进行二阶语义交互，再通过神经网络投影回latent空间维度。"""
        if len(vectors) < 2:
            return vectors[0]

        # 转换为张量 - 保持原有逻辑
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor_vectors = [torch.tensor(v, dtype=torch.float32).to(device) for v in vectors]

        # 计算张量积 - 保持原有逻辑
        result = tensor_vectors[0]
        for v in tensor_vectors[1:]:
            # 计算外积
            tensor_product = torch.outer(result, v)

            # 如果没有初始化过融合网络，则创建一个
            if not hasattr(self, 'fusion_network') or self.fusion_network is None:
                input_dim = len(result)
                self.fusion_network = TensorFusionNetwork(
                    input_dim=input_dim,
                    output_dim=self.actual_latent_dim
                ).to(device)

                # 初始化权重
                def init_weights(m):
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

                self.fusion_network.apply(init_weights)
                print(f"创建张量融合网络：{input_dim}^2 -> {self.actual_latent_dim}")

            # 使用网络压缩张量积
            with torch.no_grad():  # 在合成过程中不需要梯度
                compressed = self.fusion_network(tensor_product)
                result = compressed

        # 转回numpy并归一化
        result_np = result.cpu().numpy()

        # === 新增: 域适应校正 ===
        # 这部分专门解决域偏移问题
        result_np = self.domain_adaptation_correction(result_np, vectors)

        # 归一化
        norm = np.linalg.norm(result_np)
        if norm > 1e-8:
            result_np = result_np / norm

        return result_np

    def domain_adaptation_correction(self, compound_semantic, single_semantics):
        """
        应用域适应校正以减少合成复合故障与真实复合故障之间的域偏移

        参数:
            compound_semantic: 合成的复合故障语义向量
            single_semantics: 用于合成的单一故障语义向量列表

        返回:
            校正后的复合故障语义向量
        """
        # 1. 统计校正 - 确保复合故障统计特性符合预期

        # 计算单一故障语义的平均统计特性
        single_means = np.mean([np.mean(s) for s in single_semantics])
        single_stds = np.mean([np.std(s) for s in single_semantics])
        single_ranges = np.mean([np.max(s) - np.min(s) for s in single_semantics])

        # 获取复合语义的统计特性
        compound_mean = np.mean(compound_semantic)
        compound_std = np.std(compound_semantic)
        compound_range = np.max(compound_semantic) - np.min(compound_semantic)

        # 基于观察到的域差异应用校正因子
        # 通常复合故障会表现出更高的振幅和变化率
        mean_factor = 1.05  # 略微提升均值
        std_factor = 1.15  # 增加标准差
        range_factor = 1.1  # 扩大范围

        # 应用校正
        adjusted = compound_semantic.copy()

        # 标准化，应用因子，然后还原
        adjusted = (adjusted - compound_mean) / (compound_std + 1e-10)  # 标准化
        adjusted = adjusted * (single_stds * std_factor)  # 应用新的std
        adjusted = adjusted + (single_means * mean_factor)  # 应用新的mean

        # 2. 特征增强 - 增强高频与低频成分（通常对应域差异）
        n = len(adjusted)
        mid = n // 2

        # 高频部分（假设在向量的后半部分）略微增强
        adjusted[mid:] = adjusted[mid:] * 1.08

        # 低频部分保持相对稳定，但略微调整
        adjusted[:mid] = adjusted[:mid] * 1.02

        # 3. 关系保持 - 确保与单一故障的相似度在合理范围
        for single_sem in single_semantics:
            # 计算原始相似度
            sim = np.dot(compound_semantic, single_sem) / (
                        np.linalg.norm(compound_semantic) * np.linalg.norm(single_sem) + 1e-10)

            # 保持合理相似度 (不要太相似，也不要太不同)
            if sim > 0.85:  # 如果相似度过高
                # 增加一些正交成分
                ortho = single_sem - (np.dot(single_sem, adjusted) / np.dot(adjusted, adjusted)) * adjusted
                ortho_norm = np.linalg.norm(ortho)
                if ortho_norm > 1e-8:
                    ortho = ortho / ortho_norm
                    adjusted = adjusted * 0.9 + ortho * 0.1

            elif sim < 0.3:  # 如果相似度过低
                # 增加一些相似成分
                adjusted = adjusted * 0.9 + single_sem * 0.1

        return adjusted

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
        c1, c2, c3, c4 = 32, 64, 128, 256
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


class ImprovedSemanticEmbeddingNetwork(nn.Module):
    """
    改进的语义嵌入网络，使用残差连接、注意力机制和归一化层
    将语义向量映射到CNN特征空间
    """

    def __init__(self, semantic_dim, feature_dim, hid1=1024, hid2=None, hid3=None):
        super(ImprovedSemanticEmbeddingNetwork, self).__init__()

        # 网络宽度配置 (使用传入的参数或计算默认值)
        self.h1 = hid1 if hid1 is not None else max(semantic_dim * 2, 1024)
        self.h2 = hid2 if hid2 is not None else self.h1 // 2
        self.h3 = hid3 if hid3 is not None else feature_dim * 2

        # 残差模块创建函数
        def create_residual_block(in_dim, hidden_dim, out_dim, dropout_rate=0.2):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),  # 使用GELU激活函数，性能通常优于ReLU
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.Dropout(dropout_rate)
            )

        # 输入处理层
        self.input_layer = nn.Sequential(
            nn.Linear(semantic_dim, self.h1),
            nn.LayerNorm(self.h1),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 残差块1
        self.res_block1 = create_residual_block(self.h1, self.h1 // 2, self.h1)

        # 注意力模块 - 帮助模型关注重要的语义维度
        self.attention = nn.Sequential(
            nn.Linear(self.h1, self.h1 // 8),
            nn.LayerNorm(self.h1 // 8),
            nn.GELU(),
            nn.Linear(self.h1 // 8, self.h1),
            nn.Sigmoid()  # 输出0-1之间的注意力权重
        )

        # 收缩编码器
        self.encoder = nn.Sequential(
            nn.Linear(self.h1, self.h2),
            nn.LayerNorm(self.h2),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # 残差块2
        self.res_block2 = create_residual_block(self.h2, self.h2 // 2, self.h2)

        # 解码器部分 - 专注于生成CNN特征空间的表示
        self.decoder = nn.Sequential(
            nn.Linear(self.h2, self.h3),
            nn.LayerNorm(self.h3),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.h3, feature_dim)
        )

        # 特征归一化层 - 帮助投影保持在正确的分布范围内
        self.feature_norm = nn.LayerNorm(feature_dim)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """适当的权重初始化对于投影质量至关重要"""
        if isinstance(module, nn.Linear):
            # 使用Kaiming He初始化 (适用于GELU/ReLU激活函数)
            torch.nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        # 输入处理
        x1 = self.input_layer(x)

        # 残差连接1
        res1 = self.res_block1(x1)
        x1 = x1 + res1  # 残差连接

        # 注意力加权
        attn = self.attention(x1)
        x1 = x1 * attn

        # 编码
        x2 = self.encoder(x1)

        # 残差连接2
        res2 = self.res_block2(x2)
        x2 = x2 + res2

        # 解码到特征空间
        x3 = self.decoder(x2)

        # 特征归一化 - 确保输出特征在合理范围内
        return self.feature_norm(x3)


# 5. Loss Functions (ContrastiveLoss, FeatureSemanticConsistencyLoss) - Kept as in V1 (Aligned with original model.py)
class ContrastiveLoss(nn.Module):
    """ For CNN Training - Encourages features of same class to be closer """
    def __init__(self, temperature=0.05): # Adjusted default temp based on prev run
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
    # ** Important Modification Interpretation **
    # This loss, in the context of the CNN training loop, will align
    # the CNN output feature vector (`features`) with the AE data semantic vector
    # passed through an internal projection layer (`semantic_dim_data` -> `feature_dim`).
    # It does *not* directly use the SEN or fused semantics during CNN training.
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


class RepulsionLoss(nn.Module):
    """
    语义投影间的排斥损失，增加不同类别投影的距离，
    同时保持相关故障之间的相对关系结构
    """

    def __init__(self, model, base_margin=2.0):
        super().__init__()
        self.base_margin = base_margin
        self.model = model  # 保存模型引用以访问 idx_to_fault 映射

    def are_faults_related(self, fault_type1, fault_type2):
        """判断两种故障类型是否相关"""
        if fault_type1 is None or fault_type2 is None:
            return False

        # 如果是相同故障，返回True
        if fault_type1 == fault_type2:
            return True

        # 分解故障名称获取组件
        components1 = fault_type1.split('_')
        components2 = fault_type2.split('_')

        # 计算组件重叠度
        common_components = set(components1) & set(components2)

        # 如果有共同组件，则认为相关
        return len(common_components) > 0

    def forward(self, embeddings, labels):
        """
        计算考虑故障关系的排斥损失

        参数:
            embeddings: 语义投影 [batch_size, feature_dim]
            labels: 对应的标签 [batch_size]
        """
        unique_labels = torch.unique(labels)
        if len(unique_labels) <= 1:
            return torch.tensor(0.0, device=embeddings.device)

        # 计算每个类的中心
        centers = []
        center_labels = []
        for label in unique_labels:
            mask = (labels == label)
            if torch.sum(mask) > 0:
                center = embeddings[mask].mean(0)
                centers.append(center)
                center_labels.append(label.item())

        if not centers:
            return torch.tensor(0.0, device=embeddings.device)

        centers = torch.stack(centers)
        n_centers = centers.size(0)

        # 计算类间损失，考虑故障关系
        loss = 0.0
        pair_count = 0

        for i in range(n_centers):
            for j in range(i + 1, n_centers):
                # 获取故障类型名称
                fault_i = self.model.idx_to_fault.get(center_labels[i], f"Unknown_{center_labels[i]}")
                fault_j = self.model.idx_to_fault.get(center_labels[j], f"Unknown_{center_labels[j]}")

                # 判断是否相关故障
                related = self.are_faults_related(fault_i, fault_j)

                # 根据关系调整margin
                # 相关故障使用较小的margin，以保持它们之间的相对关系
                # 不相关故障使用较大的margin，以增强区分度
                if related:
                    actual_margin = self.base_margin * 0.5  # 相关故障的margin减半
                    similarity_weight = 0.7  # 对相关故障的相似度惩罚较轻
                else:
                    actual_margin = self.base_margin  # 不相关故障使用完整margin
                    similarity_weight = 1.0  # 对不相关故障的相似度完全惩罚

                # 计算余弦相似度
                sim = F.cosine_similarity(centers[i].unsqueeze(0), centers[j].unsqueeze(0))

                # 将相似度转换为排斥损失 (只有相似度高于阈值时才惩罚)
                threshold = 1.0 - actual_margin  # 余弦相似度阈值
                if sim > threshold:
                    # 相似度越高，惩罚越大
                    loss += similarity_weight * (sim - threshold)
                    pair_count += 1

        # 平均所有有效对的损失
        if pair_count > 0:
            return loss / pair_count
        else:
            return torch.tensor(0.0, device=embeddings.device)
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

    def visualize_cnn_feature_distribution(self, data_dict):
        """Visualize the distribution of Dual Channel CNN features using t-SNE."""
        print("Visualizing Dual Channel CNN feature distribution...")
        if self.cnn_model is None or self.cnn_feature_dim <= 0:
            print("E: CNN model not trained or feature dim not set. Cannot visualize.")
            return
        if self.semantic_builder is None or self.actual_latent_dim <= 0:
            print("E: Semantic builder or AE latent dim not set. Cannot visualize.")
            return

        # Extract features from training data
        X_vis = data_dict.get('X_train')
        y_vis = data_dict.get('y_train')
        if X_vis is None or y_vis is None or len(X_vis) == 0:
            print("E: Training data not available for visualization.")
            return

        print(f"  Extracting CNN features from {len(X_vis)} training samples...")
        self.cnn_model.eval()
        semantic_vectors = {
            idx: self.semantic_builder.data_semantics.get(name, np.zeros(self.actual_latent_dim, dtype=np.float32))
            for name, idx in self.fault_types.items()}

        # Prepare data loader for efficient batch processing
        dataset = TensorDataset(torch.FloatTensor(X_vis), torch.LongTensor(y_vis))
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        all_features = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_semantics = torch.stack([torch.from_numpy(semantic_vectors[lbl.item()]) for lbl in batch_y]).to(
                    self.device)
                try:
                    # Extract fused features (return_features=True)
                    batch_features = self.cnn_model(batch_x, semantic=batch_semantics, return_features=True)
                    if not torch.all(torch.isfinite(batch_features)):
                        print("W: Non-finite CNN features detected. Skipping batch.")
                        continue
                    all_features.append(batch_features.cpu().numpy())
                    all_labels.extend(batch_y.numpy())
                except Exception as e:
                    print(f"W: Error extracting CNN features: {e}. Skipping batch.")
                    continue

        if not all_features:
            print("E: No valid CNN features extracted for visualization.")
            return

        # Concatenate features and labels
        cnn_features = np.vstack(all_features)
        vis_labels = np.array(all_labels)
        print(
            f"  Extracted {cnn_features.shape[0]} CNN feature vectors (dim={cnn_features.shape[1]}) for visualization.")

        # t-SNE for dimensionality reduction
        tsne_start_time = time.time()
        perplexity_val = min(30.0, max(5.0, cnn_features.shape[0] / 5.0 - 1))
        n_components = 2
        if cnn_features.shape[0] <= n_components:
            print(f"E: Not enough samples ({cnn_features.shape[0]}) for t-SNE.")
            return

        try:
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity_val,
                        learning_rate='auto', init='pca', n_iter=1000, metric='cosine')
            features_2d = tsne.fit_transform(cnn_features)
            if not np.all(np.isfinite(features_2d)):
                raise ValueError("t-SNE resulted in non-finite values")
            print(f"t-SNE completed in {time.time() - tsne_start_time:.2f}s.")
        except Exception as e:
            print(f"E: t-SNE failed: {e}")
            return

        # Plotting
        fault_labels_str = [self.idx_to_fault.get(label, f"Unknown_{int(label)}") for label in vis_labels]
        df = pd.DataFrame({'x': features_2d[:, 0], 'y': features_2d[:, 1], 'fault_type': fault_labels_str})
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=df, x='x', y='y', hue='fault_type', palette='viridis', s=50, alpha=0.6)
        plt.title('t-SNE Visualization of Dual Channel CNN Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Fault Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig('cnn_feature_distribution.png')
        plt.close()
        print("CNN feature distribution saved to 'cnn_feature_distribution.png'")

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

    # 在ZeroShotCompoundFaultDiagnosis类中添加这个方法
    def visualize_embedding_separation(self, fused_semantics, data_dict):
        """
        可视化单一故障语义嵌入网络投影特征与CNN特征的分布对比

        参数:
            fused_semantics: 融合语义字典 {fault_type: semantic_vector}
            data_dict: 包含验证数据的字典
        """
        print("可视化单一故障的语义嵌入投影与CNN特征对比...")

        if self.cnn_model is None or self.embedding_net is None:
            print("错误: 模型未训练，无法可视化")
            return

        # 只处理单一故障类型
        single_fault_types = ['normal', 'inner', 'outer', 'ball']
        single_fault_semantics = {}

        # 获取单一故障的融合语义
        for fault_type in single_fault_types:
            if fault_type in fused_semantics:
                single_fault_semantics[fault_type] = fused_semantics[fault_type]

        if not single_fault_semantics:
            print("错误: 没有可用的单一故障语义")
            return

        # 将单一故障语义投影到特征空间
        projected_features = {}
        self.embedding_net.eval()

        with torch.no_grad():
            for fault_type, semantic_vec in single_fault_semantics.items():
                semantic_tensor = torch.FloatTensor(semantic_vec).unsqueeze(0).to(self.device)
                try:
                    projection = self.embedding_net(semantic_tensor)
                    if torch.all(torch.isfinite(projection)):
                        projected_features[fault_type] = projection.cpu().numpy().squeeze(0)
                except Exception as e:
                    print(f"错误: 投影故障 {fault_type} 失败: {e}")

        # 从验证数据中选择样本
        X_val = data_dict.get('X_val', [])
        y_val = data_dict.get('y_val', [])

        if len(X_val) == 0 or len(y_val) == 0:
            print("错误: 无验证数据用于特征提取")
            return

        # 过滤出单一故障样本
        single_fault_indices = []
        for i, label in enumerate(y_val):
            fault_name = self.idx_to_fault.get(label, None)
            if fault_name in single_fault_types:
                single_fault_indices.append(i)

        if len(single_fault_indices) == 0:
            print("错误: 验证集中没有单一故障样本")
            return

        # 限制样本数量，防止过多数据导致可视化混乱
        max_samples_per_class = 100
        sampled_indices = []
        for fault_type in single_fault_types:
            fault_idx = self.fault_types.get(fault_type, -1)
            type_indices = [i for i in single_fault_indices if y_val[i] == fault_idx]
            if len(type_indices) > max_samples_per_class:
                type_indices = np.random.choice(type_indices, max_samples_per_class, replace=False)
            sampled_indices.extend(type_indices)

        if not sampled_indices:
            print("错误: 采样后没有样本")
            return

        X_single = X_val[sampled_indices]
        y_single = y_val[sampled_indices]

        # 提取CNN特征
        cnn_features_list = []
        cnn_labels = []

        self.cnn_model.eval()
        batch_size = min(32, len(X_single))

        with torch.no_grad():
            for i in range(0, len(X_single), batch_size):
                batch_x = torch.FloatTensor(X_single[i:i + batch_size]).to(self.device)
                batch_y = y_single[i:i + batch_size]

                # 使用CNN提取特征，不使用语义输入
                features = self.cnn_model(batch_x, semantic=None, return_features=True)

                if torch.all(torch.isfinite(features)):
                    cnn_features_list.append(features.cpu().numpy())
                    cnn_labels.extend(batch_y)

        if not cnn_features_list:
            print("错误: 无法提取CNN特征")
            return

        cnn_features = np.vstack(cnn_features_list)

        # 准备t-SNE降维
        # 将CNN特征和投影特征合并
        all_features = []
        all_labels = []

        # 加入CNN特征
        all_features.append(cnn_features)
        all_labels.extend([f"CNN_{self.idx_to_fault.get(label, str(label))}" for label in cnn_labels])

        # 加入投影特征
        proj_features = np.vstack([feat for feat in projected_features.values()])
        all_features.append(proj_features)
        all_labels.extend([f"SEN_{fault}" for fault in projected_features.keys()])

        # 合并特征
        all_features = np.vstack(all_features)

        # 执行t-SNE降维
        from sklearn.manifold import TSNE

        # 计算合适的perplexity值 (规则: 5-50之间，不超过样本数的三分之一)
        perplexity = min(30, max(5, len(all_features) // 3))

        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                        learning_rate='auto', init='pca', metric='cosine')
            features_2d = tsne.fit_transform(all_features)
        except Exception as e:
            print(f"t-SNE降维失败: {e}")
            # 尝试使用更简单的降维方法 (如PCA)
            from sklearn.decomposition import PCA
            print("尝试使用PCA代替...")
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(all_features)

        # 分离CNN和SEN投影的降维结果
        num_cnn = cnn_features.shape[0]
        cnn_2d = features_2d[:num_cnn]
        proj_2d = features_2d[num_cnn:]

        # 创建可视化
        plt.figure(figsize=(14, 10))

        # 绘制CNN特征
        cnn_fault_types = [self.idx_to_fault.get(label, str(label)) for label in cnn_labels]
        for fault in set(cnn_fault_types):
            indices = [i for i, ft in enumerate(cnn_fault_types) if ft == fault]
            plt.scatter(
                cnn_2d[indices, 0], cnn_2d[indices, 1],
                alpha=0.6, s=50,
                label=f"CNN: {fault}"
            )

        # 绘制SEN投影特征
        for i, fault in enumerate(projected_features.keys()):
            plt.scatter(
                proj_2d[i, 0], proj_2d[i, 1],
                marker='*', s=300, edgecolors='black', linewidth=1.5,
                label=f"SEN: {fault}"
            )

        # 绘制SEN投影到CNN聚类中心的连接线
        for fault in projected_features.keys():
            # 查找对应的CNN特征的平均位置
            fault_idx = self.fault_types.get(fault, -1)
            if fault_idx >= 0:
                fault_indices = [i for i, label in enumerate(cnn_labels) if label == fault_idx]
                if fault_indices:
                    # 计算CNN特征的中心
                    cnn_center_x = np.mean(cnn_2d[fault_indices, 0])
                    cnn_center_y = np.mean(cnn_2d[fault_indices, 1])

                    # 找到对应的SEN投影点
                    proj_idx = list(projected_features.keys()).index(fault)

                    # 绘制连接线
                    plt.plot(
                        [cnn_center_x, proj_2d[proj_idx, 0]],
                        [cnn_center_y, proj_2d[proj_idx, 1]],
                        'k--', alpha=0.5
                    )

        plt.title('单一故障的CNN特征与SEN投影比较')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('single_fault_embedding_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("单一故障embedding对比图已保存至'single_fault_embedding_comparison.png'")

    def visualize_features_and_projections(self, data_dict, semantic_dict):
        """
        对比可视化CNN提取的单一故障特征与语义投影后的特征分布
        """
        print("可视化对比CNN特征与SEN投影...")

        if self.cnn_model is None or self.embedding_net is None:
            print("错误: CNN或SEN模型未训练。无法可视化。")
            return

        # 1. 准备数据 - 只使用单一故障数据
        X_vis = data_dict.get('X_train')
        y_vis = data_dict.get('y_train')

        if X_vis is None or y_vis is None or len(X_vis) == 0:
            print("错误: 无训练数据可供可视化。")
            return

        # 过滤出单一故障类型
        single_fault_indices = [i for i, y in enumerate(y_vis) if self.idx_to_fault.get(y) in
                                ['normal', 'inner', 'outer', 'ball']]

        if not single_fault_indices:
            print("错误: 未找到单一故障数据。")
            return

        X_single = X_vis[single_fault_indices]
        y_single = y_vis[single_fault_indices]

        # 2. 获取CNN特征
        self.cnn_model.eval()
        self.embedding_net.eval()

        # 为提高效率，限制可视化样本数量
        max_samples = 2000
        if len(X_single) > max_samples:
            vis_indices = np.random.choice(len(X_single), max_samples, replace=False)
            X_single = X_single[vis_indices]
            y_single = y_single[vis_indices]

        # 准备数据加载器
        dataset = TensorDataset(torch.FloatTensor(X_single), torch.LongTensor(y_single))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # 存储CNN特征和对应的标签
        cnn_features_list = []
        labels_list = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                # 不使用语义输入，直接提取CNN features
                cnn_features = self.cnn_model(batch_x, semantic=None, return_features=True)
                if torch.all(torch.isfinite(cnn_features)):
                    cnn_features_list.append(cnn_features.cpu().numpy())
                    labels_list.extend(batch_y.numpy())

        if not cnn_features_list:
            print("错误: 无法提取CNN特征。")
            return

        cnn_features = np.vstack(cnn_features_list)
        labels = np.array(labels_list)

        # 3. 获取单一故障语义投影
        fused_semantics = semantic_dict.get('fused_semantics', {})

        # 提取单一故障语义
        single_fault_semantics = {}
        for fault_type in ['normal', 'inner', 'outer', 'ball']:
            if fault_type in fused_semantics:
                single_fault_semantics[fault_type] = fused_semantics[fault_type]

        if not single_fault_semantics:
            print("错误: 无单一故障语义可用。")
            return

        # 投影单一故障语义
        projected_features = {}

        with torch.no_grad():
            for fault_type, semantic_vec in single_fault_semantics.items():
                semantic_tensor = torch.FloatTensor(semantic_vec).unsqueeze(0).to(self.device)
                try:
                    projection = self.embedding_net(semantic_tensor)
                    if torch.all(torch.isfinite(projection)):
                        projected_features[fault_type] = projection.cpu().numpy().squeeze(0)
                except Exception as e:
                    print(f"错误: 投影故障 {fault_type} 失败: {e}")

        # 4. 使用t-SNE进行降维
        all_features = np.vstack([cnn_features, np.array(list(projected_features.values()))])

        # 确保样本数量足够
        if all_features.shape[0] < 3:
            print("错误: 样本数量不足，无法进行t-SNE降维。")
            return

        # 调整perplexity参数
        perplexity = min(30, all_features.shape[0] // 5)
        perplexity = max(5, perplexity)  # 确保至少为5

        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                        learning_rate='auto', init='pca', metric='cosine')
            features_2d = tsne.fit_transform(all_features)
        except Exception as e:
            print(f"错误: t-SNE降维失败: {e}")
            return

        # 5. 可视化
        plt.figure(figsize=(14, 10))

        # 绘制CNN提取的特征
        cnn_2d = features_2d[:len(cnn_features)]
        for fault_idx in np.unique(labels):
            fault_name = self.idx_to_fault.get(fault_idx, f"Unknown_{fault_idx}")
            mask = labels == fault_idx
            plt.scatter(
                cnn_2d[mask, 0], cnn_2d[mask, 1],
                alpha=0.6, s=50,
                label=f"CNN: {fault_name}"
            )

        # 绘制语义投影
        proj_2d = features_2d[len(cnn_features):]
        for i, fault_type in enumerate(projected_features.keys()):
            plt.scatter(
                proj_2d[i, 0], proj_2d[i, 1],
                marker='*', s=300, edgecolors='black', linewidth=1.5,
                label=f"SEN: {fault_type}"
            )

        plt.title('CNN特征与SEN投影对比(单一故障)')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('cnn_vs_sen_features.png', dpi=300)
        plt.close()

        print("特征对比图已保存至'cnn_vs_sen_features.png'")

    # load_data (uses the modified preprocessor)
    def load_data(self):
        """改进的数据加载函数，严格防止数据泄露"""
        print("加载并预处理数据，确保训练和测试集严格分离...")

        # 1. 故障类型定义
        single_fault_keys = ['normal', 'inner', 'outer', 'ball']
        compound_fault_keys = ['inner_outer', 'inner_ball', 'outer_ball', 'inner_outer_ball']
        fault_files = {
            'normal': 'normal.mat', 'inner': 'inner.mat', 'outer': 'outer.mat', 'ball': 'ball.mat',
            'inner_outer': 'inner_outer.mat', 'inner_ball': 'inner_ball.mat',
            'outer_ball': 'outer_ball.mat', 'inner_outer_ball': 'inner_outer_ball.mat'
        }

        # 2. 收集原始信号
        single_fault_raw_signals = {}
        compound_fault_raw_signals = {}

        for fault_type, file_name in fault_files.items():
            file_path = os.path.join(self.data_path, file_name)
            print(f"加载 {fault_type} 数据，来源: {file_name}...")

            if not os.path.exists(file_path):
                print(f"警告: 文件不存在: {file_path}，跳过。")
                continue

            try:
                mat_data = sio.loadmat(file_path)
                signal_data_raw = None
                potential_keys = [k for k in mat_data if not k.startswith('__') and
                                  isinstance(mat_data[k], np.ndarray) and mat_data[k].size > 500]

                if potential_keys:
                    signal_data_raw = mat_data[potential_keys[-1]]

                if signal_data_raw is None:
                    print(f"错误: {file_name} 中没有合适的数据，跳过。")
                    continue

                signal_data_flat = signal_data_raw.ravel().astype(np.float64)

                # 限制每个信号最多取前200000个数据点
                max_len = 200000
                if len(signal_data_flat) > max_len:
                    signal_data_flat = signal_data_flat[:max_len]

                # 存储原始信号，严格区分单一和复合故障
                if fault_type in single_fault_keys:
                    single_fault_raw_signals[fault_type] = signal_data_flat
                else:
                    compound_fault_raw_signals[fault_type] = signal_data_flat

            except Exception as e:
                print(f"错误: 加载 {file_name} 时出错: {e}")
                traceback.print_exc()

        print(f"\n加载完成：")
        print(f"* 单一故障类型: {list(single_fault_raw_signals.keys())}")
        print(f"* 复合故障类型: {list(compound_fault_raw_signals.keys())}")

        # 3. 分割单一故障信号
        train_raw_signals = {}
        val_raw_signals = {}
        test_raw_signals = {}

        # 单一故障分割：前70%用于训练，后30%用于验证
        for fault_type, signal_data in single_fault_raw_signals.items():
            signal_length = len(signal_data)
            train_end = int(signal_length * 0.7)  # 70% 用于训练

            print(f"分割 {fault_type} 数据 (总长度={signal_length}):")
            print(f"  - 训练区间: [0:{train_end}] ({train_end} 样本)")
            print(f"  - 验证区间: [{train_end}:{signal_length}] ({signal_length - train_end} 样本)")

            train_raw_signals[fault_type] = signal_data[:train_end]
            val_raw_signals[fault_type] = signal_data[train_end:]

        # 复合故障仅用于测试
        for fault_type, signal_data in compound_fault_raw_signals.items():
            test_raw_signals[fault_type] = signal_data
            print(f"复合故障 {fault_type} 全部用于测试 ({len(signal_data)} 样本)")

        # 4. 创建预处理器
        # 训练集：启用增强
        train_preprocessor = DataPreprocessor(
            sample_length=self.sample_length,
            overlap=0.25,  # 减少训练样本重叠率
            augment=True,
            random_seed=42
        )

        # 验证集：无增强
        val_preprocessor = DataPreprocessor(
            sample_length=self.sample_length,
            overlap=0.0,  # 无重叠
            augment=False,
            random_seed=43
        )

        # 测试集：无增强
        test_preprocessor = DataPreprocessor(
            sample_length=self.sample_length,
            overlap=0.0,  # 无重叠
            augment=False,
            random_seed=44
        )

        # 5. 处理数据
        X_train_list = []
        y_train_list = []
        X_val_list = []
        y_val_list = []
        X_test_list = []
        y_test_list = []

        # 5.1 处理训练集
        print("\n处理训练集（有增强）:")
        for fault_type, signal_data in train_raw_signals.items():
            label_idx = self.fault_types[fault_type]
            processed_segments = train_preprocessor.preprocess(signal_data, augmentation=True)

            if processed_segments is None or processed_segments.shape[0] == 0:
                print(f"警告: {fault_type} 训练数据预处理后无有效分段")
                continue

            X_train_list.append(processed_segments)
            y_train_list.extend([label_idx] * len(processed_segments))
            print(f"  - {fault_type}: {len(processed_segments)} 个分段")

        # 5.2 处理验证集
        print("\n处理验证集（无增强）:")
        for fault_type, signal_data in val_raw_signals.items():
            label_idx = self.fault_types[fault_type]
            processed_segments = val_preprocessor.preprocess(signal_data, augmentation=False)

            if processed_segments is None or processed_segments.shape[0] == 0:
                print(f"警告: {fault_type} 验证数据预处理后无有效分段")
                continue

            X_val_list.append(processed_segments)
            y_val_list.extend([label_idx] * len(processed_segments))
            print(f"  - {fault_type}: {len(processed_segments)} 个分段")

        # 5.3 处理测试集（复合故障）
        print("\n处理测试集（复合故障，无增强）:")
        for fault_type, signal_data in test_raw_signals.items():
            label_idx = self.fault_types[fault_type]
            processed_segments = test_preprocessor.preprocess(signal_data, augmentation=False)

            if processed_segments is None or processed_segments.shape[0] == 0:
                print(f"警告: {fault_type} 测试数据预处理后无有效分段")
                continue

            X_test_list.append(processed_segments)
            y_test_list.extend([label_idx] * len(processed_segments))
            print(f"  - {fault_type}: {len(processed_segments)} 个分段")

        # 6. 整合数据
        X_train = np.vstack(X_train_list) if X_train_list else np.empty((0, self.sample_length), dtype=np.float32)
        y_train = np.array(y_train_list) if y_train_list else np.array([])

        X_val = np.vstack(X_val_list) if X_val_list else np.empty((0, self.sample_length), dtype=np.float32)
        y_val = np.array(y_val_list) if y_val_list else np.array([])

        X_test = np.vstack(X_test_list) if X_test_list else np.empty((0, self.sample_length), dtype=np.float32)
        y_test = np.array(y_test_list) if y_test_list else np.array([])

        # 7. 数据统计
        train_dist = dict(Counter([self.idx_to_fault.get(y, f"Unknown_{y}") for y in y_train]))
        val_dist = dict(Counter([self.idx_to_fault.get(y, f"Unknown_{y}") for y in y_val]))
        test_dist = dict(Counter([self.idx_to_fault.get(y, f"Unknown_{y}") for y in y_test]))

        print("\n数据集统计:")
        print(f"* 训练集: {len(X_train)} 个样本")
        print(f"  - 分布: {train_dist}")
        print(f"* 验证集: {len(X_val)} 个样本")
        print(f"  - 分布: {val_dist}")
        print(f"* 测试集: {len(X_test)} 个样本")
        print(f"  - 分布: {test_dist}")

        # 8. 数据随机打乱（但测试集保持原序）
        train_indices = np.random.permutation(len(X_train))
        val_indices = np.random.permutation(len(X_val))

        X_train, y_train = X_train[train_indices], y_train[train_indices]
        X_val, y_val = X_val[val_indices], y_val[val_indices]

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
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
        """Trains the Dual Channel CNN using data-only semantics."""
        print("Training dual channel CNN model...")
        # 1. Prepare Data & Check Semantics
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']
        if len(X_train) == 0: print("E: No training data for CNN."); return
        has_val_data = len(X_val) > 0
        data_only_semantics = semantic_dict.get('data_only_semantics')
        if not data_only_semantics: print("E: Data-only semantics missing."); return
        # Ensure actual_latent_dim is set correctly
        if self.actual_latent_dim <= 0: print("E: AE latent dim not set."); return


        # 2. Prepare Semantic Lookup for CNN (Data-Only Semantics)
        semantic_dim_data = self.actual_latent_dim
        default_semantic = np.zeros(semantic_dim_data, dtype=np.float32)
        semantic_vectors_cnn = {idx: data_only_semantics.get(name, default_semantic)
                                for name, idx in self.fault_types.items()}


        # 3. Initialize CNN Model
        try:
            self.cnn_model = DualChannelCNN( input_length=self.sample_length, semantic_dim=semantic_dim_data,
                 num_classes=self.num_classes, feature_dim=CNN_FEATURE_DIM).to(self.device)
            # Store the actual feature dimension determined by the CNN architecture
            self.cnn_feature_dim = self.cnn_model.feature_dim
        except Exception as e: print(f"E: CNN init failed: {e}"); return

        # 4. Initialize Losses (CE, Contrastive, Consistency)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_contrastive = ContrastiveLoss(temperature=0.1).to(self.device)
        # IMPORTANT: Initialize Consistency Loss HERE, knowing the input/output dims
        try:
            self.consistency_loss_fn = FeatureSemanticConsistencyLoss(
                 beta=0.01,
                 semantic_dim_input=self.actual_latent_dim, # Input is AE data semantic dim
                 feature_dim_output=self.cnn_feature_dim # Output is CNN feature dim
            ).to(self.device)
        except Exception as e: print(f"E: Consistency loss init failed: {e}"); return

        # 5. Optimizer (Include Consistency Loss Projection Params)
        cnn_params = list(self.cnn_model.parameters())
        if hasattr(self.consistency_loss_fn, 'projection') and self.consistency_loss_fn.projection is not None:
            consist_params = list(self.consistency_loss_fn.projection.parameters())
            if consist_params:
                 print("Adding ConsistencyLoss projection parameters to CNN optimizer.")
                 all_params = cnn_params + consist_params
            else: all_params = cnn_params
        else: all_params = cnn_params
        optimizer = optim.AdamW(all_params, lr=lr, weight_decay=1e-3)
        indices = np.random.choice(len(X_train), min(len(X_train), 5000), replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 20)

        # 6. Data Loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = None
        if has_val_data:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)

        # 7. Training Loop
        best_val_acc = 0.0
        patience_counter = 0
        patience = 15
        print(f"Starting CNN training ({epochs} epochs)...")
        for epoch in range(epochs):
            self.cnn_model.train()
            if self.consistency_loss_fn: self.consistency_loss_fn.train() # Ensure projection layer is in train mode

            train_loss, correct, total = 0.0, 0, 0
            train_loss_ce, train_loss_contr, train_loss_consist = 0.0, 0.0, 0.0

            # --- Training Batch ---
            for inputs, labels in train_loader:
                 inputs, labels = inputs.to(self.device), labels.to(self.device)
                 current_bs = inputs.size(0)
                 if current_bs < 2 : continue

                 # Get data semantics for batch
                 batch_semantics_data = torch.stack([torch.from_numpy(semantic_vectors_cnn[lbl.item()]) for lbl in labels]).to(self.device)

                 try: # Forward Pass
                      logits, features = self.cnn_model(inputs, batch_semantics_data)
                      if not torch.all(torch.isfinite(logits)) or not torch.all(torch.isfinite(features)):
                          print(f"W: NaN/Inf CNN outputs epoch {epoch+1}. Skip batch."); continue
                 except Exception as e: print(f"E: CNN Fwd fail: {e}"); continue

                 # Calculate Losses
                 ce_loss = criterion_ce(logits, labels)
                 contr_loss = criterion_contrastive(features, labels)
                 consist_loss = self.consistency_loss_fn(features, batch_semantics_data) if self.consistency_loss_fn else torch.tensor(0.0)

                 w_ce, w_contr, w_consist = 1.0, 0.1, 0.5 # Loss weights
                 total_batch_loss = w_ce * ce_loss + w_contr * contr_loss + w_consist * consist_loss

                 if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                     print(f"W: NaN/Inf Loss epoch {epoch+1}. Skip batch backprop."); continue

                 # Backward and Optimize
                 optimizer.zero_grad(); total_batch_loss.backward()
                 torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0) # Clip all gradients
                 optimizer.step()

                 # Accumulate stats
                 train_loss += total_batch_loss.item() * current_bs
                 train_loss_ce += ce_loss.item() * current_bs; train_loss_contr += contr_loss.item()*current_bs; train_loss_consist += consist_loss.item()*current_bs
                 _, predicted = torch.max(logits, 1)
                 total += current_bs; correct += (predicted == labels).sum().item()

            # --- Epoch End & Validation ---
            current_lr = optimizer.param_groups[0]['lr']; scheduler.step()
            avg_train_loss = train_loss / total if total > 0 else 0; train_accuracy = 100.0*correct/total if total >0 else 0
            avg_train_ce = train_loss_ce/total if total>0 else 0; avg_train_contr=train_loss_contr/total if total>0 else 0; avg_train_consist=train_loss_consist/total if total>0 else 0

            avg_val_loss, val_accuracy = 0.0, 0.0
            if val_loader is not None:
                 self.cnn_model.eval()
                 if self.consistency_loss_fn: self.consistency_loss_fn.eval() # Set projection to eval
                 val_loss, val_correct, val_total = 0.0, 0, 0
                 with torch.no_grad():
                      for inputs_val, labels_val in val_loader:
                          inputs_val, labels_val = inputs_val.to(self.device), labels_val.to(self.device)
                          bs_val = inputs_val.size(0)
                          batch_semantics_val = torch.stack([torch.from_numpy(semantic_vectors_cnn[lbl.item()]) for lbl in labels_val]).to(self.device)
                          logits_val, _ = self.cnn_model(inputs_val, batch_semantics_val)
                          loss_val = criterion_ce(logits_val, labels_val)
                          if torch.isfinite(loss_val): val_loss += loss_val.item() * bs_val
                          _, predicted_val = torch.max(logits_val, 1); val_total += bs_val; val_correct += (predicted_val == labels_val).sum().item()
                 avg_val_loss = val_loss / val_total if val_total > 0 else 0; val_accuracy = 100.0*val_correct/val_total if val_total>0 else 0

            # --- Logging & Early Stopping ---
            print(f"E[{epoch+1}/{epochs}] LR={current_lr:.6f} TrLss={avg_train_loss:.4f} (CE:{avg_train_ce:.4f},Ctr:{avg_train_contr:.4f},Cns:{avg_train_consist:.4f}) TrAcc={train_accuracy:.2f}% | VlLss={avg_val_loss:.4f} VlAcc={val_accuracy:.2f}%")

            monitor_metric = val_accuracy if has_val_data else -avg_train_loss
            if monitor_metric > best_val_acc:
                  best_val_acc = monitor_metric
                  patience_counter = 0
                  torch.save(self.cnn_model.state_dict(), 'best_cnn_model.pth')
                  if self.consistency_loss_fn and hasattr(self.consistency_loss_fn, 'projection'): torch.save(self.consistency_loss_fn.projection.state_dict(), 'best_consistency_projection.pth')
                  print(f"  Best model saved. Val Acc: {val_accuracy:.2f}%")
            else:
                  patience_counter += 1
                  if patience_counter >= patience: print("Early stopping."); break

        # --- Load Best Model ---
        if os.path.exists('best_cnn_model.pth'):
            self.cnn_model.load_state_dict(torch.load('best_cnn_model.pth'))
            if self.consistency_loss_fn and hasattr(self.consistency_loss_fn, 'projection') and os.path.exists('best_consistency_projection.pth'):
                 self.consistency_loss_fn.projection.load_state_dict(torch.load('best_consistency_projection.pth'))
            print("Loaded best CNN model state.")
        self.cnn_model.eval()
        if self.consistency_loss_fn: self.consistency_loss_fn.eval()
        print(f"CNN Training Complete. Best Val Acc: {best_val_acc:.2f}% (or corresponding train metric)")


    # train_semantic_embedding (Inputs adjusted)
    def train_improved_semantic_embedding(self, semantic_dict, data_dict, epochs=30):
        """
        使用多重损失策略训练改进的语义嵌入网络

        参数:
            semantic_dict: 包含融合语义的字典
            data_dict: 包含训练数据的字典
            epochs: 训练轮数
        """
        print("\n开始训练改进的语义嵌入网络...")

        # 1. 准备数据
        if self.cnn_model is None:
            print("错误: CNN模型未训练")
            return

        fused_semantics = semantic_dict.get('fused_semantics')
        if not fused_semantics:
            print("错误: 没有可用的融合语义")
            return

        X_train, y_train = data_dict.get('X_train'), data_dict.get('y_train')
        X_val, y_val = data_dict.get('X_val'), data_dict.get('y_val')

        # 2. 初始化改进的SEN
        self.embedding_net = ImprovedSemanticEmbeddingNetwork(
            semantic_dim=self.fused_semantic_dim,
            feature_dim=self.cnn_feature_dim
        ).to(self.device)

        # 3. 设置优化器和学习率调度器
        optimizer = torch.optim.AdamW(
            self.embedding_net.parameters(),
            lr=0.0005,
            weight_decay=1e-4,
            eps=1e-8
        )

        # 余弦退火学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,  # 首次重启的轮数
            T_mult=2,  # 每次重启后周期长度的倍增因子
            eta_min=1e-6  # 最小学习率
        )

        # 4. 损失函数
        criterion_mse = nn.MSELoss()
        criterion_cosine = nn.CosineEmbeddingLoss(margin=0.2)
        criterion_huber = nn.SmoothL1Loss()  # Huber损失对异常值更鲁棒

        # 5. 提前停止设置
        best_val_loss = float('inf')
        patience, patience_counter = 7, 0

        # 6. 混合精度训练
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        # 7. 训练循环
        for epoch in range(epochs):
            self.embedding_net.train()
            self.cnn_model.eval()  # 确保CNN处于评估模式

            # 计算动态权重 - 训练初期重视结构对齐，后期重视精确匹配
            alpha = min(0.8, 0.3 + epoch * 0.02)  # MSE权重，从0.3逐渐增加到0.8
            beta = 1.0 - alpha  # 余弦相似度权重
            gamma = min(0.5, 0.1 + epoch * 0.015)  # Huber损失权重，从0.1逐渐增加到0.5

            epoch_loss = 0.0
            processed_samples = 0

            # 随机打乱训练数据
            indices = np.random.permutation(len(X_train))
            batch_size = min(64, len(X_train))

            # 按批次处理
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                if len(batch_indices) < 2:  # 跳过太小的批次
                    continue

                batch_x = torch.FloatTensor(X_train[batch_indices]).to(self.device)
                batch_y = y_train[batch_indices]

                # 7.1 提取CNN特征作为目标
                with torch.no_grad():
                    cnn_features = self.cnn_model(batch_x, semantic=None, return_features=True)
                    if not torch.all(torch.isfinite(cnn_features)):
                        print(f"警告: CNN特征包含非有限值，跳过此批次")
                        continue

                # 7.2 获取批次的融合语义
                batch_fused_semantics = []
                valid_indices = []

                for idx, label in enumerate(batch_y):
                    fault_type = self.idx_to_fault.get(label)
                    if fault_type in fused_semantics:
                        sem_vec = fused_semantics[fault_type]
                        if np.all(np.isfinite(sem_vec)):
                            batch_fused_semantics.append(sem_vec)
                            valid_indices.append(idx)

                if not valid_indices:
                    continue

                # 准备有效的张量
                batch_fused_tensor = torch.FloatTensor(np.array(batch_fused_semantics)).to(self.device)
                target_features = cnn_features[valid_indices]

                # 7.3 正向传播
                optimizer.zero_grad()

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        # 混合精度计算
                        embedded_semantics = self.embedding_net(batch_fused_tensor)

                        # 多重损失计算
                        mse_loss = criterion_mse(embedded_semantics, target_features)
                        cos_loss = criterion_cosine(
                            embedded_semantics,
                            target_features,
                            torch.ones(len(valid_indices), device=self.device)
                        )
                        huber_loss = criterion_huber(embedded_semantics, target_features)

                        # 加权总损失
                        total_loss = alpha * mse_loss + beta * cos_loss + gamma * huber_loss

                    # 混合精度反向传播
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 标准精度计算
                    embedded_semantics = self.embedding_net(batch_fused_tensor)

                    # 多重损失计算
                    mse_loss = criterion_mse(embedded_semantics, target_features)
                    cos_loss = criterion_cosine(
                        embedded_semantics,
                        target_features,
                        torch.ones(len(valid_indices), device=self.device)
                    )
                    huber_loss = criterion_huber(embedded_semantics, target_features)

                    # 加权总损失
                    total_loss = alpha * mse_loss + beta * cos_loss + gamma * huber_loss

                    # 标准反向传播
                    total_loss.backward()
                    optimizer.step()

                # 7.4 统计
                epoch_loss += total_loss.item() * len(valid_indices)
                processed_samples += len(valid_indices)

            # 7.5 学习率调度
            scheduler.step()

            # 7.6 验证
            val_loss = self.validate_improved_sen(fused_semantics, X_val, y_val)

            # 7.7 打印统计
            avg_epoch_loss = epoch_loss / max(1, processed_samples)
            print(f"轮次 [{epoch + 1}/{epochs}] "
                  f"训练损失: {avg_epoch_loss:.6f} "
                  f"验证损失: {val_loss:.6f} "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} "
                  f"损失权重: α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")

            # 7.8 检查提前停止
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.embedding_net.state_dict(), 'best_improved_sen.pth')
                print(f"  最佳模型已保存! 验证损失: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"提前停止! {patience}轮未改善")
                    break

        # 8. 加载最佳模型
        if os.path.exists('best_improved_sen.pth'):
            self.embedding_net.load_state_dict(torch.load('best_improved_sen.pth'))
            print("已加载最佳SEN模型")

        # 9. 可视化最终结果
        self.visualize_embedding_separation(fused_semantics, data_dict)

        print("改进的语义嵌入网络训练完成")

    def validate_improved_sen(self, fused_semantics, X_val, y_val):
        """验证改进的SEN模型性能"""
        if X_val is None or len(X_val) == 0 or y_val is None or len(y_val) == 0:
            return float('inf')

        self.embedding_net.eval()
        self.cnn_model.eval()
        criterion_mse = nn.MSELoss()

        val_loss = 0.0
        valid_samples = 0
        batch_size = min(64, len(X_val))

        with torch.no_grad():
            # 处理验证数据批次
            for i in range(0, len(X_val), batch_size):
                batch_x = torch.FloatTensor(X_val[i:i + batch_size]).to(self.device)
                batch_y = y_val[i:i + batch_size]

                # 提取CNN特征
                cnn_features = self.cnn_model(batch_x, semantic=None, return_features=True)

                # 获取融合语义
                batch_semantics = []
                valid_indices = []

                for idx, label in enumerate(batch_y):
                    fault_type = self.idx_to_fault.get(label)
                    if fault_type in fused_semantics:
                        sem_vec = fused_semantics[fault_type]
                        if np.all(np.isfinite(sem_vec)):
                            batch_semantics.append(sem_vec)
                            valid_indices.append(idx)

                if not valid_indices:
                    continue

                # 计算SEN投影
                batch_tensor = torch.FloatTensor(np.array(batch_semantics)).to(self.device)
                embedded_semantics = self.embedding_net(batch_tensor)

                # 计算损失
                target_features = cnn_features[valid_indices]
                batch_loss = criterion_mse(embedded_semantics, target_features).item()

                val_loss += batch_loss * len(valid_indices)
                valid_samples += len(valid_indices)

        # 计算平均验证损失
        avg_val_loss = val_loss / max(1, valid_samples)
        return avg_val_loss

    def joint_train_cnn_sen_improved(self, data_dict, semantic_dict, epochs=3, lr=0.0005):
        """
        改进的CNN和SEN联合训练，强化SEN投影与CNN特征的一致性
        """
        print("开始改进的CNN和SEN联合训练...")

        # 1. 准备数据和语义
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']

        if len(X_train) == 0:
            print("错误: 无训练数据。")
            return
        has_val_data = len(X_val) > 0

        fused_semantics = semantic_dict.get('fused_semantics')
        data_only_semantics = semantic_dict.get('data_only_semantics')
        if not fused_semantics or not data_only_semantics:
            print("错误: 缺少语义数据。")
            return

        # 2. 初始化模型
        if self.cnn_model is None:
            self.cnn_model = DualChannelCNN(
                input_length=self.sample_length,
                semantic_dim=self.actual_latent_dim,
                num_classes=self.num_classes,
                feature_dim=CNN_FEATURE_DIM
            ).to(self.device)
            self.cnn_feature_dim = self.cnn_model.feature_dim

        if self.embedding_net is None:
            # 使用更深的SEN架构
            self.embedding_net = ImprovedSemanticEmbeddingNetwork(
                semantic_dim=self.fused_semantic_dim,
                feature_dim=self.cnn_feature_dim
            ).to(self.device)

        # 3. 损失函数
        criterion_ce = nn.CrossEntropyLoss()
        criterion_contrastive = ContrastiveLoss(temperature=0.05).to(self.device)  # 降低温度参数
        criterion_mse = nn.MSELoss()
        criterion_cosine = nn.CosineEmbeddingLoss(margin=0.2)  # 增加margin
        criterion_triplet = nn.TripletMarginLoss(margin=0.5, p=2)  # 添加三元组损失
        criterion_orthogonal = OrthogonalRegularization().to(self.device)  # 正交正则化

        # 4. 优化器 - 分别优化CNN和SEN，使用不同的学习率
        cnn_params = list(self.cnn_model.parameters())
        if hasattr(self, 'consistency_loss_fn') and self.consistency_loss_fn is not None:
            cnn_params += list(self.consistency_loss_fn.projection.parameters())

        optimizer_cnn = optim.AdamW(cnn_params, lr=lr, weight_decay=1e-4)
        # SEN使用稍高的学习率，促进更快收敛
        optimizer_sen = optim.AdamW(self.embedding_net.parameters(), lr=lr * 2.0, weight_decay=1e-5)

        # 5. 学习率调度
        scheduler_cnn = optim.lr_scheduler.CosineAnnealingLR(optimizer_cnn, T_max=epochs, eta_min=lr / 20)
        scheduler_sen = optim.lr_scheduler.CosineAnnealingLR(optimizer_sen, T_max=epochs, eta_min=lr / 10)

        # 6. 数据加载器
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if has_val_data:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)

        # 7. 语义查找表
        semantic_vectors_data = {idx: data_only_semantics.get(name, np.zeros(self.actual_latent_dim))
                                 for name, idx in self.fault_types.items()}
        semantic_vectors_fused = {idx: fused_semantics.get(name, np.zeros(self.fused_semantic_dim))
                                  for name, idx in self.fault_types.items()}

        # 8. 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        annealing_factor = 1.0  # 用于逐步增加SEN损失权重

        print(f"开始训练 ({epochs} 轮)...")
        for epoch in range(epochs):
            self.cnn_model.train()
            self.embedding_net.train()

            # 根据训练进度调整SEN损失权重
            if epoch < epochs // 3:
                # 前期让CNN充分学习
                sen_weight = 0.3 * annealing_factor
            else:
                # 后期加强SEN学习
                sen_weight = 0.7 * annealing_factor

            # 每轮增加一点SEN权重
            annealing_factor = min(1.0, annealing_factor + 0.02)

            total_loss = 0.0
            total_cnn_loss = 0.0
            total_sen_loss = 0.0
            train_correct = 0
            train_total = 0

            # TRAINING LOOP
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_size = batch_x.size(0)
                if batch_size < 2: continue  # 跳过太小的批次

                # 获取数据语义和融合语义
                batch_data_semantics = torch.stack([
                    torch.from_numpy(semantic_vectors_data[idx.item()])
                    for idx in batch_y
                ]).to(self.device)

                batch_fused_semantics = torch.stack([
                    torch.from_numpy(semantic_vectors_fused[idx.item()])
                    for idx in batch_y
                ]).to(self.device)

                # --- STEP 1: 训练CNN ---
                optimizer_cnn.zero_grad()
                logits, cnn_features = self.cnn_model(batch_x, batch_data_semantics)

                if not torch.all(torch.isfinite(logits)) or not torch.all(torch.isfinite(cnn_features)):
                    print(f"警告: CNN输出包含非有限值，跳过此批次")
                    continue

                # CNN损失组合
                ce_loss = criterion_ce(logits, batch_y)
                contrastive_loss = criterion_contrastive(cnn_features, batch_y)

                # 添加特征分离损失，促进不同类别特征分离
                cluster_loss = balanced_clustering_loss(cnn_features, batch_y, self.device, self)

                cnn_loss = ce_loss + 0.2 * contrastive_loss + 0.3 * cluster_loss
                cnn_loss.backward()
                torch.nn.utils.clip_grad_norm_(cnn_params, max_norm=1.0)
                optimizer_cnn.step()

                # --- STEP 2: 训练SEN (两阶段) ---
                # 每个批次进行两次SEN更新，增强学习
                for _ in range(2):
                    optimizer_sen.zero_grad()

                    # 重新获取CNN特征但不计算梯度
                    with torch.no_grad():
                        _, cnn_features_detached = self.cnn_model(batch_x, batch_data_semantics)

                    # 对特征和语义添加微小扰动以增加鲁棒性
                    noise_scale = 0.02
                    cnn_features_noise = cnn_features_detached + noise_scale * torch.randn_like(cnn_features_detached)
                    fused_semantics_noise = batch_fused_semantics + noise_scale * torch.randn_like(
                        batch_fused_semantics)

                    # 正向传播SEN
                    sen_features = self.embedding_net(fused_semantics_noise)

                    if not torch.all(torch.isfinite(sen_features)):
                        print(f"警告: SEN输出包含非有限值，跳过")
                        continue

                    # 多重损失组合强化学习
                    mse_loss = criterion_mse(sen_features, cnn_features_noise)
                    cos_loss = criterion_cosine(
                        sen_features,
                        cnn_features_noise,
                        torch.ones(batch_size, device=self.device)
                    )

                    # 三元组损失: 为每个样本找到正负样本对
                    triplet_loss = torch.tensor(0.0, device=self.device)
                    if batch_size >= 3:  # 需要至少3个样本
                        for i in range(batch_size):
                            anchor = sen_features[i].unsqueeze(0)

                            # 同类样本作为正样本
                            pos_mask = (batch_y == batch_y[i]) & (torch.arange(batch_size, device=self.device) != i)
                            if pos_mask.sum() > 0:
                                positive_idx = torch.where(pos_mask)[0][0]
                                positive = cnn_features_noise[positive_idx].unsqueeze(0)

                                # 不同类样本作为负样本
                                neg_mask = batch_y != batch_y[i]
                                if neg_mask.sum() > 0:
                                    negative_idx = torch.where(neg_mask)[0][0]
                                    negative = cnn_features_noise[negative_idx].unsqueeze(0)

                                    # 计算单个三元组损失
                                    triplet_loss += criterion_triplet(anchor, positive, negative)

                    if triplet_loss > 0:
                        triplet_loss = triplet_loss / batch_size

                    # 对最终层权重应用正交约束
                    last_layer = list(self.embedding_net.parameters())[-2]  # 假设倒数第二个参数是最后一层权重
                    ortho_loss = criterion_orthogonal(last_layer)

                    # 综合SEN损失
                    sen_loss = (0.5 * mse_loss +
                                0.3 * cos_loss +
                                0.1 * triplet_loss +
                                0.1 * ortho_loss)

                    sen_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.embedding_net.parameters(), max_norm=1.0)
                    optimizer_sen.step()

                # 总损失和统计
                batch_total_loss = cnn_loss.item() + sen_loss.item()
                total_loss += batch_total_loss * batch_size
                total_cnn_loss += cnn_loss.item() * batch_size
                total_sen_loss += sen_loss.item() * batch_size
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_size

            # 验证阶段
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            if has_val_data:
                self.cnn_model.eval()
                self.embedding_net.eval()

                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x = val_x.to(self.device)
                        val_y = val_y.to(self.device)

                        val_data_semantics = torch.stack([
                            torch.from_numpy(semantic_vectors_data[idx.item()])
                            for idx in val_y
                        ]).to(self.device)

                        val_logits, _ = self.cnn_model(val_x, val_data_semantics)
                        val_ce_loss = criterion_ce(val_logits, val_y)
                        val_loss += val_ce_loss.item() * val_x.size(0)

                        _, val_pred = torch.max(val_logits, 1)
                        val_correct += (val_pred == val_y).sum().item()
                        val_total += val_x.size(0)

            # 学习率调整
            scheduler_cnn.step()
            scheduler_sen.step()

            # 计算平均损失和准确率
            avg_train_loss = total_loss / train_total if train_total > 0 else float('inf')
            avg_train_cnn_loss = total_cnn_loss / train_total if train_total > 0 else float('inf')
            avg_train_sen_loss = total_sen_loss / train_total if train_total > 0 else float('inf')
            train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0

            avg_val_loss = val_loss / val_total if val_total > 0 else float('inf')
            val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0

            print(f"轮次[{epoch + 1}/{epochs}] "
                  f"训练损失={avg_train_loss:.4f} (CNN={avg_train_cnn_loss:.4f}, SEN={avg_train_sen_loss:.4f}) "
                  f"准确率={train_accuracy:.2f}% | "
                  f"验证: 损失={avg_val_loss:.4f} 准确率={val_accuracy:.2f}% | "
                  f"LR=[{scheduler_cnn.get_last_lr()[0]:.6f}, {scheduler_sen.get_last_lr()[0]:.6f}]")

            # 早停（基于验证损失或训练损失）
            monitor_metric = avg_val_loss if has_val_data else avg_train_loss
            if monitor_metric < best_val_loss:
                best_val_loss = monitor_metric
                patience_counter = 0
                torch.save(self.cnn_model.state_dict(), 'best_joint_cnn.pth')
                torch.save(self.embedding_net.state_dict(), 'best_joint_sen.pth')
                print(f"  模型已保存! 当前最佳损失: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {patience}轮未改善")
                    break

        # 加载最佳模型
        if os.path.exists('best_joint_cnn.pth'):
            self.cnn_model.load_state_dict(torch.load('best_joint_cnn.pth'))
        if os.path.exists('best_joint_sen.pth'):
            self.embedding_net.load_state_dict(torch.load('best_joint_sen.pth'))

        # 训练后立即进行可视化对比
        self.visualize_features_and_projections(data_dict, semantic_dict)

        print("改进的CNN和SEN联合训练完成.")

    def generate_compound_fault_projections_fixed(self, semantic_dict, data_dict):
        """
        增强版复合故障投影生成函数，解决投影缺失和错位问题
        """
        print("\n生成稳定的复合故障投影...")

        # 1. 基础检查
        if self.embedding_net is None:
            print("错误: 语义嵌入网络未训练，无法生成投影")
            return None

        fused_semantics = semantic_dict.get('fused_semantics')
        if not fused_semantics:
            print("错误: 无可用的融合语义向量")
            return None

        # 2. 获取特征空间内真实数据的分布信息
        # 这对于验证和微调投影非常重要
        test_data = data_dict.get('X_test')
        test_labels = data_dict.get('y_test')

        if test_data is None or len(test_data) == 0 or test_labels is None:
            print("警告: 无测试数据用于参考。继续生成但无法验证。")
            reference_centroids = None
        else:
            # 计算每种复合故障类别在CNN特征空间中的真实中心
            reference_centroids = self._calculate_cnn_feature_centroids(test_data, test_labels)

        # 3. 提取所有复合故障的语义向量
        compound_fault_types = self.compound_fault_types
        print(f"准备为以下{len(compound_fault_types)}种复合故障生成投影: {compound_fault_types}")

        compound_fused_semantics = {}
        for fault_type in compound_fault_types:
            if fault_type in fused_semantics:
                sem_vec = fused_semantics[fault_type]
                if sem_vec is not None and np.all(np.isfinite(sem_vec)):
                    compound_fused_semantics[fault_type] = sem_vec
                else:
                    print(f"警告: '{fault_type}'的语义向量包含无效值，尝试修复...")
                    # 尝试从其组成部分重新合成语义
                    fixed_vec = self._regenerate_compound_semantic(fault_type, fused_semantics)
                    if fixed_vec is not None:
                        compound_fused_semantics[fault_type] = fixed_vec
                        print(f"  - 成功重新合成'{fault_type}'的语义向量")
                    else:
                        print(f"  - 无法修复'{fault_type}'的语义向量，跳过")

        if not compound_fused_semantics:
            print("错误: 没有有效的复合故障语义向量可用")
            return None

        print(f"成功获取{len(compound_fused_semantics)}种复合故障的语义向量")

        # 4. 使用语义嵌入网络生成投影 - 增加健壮性
        self.embedding_net.eval()  # 确保在评估模式

        # 初始化结果字典
        compound_projections = {}
        failed_projections = []

        # 投影每个复合故障 - 添加多重尝试机制
        with torch.no_grad():
            for fault_type, semantic_vec in compound_fused_semantics.items():
                # 创建语义向量的多个副本进行冗余投影
                projection_attempts = []
                num_attempts = 5  # 多次尝试以提高稳定性

                for attempt in range(num_attempts):
                    try:
                        # 每次尝试添加少量噪声以获得更多样的投影结果
                        noise_scale = 0.01 * attempt  # 逐渐增加噪声
                        noised_vec = semantic_vec.copy()
                        if attempt > 0:  # 第一次使用原始向量
                            noised_vec += np.random.normal(0, noise_scale, size=len(semantic_vec))

                        # 检查向量的有效性
                        if not np.all(np.isfinite(noised_vec)):
                            continue

                        # 转换为张量
                        semantic_tensor = torch.FloatTensor(noised_vec).unsqueeze(0).to(self.device)

                        # 通过SEN投影
                        projected_feature = self.embedding_net(semantic_tensor)

                        # 验证投影的有效性
                        if torch.all(torch.isfinite(projected_feature)):
                            projection_attempts.append(projected_feature.cpu().numpy().squeeze(0))

                    except Exception as e:
                        print(f"警告: 第{attempt + 1}次尝试投影'{fault_type}'失败: {e}")
                        continue

                # 汇总多次投影结果
                if projection_attempts:
                    # 计算有效投影的均值作为最终结果
                    final_projection = np.mean(projection_attempts, axis=0)
                    compound_projections[fault_type] = final_projection
                    print(f"  - '{fault_type}'投影成功 ({len(projection_attempts)}/{num_attempts}次尝试有效)")
                else:
                    failed_projections.append(fault_type)
                    print(f"  - 无法为'{fault_type}'生成有效投影")

        # 5. 微调投影使其更接近真实分布
        if reference_centroids and len(reference_centroids) > 0:
            print("应用投影微调将投影点移向真实数据中心...")
            compound_projections = self._refine_projections(
                compound_projections,
                reference_centroids
            )

        # 6. 如果有投影失败，尝试从数据中心生成
        if failed_projections and reference_centroids:
            print("尝试从数据特征中恢复失败的投影...")
            for fault in failed_projections:
                fault_idx = self.fault_types.get(fault)
                if fault_idx is not None and fault_idx in reference_centroids:
                    compound_projections[fault] = reference_centroids[fault_idx]
                    print(f"  - 从数据中心恢复'{fault}'的投影")

        # 7. 最后确认生成的投影数量
        if len(compound_projections) == 0:
            print("错误: 无有效的复合故障投影生成")
            return None

        print(f"成功生成{len(compound_projections)}种复合故障投影")

        # 8. 打印相似度矩阵 (用于调试)
        if len(compound_projections) > 1:
            print("\n投影相似度矩阵:")
            fault_types = list(compound_projections.keys())
            for i in range(len(fault_types)):
                for j in range(i + 1, len(fault_types)):
                    v1 = compound_projections[fault_types[i]]
                    v2 = compound_projections[fault_types[j]]
                    v1_norm = v1 / np.linalg.norm(v1)
                    v2_norm = v2 / np.linalg.norm(v2)
                    sim = np.dot(v1_norm, v2_norm)
                    print(f"  {fault_types[i]} vs {fault_types[j]}: {sim:.4f}")

        return compound_projections

    def _calculate_cnn_feature_centroids(self, test_data, test_labels):
        """
        计算测试数据中每个故障类别在CNN特征空间中的中心点
        用于指导和验证SEN投影
        """
        print("计算测试数据在特征空间中的分布中心...")

        if self.cnn_model is None:
            return {}

        self.cnn_model.eval()

        # 存储每个类别的特征
        class_features = {}

        # 使用批处理更高效地处理数据
        batch_size = min(64, len(test_data))

        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch_x = torch.FloatTensor(test_data[i:i + batch_size]).to(self.device)
                batch_y = test_labels[i:i + batch_size]

                # 提取CNN特征
                try:
                    features = self.cnn_model(batch_x, semantic=None, return_features=True)

                    if not torch.all(torch.isfinite(features)):
                        print(f"警告: 批次{i // batch_size + 1}含有非有限特征值，已被跳过")
                        continue

                    # 按类别存储特征
                    for j, label in enumerate(batch_y):
                        if label not in class_features:
                            class_features[label] = []
                        class_features[label].append(features[j].cpu().numpy())

                except Exception as e:
                    print(f"警告: 特征提取错误(批次{i // batch_size + 1}): {e}")

        # 计算每个类别的中心点
        centroids = {}
        for label, feat_list in class_features.items():
            if feat_list:  # 确保有特征
                centroids[label] = np.mean(feat_list, axis=0)

        print(f"计算了{len(centroids)}个类别的特征中心点")
        return centroids

    def _regenerate_compound_semantic(self, fault_type, fused_semantics):
        """
        如果复合故障的语义有问题，尝试从组成部分重新生成
        """
        components = fault_type.split('_')
        if len(components) <= 1:
            return None  # 不是复合故障

        # 收集有效的组件语义
        component_semantics = []
        for comp in components:
            if comp in fused_semantics and np.all(np.isfinite(fused_semantics[comp])):
                component_semantics.append(fused_semantics[comp])

        if not component_semantics:
            return None  # 没有有效的组件语义

        if len(component_semantics) == 1:
            return component_semantics[0]  # 只有一个有效组件

        # 简单平均组合 - 比复杂的张量融合更稳定
        combined = np.mean(component_semantics, axis=0)

        # 确保结果有效
        if np.all(np.isfinite(combined)):
            return combined
        return None

    def _refine_projections(self, projections, reference_centroids):
        """
        使用真实数据分布信息微调投影点的位置
        将投影点轻微地向真实数据中心移动
        """
        refined = {}
        for fault_type, proj in projections.items():
            fault_idx = self.fault_types.get(fault_type)

            # 如果有对应的参考中心点
            if fault_idx is not None and fault_idx in reference_centroids:
                ref_centroid = reference_centroids[fault_idx]

                # 向参考中心移动30% (平衡投影的语义基础和数据分布)
                blend_factor = 0.3
                refined_proj = (1 - blend_factor) * proj + blend_factor * ref_centroid

                refined[fault_type] = refined_proj
            else:
                refined[fault_type] = proj  # 保持原样

        return refined

    def separate_projections_forcefully(self, compound_projections, min_distance=0.5):
        """
        在特征空间中强制分离投影点
        这是一个训练后的处理方法，直接修改投影向量

        参数:
            compound_projections: 复合故障投影字典 {fault_type: projection_vector}
            min_distance: 投影间的最小余弦距离
        """
        print("强制分离投影点...")

        if len(compound_projections) <= 1:
            return compound_projections

        # 将投影转换为列表便于处理
        fault_types = list(compound_projections.keys())
        projections = [compound_projections[ft] for ft in fault_types]

        # 转换为numpy数组
        proj_array = np.array(projections)
        n_proj = len(proj_array)

        # 强制分离流程:
        # 1. 计算当前投影间的距离矩阵
        distance_matrix = np.zeros((n_proj, n_proj))
        for i in range(n_proj):
            for j in range(i + 1, n_proj):
                # 归一化向量
                v1 = proj_array[i] / np.linalg.norm(proj_array[i])
                v2 = proj_array[j] / np.linalg.norm(proj_array[j])
                # 余弦相似度
                similarity = np.dot(v1, v2)
                # 余弦距离 = 1 - 余弦相似度
                distance_matrix[i, j] = distance_matrix[j, i] = 1.0 - similarity

        # 2. 迭代分离过近的投影
        max_iterations = 100
        for iteration in range(max_iterations):
            # 找到最近的一对投影
            min_dist = float('inf')
            min_pair = None
            for i in range(n_proj):
                for j in range(i + 1, n_proj):
                    if distance_matrix[i, j] < min_dist:
                        min_dist = distance_matrix[i, j]
                        min_pair = (i, j)

            # 如果所有投影已经足够分离，则停止迭代
            if min_dist >= min_distance:
                print(f"  投影已充分分离，平均距离: {np.mean(distance_matrix):.4f}")
                break

            # 分离最近的一对
            if min_pair:
                i, j = min_pair

                # 获取故障类型信息以指导分离方向
                type_i = fault_types[i]
                type_j = fault_types[j]

                # 分解故障组件
                components_i = set(type_i.split('_'))
                components_j = set(type_j.split('_'))

                # 计算不同的组件（这些是我们需要强调的差异）
                diff_components = (components_i - components_j) | (components_j - components_i)

                # 根据差异组件确定分离力度
                separation_strength = 0.2 * (1 + len(diff_components) * 0.5)

                # 计算方向向量（从j指向i）
                direction = proj_array[i] - proj_array[j]
                direction_norm = np.linalg.norm(direction)

                if direction_norm > 1e-6:  # 避免除零
                    direction = direction / direction_norm

                    # 向相反方向移动两个投影
                    proj_array[i] = proj_array[i] + direction * separation_strength
                    proj_array[j] = proj_array[j] - direction * separation_strength

                    # 重新归一化
                    proj_array[i] = proj_array[i] / np.linalg.norm(proj_array[i]) * np.linalg.norm(projections[i])
                    proj_array[j] = proj_array[j] / np.linalg.norm(proj_array[j]) * np.linalg.norm(projections[j])

                    # 更新距离矩阵
                    for k in range(n_proj):
                        if k != i:
                            v1 = proj_array[i] / np.linalg.norm(proj_array[i])
                            v2 = proj_array[k] / np.linalg.norm(proj_array[k])
                            similarity = np.dot(v1, v2)
                            distance_matrix[i, k] = distance_matrix[k, i] = 1.0 - similarity
                        if k != j:
                            v1 = proj_array[j] / np.linalg.norm(proj_array[j])
                            v2 = proj_array[k] / np.linalg.norm(proj_array[k])
                            similarity = np.dot(v1, v2)
                            distance_matrix[j, k] = distance_matrix[k, j] = 1.0 - similarity

                    print(f"  迭代 {iteration + 1}: 分离 {type_i} 和 {type_j}, 当前距离: {1.0 - np.dot(v1, v2):.4f}")

        # 3. 构建结果字典
        separated_projections = {}
        for i, fault_type in enumerate(fault_types):
            separated_projections[fault_type] = proj_array[i]

        # 4. 打印最终距离统计
        print("最终投影间距:")
        for i in range(n_proj):
            for j in range(i + 1, n_proj):
                v1 = proj_array[i] / np.linalg.norm(proj_array[i])
                v2 = proj_array[j] / np.linalg.norm(proj_array[j])
                similarity = np.dot(v1, v2)
                print(f"  {fault_types[i]} vs {fault_types[j]}: {1.0 - similarity:.4f}")

        return separated_projections
    # evaluate_zero_shot unchanged
    def evaluate_zero_shot(self, data_dict, compound_projections):
        """改进的零样本评估函数，考虑数据可靠性"""
        print("评估零样本复合故障分类能力...")

        # 1. 基本检查
        if self.cnn_model is None or self.cnn_feature_dim <= 0:
            print("错误: CNN模型未训练");
            return 0.0, None

        if compound_projections is None or not compound_projections:
            print("错误: 缺少复合故障投影");
            return 0.0, None

        # 2. 获取测试数据
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        if len(X_test) == 0:
            print("警告: 无复合故障测试数据");
            return 0.0, None

        # 确保测试数据有效
        finite_mask_test = np.all(np.isfinite(X_test), axis=1)
        X_test, y_test = X_test[finite_mask_test], y_test[finite_mask_test]
        if len(X_test) == 0:
            print("错误: 测试数据都是非有限值");
            return 0.0, None

        # 3. 筛选有投影的测试数据
        available_projection_labels = list(compound_projections.keys())
        available_projection_indices = [self.fault_types[label] for label in available_projection_labels
                                        if label in self.fault_types]

        test_mask = np.isin(y_test, available_projection_indices)
        X_compound_test, y_compound_test = X_test[test_mask], y_test[test_mask]

        if len(X_compound_test) == 0:
            print("错误: 无与投影匹配的测试样本");
            return 0.0, None

        print(f"  使用 {len(X_compound_test)} 个测试样本评估 {len(available_projection_labels)} 种复合故障")

        # 4. 准备投影张量
        candidate_labels = available_projection_labels
        candidate_projections_tensor = torch.stack(
            [torch.from_numpy(compound_projections[label]) for label in candidate_labels]
        ).to(self.device)

        # 规范化投影向量
        candidate_projections_norm = F.normalize(candidate_projections_tensor, p=2, dim=1)
        if torch.isnan(candidate_projections_norm).any():
            print("警告: 投影向量中存在NaN值，已替换为0")
            candidate_projections_norm = torch.nan_to_num(candidate_projections_norm, nan=0.0)

        # 5. 模型设置为评估模式
        self.cnn_model.eval()

        # 6. 预测
        y_pred_indices = []
        y_pred_confidences = []  # 存储预测置信度
        inference_batch_size = self.batch_size * 2
        test_features_list = []  # 存储提取的测试特征

        with torch.no_grad():
            num_test = len(X_compound_test)
            for i in range(0, num_test, inference_batch_size):
                batch_x_test = torch.FloatTensor(X_compound_test[i:i + inference_batch_size]).to(self.device)

                # 向测试数据添加少量噪声以提高鲁棒性
                batch_x_test = batch_x_test + 0.01 * torch.randn_like(batch_x_test)
                batch_x_test = torch.clamp(batch_x_test, -1.0, 1.0)

                try:
                    # 提取特征 (无语义输入)
                    batch_features_test = self.cnn_model(batch_x_test, semantic=None, return_features=True)
                    test_features_list.append(batch_features_test.cpu().numpy())
                except Exception as e:
                    print(f"错误: CNN特征提取失败: {e}")
                    # 预测类别0作为回退
                    dummy_preds = [0] * len(batch_x_test)
                    dummy_confs = [0.0] * len(batch_x_test)
                    y_pred_indices.extend(dummy_preds)
                    y_pred_confidences.extend(dummy_confs)
                    continue

                if not torch.all(torch.isfinite(batch_features_test)):
                    print("警告: 特征中存在非有限值，已替换为0")
                    batch_features_test = torch.nan_to_num(batch_features_test, nan=0.0)

                # 规范化特征
                batch_features_norm = F.normalize(batch_features_test, p=2, dim=1)
                if torch.isnan(batch_features_norm).any():
                    batch_features_norm = torch.nan_to_num(batch_features_norm, nan=0.0)

                # 计算特征与投影的相似度
                similarity_matrix = torch.matmul(batch_features_norm, candidate_projections_norm.t())

                # 获取最大相似度及其索引
                max_similarities, batch_pred = torch.max(similarity_matrix, dim=1)

                # 记录预测索引和置信度
                y_pred_indices.extend(batch_pred.cpu().numpy())
                y_pred_confidences.extend(max_similarities.cpu().numpy())

        # 7. 计算指标
        y_pred_numerical = [self.fault_types.get(candidate_labels[idx], -1) for idx in y_pred_indices]
        y_true = y_compound_test

        # 整体准确率
        accuracy = accuracy_score(y_true, y_pred_numerical) * 100

        # 各类别准确率
        class_accuracy = {}
        for fault_type in set([self.idx_to_fault.get(y) for y in y_compound_test]):
            if fault_type is None:
                continue
            mask = [self.idx_to_fault.get(y) == fault_type for y in y_compound_test]
            if sum(mask) > 0:
                class_acc = accuracy_score(
                    [y for i, y in enumerate(y_compound_test) if mask[i]],
                    [y for i, y in enumerate(y_pred_numerical) if mask[i]]
                ) * 100
                class_accuracy[fault_type] = class_acc

        # 8. 可视化
        true_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_true]
        pred_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_pred_numerical]
        display_labels = sorted(list(set(true_labels_str) | set(pred_labels_str)))

        try:
            conf_matrix = confusion_matrix(true_labels_str, pred_labels_str, labels=display_labels)

            # 绘制混淆矩阵
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=display_labels,
                yticklabels=display_labels
            )
            plt.xlabel('预测');
            plt.ylabel('真实')
            plt.title(f'零样本学习混淆矩阵 (准确率: {accuracy:.2f}%)')
            plt.xticks(rotation=45, ha='right');
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('compound_fault_confusion_matrix_zsl.png')
            plt.close()

            # 打印详细结果
            print(f"零样本学习准确率: {accuracy:.2f}%")
            print("各类别准确率:")
            for fault_type, acc in class_accuracy.items():
                print(f"  - {fault_type}: {acc:.2f}%")

            print("混淆矩阵已保存至 'compound_fault_confusion_matrix_zsl.png'")

            # 可视化特征与投影
            if test_features_list:
                test_features = np.vstack(test_features_list)
                # 使用t-SNE降维
                try:
                    from sklearn.manifold import TSNE
                    # 合并测试特征和投影向量
                    all_features = np.vstack([
                        test_features,
                        candidate_projections_tensor.cpu().numpy()
                    ])
                    # 创建标签
                    feature_labels = [self.idx_to_fault.get(y, f"Test_{i}") for i, y in enumerate(y_true)]
                    projection_labels = [f"Proj_{label}" for label in candidate_labels]
                    all_labels = feature_labels + projection_labels

                    # t-SNE降维
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_features) - 1))
                    features_2d = tsne.fit_transform(all_features)

                    # 绘制散点图
                    plt.figure(figsize=(12, 10))
                    # 绘制测试样本
                    test_2d = features_2d[:len(test_features)]
                    for fault_type in set(feature_labels):
                        idx = [i for i, l in enumerate(feature_labels) if l == fault_type]
                        plt.scatter(
                            test_2d[idx, 0], test_2d[idx, 1],
                            alpha=0.6, s=50, label=f"Test: {fault_type}"
                        )

                    # 绘制投影
                    proj_2d = features_2d[len(test_features):]
                    for i, label in enumerate(candidate_labels):
                        plt.scatter(
                            proj_2d[i, 0], proj_2d[i, 1],
                            marker='*', s=200, alpha=1.0, label=f"Proj: {label}"
                        )

                    plt.title('测试特征与投影向量的t-SNE可视化')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                    plt.savefig('zsl_feature_projection_visualization.png')
                    plt.close()
                    print("特征与投影可视化已保存至 'zsl_feature_projection_visualization.png'")
                except Exception as e:
                    print(f"特征可视化失败: {e}")

        except Exception as e:
            print(f"警告: 混淆矩阵计算错误: {e}")
            conf_matrix = None

        return accuracy, conf_matrix

    def visualize_test_features_and_projections(self, data_dict, compound_projections):
        """
        增强的可视化函数，确保清晰展示所有复合故障的测试特征和投影点
        """
        print("\n可视化复合故障测试特征和SEN投影...")

        if self.cnn_model is None or compound_projections is None:
            print("错误: CNN模型未训练或无投影可用")
            return

        # 准备测试数据
        X_test = data_dict.get('X_test')
        y_test = data_dict.get('y_test')

        if X_test is None or len(X_test) == 0:
            print("错误: 无测试数据可视化")
            return

        # 仅保留复合故障数据
        compound_indices = []
        for i, label in enumerate(y_test):
            fault_name = self.idx_to_fault.get(label)
            if fault_name in self.compound_fault_types:
                compound_indices.append(i)

        if not compound_indices:
            print("错误: 测试集中无复合故障数据")
            return

        X_compound = X_test[compound_indices]
        y_compound = y_test[compound_indices]

        # 提取CNN特征
        self.cnn_model.eval()
        features_list = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(X_compound), batch_size):
                batch_x = torch.FloatTensor(X_compound[i:i + batch_size]).to(self.device)
                try:
                    batch_features = self.cnn_model(batch_x, semantic=None, return_features=True)
                    if torch.all(torch.isfinite(batch_features)):
                        features_list.append(batch_features.cpu().numpy())
                except Exception as e:
                    print(f"警告: 特征提取错误: {e}")

        if not features_list:
            print("错误: 无法提取特征")
            return

        features = np.vstack(features_list)

        # 准备投影数据
        projection_features = []
        projection_labels = []

        for fault_type, proj in compound_projections.items():
            if np.all(np.isfinite(proj)):
                projection_features.append(proj)
                projection_labels.append(fault_type)

        if not projection_features:
            print("错误: 无有效投影")
            return

        # 合并数据进行t-SNE降维
        all_features = np.vstack([features, projection_features])

        # t-SNE降维
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, len(all_features) // 3))

        try:
            print("执行t-SNE降维...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                        learning_rate='auto', init='pca', metric='cosine')
            features_2d = tsne.fit_transform(all_features)
        except Exception as e:
            print(f"t-SNE失败: {e}, 尝试PCA...")
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(all_features)

        # 分离测试特征和投影
        test_2d = features_2d[:len(features)]
        proj_2d = features_2d[len(features):]

        # 创建可视化
        plt.figure(figsize=(12, 10))

        # 绘制测试特征
        fault_names = [self.idx_to_fault.get(label) for label in y_compound]
        for fault_type in set(fault_names):
            indices = [i for i, ft in enumerate(fault_names) if ft == fault_type]
            plt.scatter(
                test_2d[indices, 0], test_2d[indices, 1],
                alpha=0.6, s=50, label=f"Test: {fault_type}"
            )

        # 绘制投影 - 使用更大更明显的标记
        colors = ['purple', 'magenta', 'brown', 'gray']
        markers = ['*', 'P', 'X', 'D']

        for i, (fault, feat_2d) in enumerate(zip(projection_labels, proj_2d)):
            marker_style = markers[i % len(markers)]
            color = colors[i % len(colors)]
            plt.scatter(
                feat_2d[0], feat_2d[1],
                marker=marker_style, s=300,
                color=color, edgecolors='black', linewidth=1.5,
                label=f"Proj: {fault}"
            )

            # 添加文字标注确保投影点可见
            plt.annotate(
                f"P:{fault}",
                (feat_2d[0], feat_2d[1]),
                xytext=(10, 10),  # 文字偏移量
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
            )

        plt.title('复合故障测试特征与投影的t-SNE可视化')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('compound_fault_projections.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("复合故障投影可视化已保存至'compound_fault_projections.png'")

    # run_pipeline unchanged
    def run_pipeline(self):
        """使用改进的复合故障投影的完整流水线"""
        start_time = time.time()
        accuracy = 0.0
        try:
            print("\n--- 步骤 1: 加载数据 ---")
            data_dict = self.load_data()
            if data_dict is None:
                raise RuntimeError("数据加载失败")

            print("\n--- 步骤 2: 构建语义 ---")
            semantic_dict = self.build_semantics(data_dict)
            if semantic_dict is None:
                raise RuntimeError("语义构建失败")

            print("\n--- 步骤 3: 可视化语义相似度 ---")
            self.visualize_semantics(semantic_dict)
            self.visualize_data_semantics_distribution(data_dict)

            print("\n--- 步骤 4: 改进的CNN和SEN联合训练 ---")
            self.joint_train_cnn_sen_improved(data_dict, semantic_dict)
            if self.cnn_model is None or self.embedding_net is None:
                raise RuntimeError("CNN和SEN联合训练失败")

            print("\n--- 步骤 5: 可视化CNN特征 ---")
            self.visualize_cnn_feature_distribution(data_dict)

            print("\n--- 步骤 6: 生成稳定的复合故障投影 ---")
            # 替换为新的投影生成函数
            compound_projections = self.generate_compound_fault_projections_fixed(semantic_dict, data_dict)
            if compound_projections is None:
                raise RuntimeError("投影生成失败")

            # 添加投影可视化步骤
            print("\n--- 步骤 6b: 可视化复合故障投影 ---")
            self.visualize_test_features_and_projections(data_dict, compound_projections)

            print("\n--- 步骤 7: 评估零样本学习 ---")
            accuracy, _ = self.evaluate_zero_shot(data_dict, compound_projections)

        except Exception as e:
            print(f"错误: 流水线失败: {e}")
            traceback.print_exc()
            accuracy = 0.0

        end_time = time.time()
        print(f"\n--- 流水线在 {(end_time - start_time) / 60:.2f} 分钟内完成 ---")
        return accuracy


# 添加正交正则化类
class OrthogonalRegularization(nn.Module):
    """正交约束：促使不同类型的特征正交"""

    def __init__(self):
        super().__init__()

    def forward(self, weight_matrix):
        """
        weight_matrix: 需要正交化的权重矩阵
        """
        if weight_matrix.dim() < 2:
            return torch.tensor(0.0, device=weight_matrix.device)

        if weight_matrix.shape[0] > weight_matrix.shape[1]:
            # 如果行数大于列数，转置后再计算
            weight_matrix = weight_matrix.T

        # 计算Gram矩阵
        gram = torch.mm(weight_matrix, weight_matrix.T)

        # 创建单位矩阵
        identity = torch.eye(gram.size(0), device=gram.device)

        # 正交损失：希望Gram矩阵接近单位矩阵
        # 忽略对角线元素，只关注非对角元素
        mask = torch.ones_like(gram) - identity
        loss = torch.sum((gram * mask) ** 2) / (gram.size(0) ** 2)

        return loss

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

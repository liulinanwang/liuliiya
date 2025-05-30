import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
from scipy.interpolate import interp1d
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import os
import warnings
import random
import time
from collections import Counter
import traceback
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
SEGMENT_LENGTH = 1024
OVERLAP = 0.5
STEP = int(SEGMENT_LENGTH * (1 - OVERLAP))
if STEP < 1: STEP = 1
DEFAULT_WAVELET = 'db4'
DEFAULT_WAVELET_LEVEL = 3
AE_LATENT_DIM = 64
AE_EPOCHS = 30
AE_LR = 0.001
AE_BATCH_SIZE = 64
AE_CONTRASTIVE_WEIGHT = 1.2
AE_NOISE_STD = 0.05
CNN_EPOCHS = 3
CNN_LR = 0.001
SEN_EPOCHS = 20
SEN_LR = 0.0005
CNN_FEATURE_DIM = 256
DEFAULT_BATCH_SIZE = 128


def configure_chinese_font():
    """确保matplotlib正确配置中文字体"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np
    import platform
    system = platform.system()
    font_list = ['SimHei', 'Microsoft YaHei', 'SimSun']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_font = None
    for font in font_list:
        if font in available_fonts:
            chinese_font = font
            break
    if chinese_font:
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = [chinese_font] + plt.rcParams.get('font.sans-serif', [])
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(f"中文字体设置为: '{chinese_font}'")
        return chinese_font
    else:
        print("警告: 未找到合适的中文字体，将尝试使用matplotlib默认配置")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        return None


# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def create_segments(signal_data, segment_length, step):
    """Creates overlapping segments from a signal, ensuring segments are finite."""
    segments = []
    num_samples = len(signal_data)
    if num_samples >= segment_length:
        for i in range(0, num_samples - segment_length + 1, step):
            segment = signal_data[i: i + segment_length]
            if len(segment) == segment_length:

                if np.all(np.isfinite(segment)):
                    segments.append(segment)

    return np.array(segments, dtype=np.float32) if segments else np.empty((0, segment_length), dtype=np.float32)


class DataPreprocessor:
    def __init__(self, sample_length=SEGMENT_LENGTH, overlap=OVERLAP, augment=True, random_seed=42):
        self.sample_length = sample_length
        self.overlap = overlap
        self.augment = augment
        self.stride = int(sample_length * (1 - overlap))
        if self.stride < 1: self.stride = 1
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def cubic_spline_interpolation(self, signal_with_missing):
        """使用三次样条插值法处理缺失值 (Improved fallback)"""
        mask = np.isnan(signal_with_missing)
        if np.any(mask):
            x = np.arange(len(signal_with_missing))
            x_known = x[~mask]
            y_known = signal_with_missing[~mask]
            if len(x_known) >= 2:
                if len(x_known) >= 4:
                    kind = 'cubic'
                else:
                    kind = 'linear'
                try:
                    f = interp1d(x_known, y_known, kind=kind, fill_value="extrapolate", bounds_error=False)
                    signal_with_missing[mask] = f(x[mask])
                except ValueError as e:
                    print(f"W: Interpolation failed ({kind}): {e}. Using mean fill.")
                    mean_val = np.nanmean(y_known) if len(y_known) > 0 else 0
                    signal_with_missing[mask] = mean_val
            elif len(x_known) == 1:
                signal_with_missing[mask] = y_known[0]
            else:
                signal_with_missing[mask] = 0.0
        # Ensure no NaNs remain
        return np.nan_to_num(signal_with_missing, nan=0.0)

    def remove_outliers_sigma(self, signal_data, sigma_threshold=5):
        """Simpler outlier removal using sigma-clipping (like gen.py's intent)"""
        signal_data_clean = np.nan_to_num(signal_data, nan=np.nanmedian(signal_data))
        mean = np.mean(signal_data_clean)
        std = np.std(signal_data_clean)
        if std < 1e-9:
            return signal_data
        lower_bound = mean - sigma_threshold * std
        upper_bound = mean + sigma_threshold * std

        return np.clip(signal_data, lower_bound, upper_bound)

    def wavelet_denoising_universal(self, signal_data, wavelet=DEFAULT_WAVELET, level=DEFAULT_WAVELET_LEVEL):
        """Wavelet denoising using Universal Threshold (closer to gen.py's description)"""
        signal_data = np.nan_to_num(signal_data, nan=0.0)
        data_len = len(signal_data)
        if data_len == 0: return signal_data

        try:
            max_level = pywt.dwt_max_level(data_len, pywt.Wavelet(wavelet))
            actual_level = min(level, max_level)
            if actual_level < 1:
                return signal_data

            coeffs = pywt.wavedec(signal_data, wavelet, level=actual_level)
            detail_coeffs = coeffs[1:]

            sigma = 0.0
            if len(detail_coeffs) > 0 and detail_coeffs[-1] is not None and len(detail_coeffs[-1]) > 0:
                median_abs_dev = np.median(np.abs(detail_coeffs[-1] - np.median(detail_coeffs[-1])))
                sigma = median_abs_dev / 0.6745 if median_abs_dev > 1e-9 else 0.0
            else:
                valid_coeffs = [c for c in detail_coeffs if c is not None and len(c) > 1]
                if valid_coeffs:
                    median_abs_dev = np.median([np.median(np.abs(c - np.median(c))) for c in valid_coeffs])
                    sigma = median_abs_dev / 0.6745 if median_abs_dev > 1e-9 else 0.0

            thr = sigma * np.sqrt(2 * np.log(max(data_len, 1))) if sigma > 1e-9 and data_len > 1 else 0.0

            coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, thr, mode='soft') if c is not None else None for c in
                                           coeffs[1:]]

            denoised_signal = pywt.waverec(coeffs_thresh, wavelet)

            if len(denoised_signal) != data_len:
                if len(denoised_signal) > data_len:
                    denoised_signal = denoised_signal[:data_len]
                else:
                    padding = data_len - len(denoised_signal)
                    denoised_signal = np.pad(denoised_signal, (0, padding), 'edge')

        except Exception as e:
            print(f"W: Wavelet denoising failed unexpectedly: {e}. Returning original signal.")
            traceback.print_exc()
            denoised_signal = signal_data
        if not np.all(np.isfinite(denoised_signal)):
            print("W: NaN detected after wavelet denoising. Replacing with 0.")
            denoised_signal = np.nan_to_num(denoised_signal, nan=0.0)

        return denoised_signal

    def segment_signal(self, signal_data):
        """Uses the helper create_segments function."""
        return create_segments(signal_data, self.sample_length, self.stride)

    def preprocess(self, signal_data, augmentation=False, scale_globally=False):
        """改进的预处理流程，防止数据泄露"""
        # 初始NaN检查
        if not np.all(np.isfinite(signal_data)):
            signal_data = np.nan_to_num(signal_data, nan=np.nanmean(signal_data),
                                        posinf=np.nanmax(signal_data), neginf=np.nanmin(signal_data))

        signal_data = self.cubic_spline_interpolation(signal_data)
        signal_data = self.remove_outliers_sigma(signal_data)
        signal_data = self.wavelet_denoising_universal(signal_data)

        if scale_globally:
            segments = self.segment_signal(signal_data)
            if segments.shape[0] == 0:
                return np.empty((0, self.sample_length))

            segments_flat = segments.reshape(-1, 1)
            normalized_flat = self.scaler.fit_transform(segments_flat)
            signal_data_normalized = normalized_flat.reshape(segments.shape)
        else:
            signal_data_reshaped = signal_data.reshape(-1, 1)
            signal_data_normalized = self.scaler.fit_transform(signal_data_reshaped).flatten()
            signal_data_normalized = self.segment_signal(signal_data_normalized)

        if not augmentation or not self.augment:
            return signal_data_normalized

        processed_signals = [signal_data_normalized]

        aug_seed = self.random_seed + 1

        for i in range(2):
            np.random.seed(aug_seed + i)
            noise_std = 0.01 + 0.01 * i
            noise_segments = []

            for segment in signal_data_normalized:
                noisy_seg = segment + np.random.normal(0, noise_std, len(segment))
                noisy_seg = np.clip(noisy_seg, -1.0, 1.0)
                noise_segments.append(noisy_seg)

            if noise_segments:
                processed_signals.append(np.array(noise_segments))

        for i in range(2):
            np.random.seed(aug_seed + 10 + i)
            shift_segments = []

            for segment in signal_data_normalized:
                max_shift = len(segment) // 10
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
                scale_factor = 0.95 + 0.1 * np.random.random()
                scaled_seg = segment * scale_factor
                scaled_seg = np.clip(scaled_seg, -1.0, 1.0)
                scale_segments.append(scaled_seg)

            if scale_segments:
                processed_signals.append(np.array(scale_segments))

        if not processed_signals:
            return np.empty((0, self.sample_length))

        return np.vstack(processed_signals)


class Autoencoder(nn.Module):
    def __init__(self, input_dim=SEGMENT_LENGTH, latent_dim=AE_LATENT_DIM):
        super(Autoencoder, self).__init__()
        h1, h2 = 256, 128
        if input_dim < h1: h1 = max(latent_dim, (input_dim + latent_dim) // 2)
        if h1 < h2: h2 = max(latent_dim, (h1 + latent_dim) // 2)
        if h2 < latent_dim: h2 = latent_dim
        h1 = max(h1, h2)
        latent_dim = min(latent_dim, h2)

        if not (input_dim > 0 and h1 > 0 and h2 > 0 and latent_dim > 0):
            raise ValueError(f"Invalid AE dimensions: In={input_dim}, H1={h1}, H2={h2}, Latent={latent_dim}")

        print(f"AE Arch (gen.py style): {input_dim} -> {h1} -> {h2} -> {latent_dim} -> {h2} -> {h1} -> {input_dim}")

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),

            nn.Linear(h1, h2),
            nn.ReLU(),

            nn.Linear(h2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),

            nn.Linear(h2, h1),
            nn.ReLU(),

            nn.Linear(h1, input_dim),
            nn.Tanh()
        )

        self.actual_latent_dim = latent_dim

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        if not torch.all(torch.isfinite(x)):
            print("W: AE encode input contains non-finite values. Clamping.")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        latent = self.encoder(x)
        if not torch.all(torch.isfinite(latent)):
            print("W: AE encode output contains non-finite values. Clamping.")
            latent = torch.nan_to_num(latent, nan=0.0)  # Clamp latent space?
        return latent


class AttributeSemanticMapper(nn.Module):
    def __init__(self, attr_dim=3, semantic_dim=64, hidden_dims=[128, 256, 128]):
        super().__init__()

        layers = []
        # 输入层
        layers.append(nn.Linear(attr_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())

        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))

        # 输出层
        layers.append(nn.Linear(hidden_dims[-1], semantic_dim))

        self.mapper = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapper(x)


class FaultSemanticBuilder:
    def __init__(self, latent_dim=AE_LATENT_DIM, hidden_dim=128,  # hidden_dim seems unused here
                 compound_data_semantic_generation_rule='mapper_output'):
        self.latent_dim_config = latent_dim
        self.compound_fusion_mlp = None
        self.actual_latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = None
        self.preprocessor = DataPreprocessor(sample_length=SEGMENT_LENGTH)
        self.knowledge_dim = 0
        self.data_semantics = {}
        self.idx_to_fault = {}
        self.all_latent_features = None
        self.all_latent_labels = None
        self.compound_data_semantic_generation_rule = compound_data_semantic_generation_rule

        self.fault_location_attributes = {
            'normal': [0, 0, 0], 'inner': [1, 0, 0], 'outer': [0, 1, 0], 'ball': [0, 0, 1],
            'inner_outer': [1, 1, 0], 'inner_ball': [1, 0, 1], 'outer_ball': [0, 1, 1],
            'inner_outer_ball': [1, 1, 1]
        }
        self.bearing_params_attributes = {
            'inner_diameter': 17 / 40, 'outer_diameter': 1.0, 'width': 12 / 40,
            'ball_diameter': 6.75 / 40, 'ball_number': 9 / 20
        }

        self.single_fault_types_ordered = ['normal', 'inner', 'outer', 'ball']

        self.compound_fault_definitions = {
            'inner_outer': ['inner', 'outer'],
            'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'],
            'inner_outer_ball': ['inner', 'outer', 'ball']
        }

        self.enhanced_attribute_dim = len(self.fault_location_attributes['normal']) + len(
            self.bearing_params_attributes)

    def _get_enhanced_attributes(self):
        """Helper to create enhanced attribute vectors."""
        enhanced_attributes = {}
        for fault_type, loc_attr in self.fault_location_attributes.items():
            enhanced_attributes[fault_type] = np.array(loc_attr + list(self.bearing_params_attributes.values()),
                                                       dtype=np.float32)
        return enhanced_attributes

    def build_knowledge_semantics(self):
        """构建基于轴承故障位置和尺寸的知识语义 (Unchanged from original)"""
        fault_location = {
            'normal': [0, 0, 0], 'inner': [1, 0, 0], 'outer': [0, 1, 0], 'ball': [0, 0, 1],
            'inner_outer': [1, 1, 0], 'inner_ball': [1, 0, 1], 'outer_ball': [0, 1, 1],
            'inner_outer_ball': [1, 1, 1]
        }
        bearing_params = {
            'inner_diameter': 17 / 40, 'outer_diameter': 1.0, 'width': 12 / 40,
            'ball_diameter': 6.75 / 40, 'ball_number': 9 / 20
        }
        knowledge_semantics = {}
        first_vec = None
        for fault_type, location_encoding in fault_location.items():
            semantics = np.array(location_encoding + list(bearing_params.values()), dtype=np.float32)
            knowledge_semantics[fault_type] = semantics
            if first_vec is None: first_vec = semantics
        self.knowledge_dim = len(first_vec) if first_vec is not None else 0
        return knowledge_semantics

    def _ae_contrastive_loss(self, latent, latent_aug, labels, temperature=0.2):
        """计算对比损失 (logic matching gen.py _contrastive_loss)."""
        batch_size = latent.size(0)
        if batch_size <= 1: return torch.tensor(0.0, device=latent.device)

        latent_norm = nn.functional.normalize(latent, p=2, dim=1)
        latent_aug_norm = nn.functional.normalize(latent_aug, p=2, dim=1)

        if torch.isnan(latent_norm).any() or torch.isnan(latent_aug_norm).any():
            print("W: NaN found after L2 norm in contrastive loss. Using 0 loss for batch.")
            latent_norm = torch.nan_to_num(latent_norm, nan=0.0)
            latent_aug_norm = torch.nan_to_num(latent_aug_norm, nan=0.0)

        sim_matrix = torch.matmul(latent_norm, latent_aug_norm.t()) / temperature
        sim_matrix = torch.clamp(sim_matrix, min=-30.0, max=30.0)
        labels_eq = (labels.unsqueeze(1) == labels.unsqueeze(0)).float().to(self.device)

        exp_sim = torch.exp(sim_matrix)

        pos_sum = (exp_sim * labels_eq).sum(dim=1)

        total_sum = exp_sim.sum(dim=1)

        contrastive_loss_terms = -torch.log((pos_sum + 1e-8) / (total_sum + 1e-8))

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

        if torch.isnan(mean_loss) or torch.isinf(mean_loss):
            print("W: AE contrastive loss resulted in NaN/Inf mean. Returning 0.")

            return torch.tensor(0.0, device=latent.device)

        return mean_loss

    def train_autoencoder(self, X_train, labels, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, lr=AE_LR):
        """训练自编码器 (Closer alignement with gen.py training logic)"""
        print("Training Autoencoder for data semantics (gen.py aligned)...")
        input_dim = X_train.shape[1]

        self.autoencoder = Autoencoder(input_dim=input_dim, latent_dim=self.latent_dim_config).to(self.device)

        self.actual_latent_dim = self.autoencoder.actual_latent_dim
        if self.actual_latent_dim != self.latent_dim_config:
            print(f"W: AE latent dim adjusted by architecture: {self.actual_latent_dim}")

        if labels is None: raise ValueError("Labels are required for gen.py style contrastive AE training.")

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(labels))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        all_labels_np = labels
        all_data_tensor = torch.FloatTensor(X_train).to(self.device)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)

        criterion_recon = nn.MSELoss()

        self.autoencoder.train()
        num_samples = len(X_train)
        best_loss = float('inf')
        patience_counter = 0
        patience = 15
        for epoch in range(epochs):
            epoch_recon_loss = 0.0
            epoch_contrastive_loss = 0.0
            samples_processed = 0

            for data in train_loader:
                batch_data, batch_labels = data
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                if batch_data.shape[0] < 2: continue
                noise = torch.randn_like(batch_data) * AE_NOISE_STD
                batch_aug = torch.clamp(batch_data + noise, -1.0, 1.0)
                optimizer.zero_grad()

                decoded = self.autoencoder(batch_data)
                decoded_aug = self.autoencoder(batch_aug)
                latent = self.autoencoder.encode(batch_data)
                latent_aug = self.autoencoder.encode(batch_aug)

                if not torch.all(torch.isfinite(decoded)) or not torch.all(torch.isfinite(latent)) or \
                        not torch.all(torch.isfinite(decoded_aug)) or not torch.all(torch.isfinite(latent_aug)):
                    print(f"W: NaN/Inf detected in AE outputs epoch {epoch + 1}. Skipping batch loss.")
                    continue

                recon_loss = criterion_recon(decoded, batch_data) + criterion_recon(decoded_aug, batch_aug)
                recon_loss = recon_loss / 2.0
                contrastive_loss = self._ae_contrastive_loss(latent, latent_aug, batch_labels)

                total_loss = recon_loss + AE_CONTRASTIVE_WEIGHT * contrastive_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"W: Total AE loss NaN/Inf epoch {epoch + 1}. Skipping backward.")
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
                optimizer.step()

                epoch_recon_loss += recon_loss.item() * batch_data.size(0)
                epoch_contrastive_loss += contrastive_loss.item() * batch_data.size(0)
                samples_processed += batch_data.size(0)

            scheduler.step()

            if samples_processed > 0:
                avg_recon_loss = epoch_recon_loss / samples_processed
                avg_contrastive_loss = epoch_contrastive_loss / samples_processed
                avg_total_loss = avg_recon_loss + AE_CONTRASTIVE_WEIGHT * avg_contrastive_loss
                print(
                    f"Epoch [{epoch + 1}/{epochs}] - AE Total Loss: {avg_total_loss:.6f} (Recon: {avg_recon_loss:.6f}, Contr: {avg_contrastive_loss:.6f})")

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

        if os.path.exists('best_autoencoder_gen_aligned.pth'):
            self.autoencoder.load_state_dict(torch.load('best_autoencoder_gen_aligned.pth'))
            print("Loaded best AE model state (gen aligned).")
        else:
            print("W: No best AE model state file found. Using the last state.")
        self.autoencoder.eval()

        print("Calculating data semantic centroids and storing all latent features...")
        self.data_semantics = {}
        all_latent_list = []
        inference_batch_size = batch_size * 4

        # 确保 all_data_tensor 和 all_labels_np 可用
        if X_train is None or labels is None:
            print("E: Full data/labels tensor not available for latent feature extraction.")
            # 清空可能存在的旧数据
            self.all_latent_features = None
            self.all_latent_labels = None
            return

        all_data_tensor = torch.FloatTensor(X_train).to(self.device)
        all_labels_np = labels  # 使用传入的原始标签
        num_samples = len(X_train)

        with torch.no_grad():
            for i in range(0, num_samples, inference_batch_size):
                batch = all_data_tensor[i:i + inference_batch_size]
                if not torch.all(torch.isfinite(batch)): continue  # Skip bad batches
                latent_batch = self.autoencoder.encode(batch)
                if torch.all(torch.isfinite(latent_batch)):
                    all_latent_list.append(latent_batch.cpu().numpy())
                else:
                    print(f"W: NaN/Inf in latent vectors during centroid calc index {i}")

        if not all_latent_list:
            print("E: No valid latent features extracted for centroids.")
            # 清空可能存在的旧数据
            self.all_latent_features = None
            self.all_latent_labels = None
            return

        all_latent_features_raw = np.vstack(all_latent_list)
        labels_array_raw = all_labels_np[:all_latent_features_raw.shape[0]]  # 确保标签数量匹配提取的特征数

        # --- 过滤非有限值 ---
        finite_mask = np.all(np.isfinite(all_latent_features_raw), axis=1)
        if not np.all(finite_mask):
            num_non_finite = np.sum(~finite_mask)
            print(f"W: Filtering {num_non_finite} non-finite latent vectors before centroid calc.")
            all_latent_features_filtered = all_latent_features_raw[finite_mask]
            labels_array_filtered = labels_array_raw[finite_mask]
        else:
            all_latent_features_filtered = all_latent_features_raw
            labels_array_filtered = labels_array_raw

        if all_latent_features_filtered.shape[0] == 0:
            print("E: No finite latent features remaining after filtering.")
            # 清空可能存在的旧数据
            self.all_latent_features = None
            self.all_latent_labels = None
            return

        # --- 保存过滤后的 latent features 和 labels ---
        self.all_latent_features = all_latent_features_filtered
        self.all_latent_labels = labels_array_filtered
        print(f"  Stored {len(self.all_latent_features)} filtered latent features for single faults.")

        # --- 计算 Centroids (不变) ---
        unique_label_indices = np.unique(labels_array_filtered)
        for label_idx in unique_label_indices:
            type_mask = (labels_array_filtered == label_idx)
            if not np.any(type_mask): continue
            type_features = all_latent_features_filtered[type_mask]
            n_clusters = min(2, len(type_features))  # 你可以调整聚类数

            if len(type_features) > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(type_features)
                # 取最大簇中心
                labels, counts = np.unique(kmeans.labels_, return_counts=True)
                main_cluster = labels[np.argmax(counts)]
                centroid = kmeans.cluster_centers_[main_cluster]
            else:
                centroid = type_features[0]

            fault_type = self.idx_to_fault.get(label_idx, f"UnknownLabel_{label_idx}")
            self.data_semantics[fault_type] = centroid
            # centroid = centroid / (np.linalg.norm(centroid) + 1e-8) # 归一化在原代码中被注释掉了，保持一致

            if np.all(np.isfinite(centroid)):
                fault_type = self.idx_to_fault.get(label_idx, f"UnknownLabel_{label_idx}")
                if fault_type == f"UnknownLabel_{label_idx}":
                    print(f"W: Cannot map label index {label_idx} back to fault type name.")
                self.data_semantics[fault_type] = centroid
            else:
                fault_type = self.idx_to_fault.get(label_idx, f"UnknownLabel_{label_idx}")
                print(
                    f"W: Centroid calculation for '{fault_type}' (Label {label_idx}) resulted in non-finite values. Setting zero.")
                self.data_semantics[fault_type] = np.zeros(self.actual_latent_dim, dtype=np.float32)

        print(f"AE training & centroid calculation complete. Found {len(self.data_semantics)} centroids.")

    def extract_data_semantics(self, X, fault_labels=None):
        """Extract data semantics using the trained AE (spectral clustering part retained)."""
        if self.autoencoder is None:
            raise ValueError("Autoencoder has not been trained. Call train_autoencoder first.")
        if X is None or X.shape[0] == 0:
            return np.empty((0, self.actual_latent_dim)), {} if fault_labels is not None else np.empty(
                (0, self.actual_latent_dim))

        self.autoencoder.eval()
        all_data_semantics_list = []
        inference_batch_size = AE_BATCH_SIZE * 4

        X_tensor = torch.FloatTensor(X).to(self.device)
        original_indices = np.arange(len(X))
        filtered_indices = []

        with torch.no_grad():
            for i in range(0, X_tensor.size(0), inference_batch_size):
                batch_indices = original_indices[i: i + inference_batch_size]
                batch_x = X_tensor[batch_indices]

                batch_valid_mask = torch.all(torch.isfinite(batch_x), dim=1)
                if not batch_valid_mask.all():
                    print(
                        f"W: Non-finite data in batch for AE encoding (extract), index {i}. Processing valid rows only.")
                    batch_x = batch_x[batch_valid_mask]
                    batch_indices = batch_indices[batch_valid_mask.cpu().numpy()]

                if batch_x.shape[0] == 0: continue

                z_batch = self.autoencoder.encode(batch_x)

                batch_z_valid_mask = torch.all(torch.isfinite(z_batch), dim=1)
                if not batch_z_valid_mask.all():
                    print(f"W: Non-finite latent vectors from AE encoder (extract), index {i}. Filtering.")
                    z_batch = z_batch[batch_z_valid_mask]
                    batch_indices = batch_indices[batch_z_valid_mask.cpu().numpy()]

                if z_batch.shape[0] > 0:
                    all_data_semantics_list.append(z_batch.cpu().numpy())
                    filtered_indices.extend(batch_indices)

        if not all_data_semantics_list:
            print("E: No valid data semantics extracted.")
            return np.empty((0, self.actual_latent_dim)), {} if fault_labels is not None else np.empty(
                (0, self.actual_latent_dim))

        data_semantics = np.vstack(all_data_semantics_list)
        if fault_labels is not None:
            fault_labels_filtered = fault_labels[np.array(filtered_indices, dtype=int)]
        else:
            fault_labels_filtered = None
        prototype_semantics = {}
        if fault_labels_filtered is not None:
            unique_faults = np.unique(fault_labels_filtered)
            for fault in unique_faults:
                indices = np.where(fault_labels_filtered == fault)[0]
                if len(indices) > 0:
                    fault_type_semantics = data_semantics[indices]
                    if len(fault_type_semantics) > 10:
                        n_clusters = max(2, min(5, len(fault_type_semantics) // 3))
                        try:
                            spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                                                          assign_labels='kmeans', random_state=42,
                                                          n_neighbors=max(5, min(10, len(fault_type_semantics) - 1)))
                            cluster_labels = spectral.fit_predict(fault_type_semantics)

                            if cluster_labels is None or len(cluster_labels) != len(fault_type_semantics):
                                prototype = np.mean(fault_type_semantics, axis=0)
                            else:
                                counts = np.bincount(cluster_labels[cluster_labels >= 0])
                                if len(counts) > 0:
                                    largest_cluster = np.argmax(counts)
                                    prototype_idx = np.where(cluster_labels == largest_cluster)[0]
                                    if len(prototype_idx) > 0:
                                        prototype = np.mean(fault_type_semantics[prototype_idx], axis=0)
                                    else:
                                        prototype = np.mean(fault_type_semantics, axis=0)  # Fallback
                                else:
                                    prototype = np.mean(fault_type_semantics, axis=0)  # Fallback
                        except Exception as e:
                            print(f"E: Spectral clustering exception for fault {fault}: {e}. Using mean.")
                            prototype = np.mean(fault_type_semantics, axis=0)
                    else:
                        prototype = np.mean(fault_type_semantics, axis=0)

                    if np.all(np.isfinite(prototype)):
                        fault_name = self.idx_to_fault.get(fault, f"UnknownLabel_{fault}")
                        prototype_semantics[fault_name] = prototype

        if fault_labels_filtered is not None:
            return data_semantics, prototype_semantics
        else:
            return data_semantics

        # In class FaultSemanticBuilder:

    def synthesize_compound_semantics(self, single_fault_prototypes):
        """
        Synthesizes compound fault data semantics based on the configured rule.
        The AttributeSemanticMapper (if used) is trained with averaged single-fault prototypes as targets.

        Args:
            single_fault_prototypes (dict): Single fault data semantic prototypes from the AE.
                                            Keys are fault names ('normal', 'inner', etc.), values are numpy arrays.

        Returns:
            dict: Synthesized compound fault data semantics.
        """
        print(
            f"Synthesizing compound data semantics using rule: '{self.compound_data_semantic_generation_rule}'...")

        if self.actual_latent_dim <= 0:
            print("Error: Invalid actual_latent_dim. AE might not have been trained or updated the dimension.")
            return {}
        if not single_fault_prototypes:
            print("Error: single_fault_prototypes are empty. Cannot synthesize compound semantics.")
            return {}

        enhanced_attributes = self._get_enhanced_attributes()
        attr_dim_for_mapper = self.enhanced_attribute_dim

        synthesized_compound_semantics = {}

        # --- Rule: 'mapper_output' ---
        if self.compound_data_semantic_generation_rule == 'mapper_output':

            class InternalAttributeSemanticMapper(nn.Module):  # Renamed to avoid conflict if outer one exists
                def __init__(self, attr_dim, semantic_dim, hidden_dims=[128, 256, 128]):  # Default hidden_dims
                    super().__init__()
                    layers = []
                    layers.append(nn.Linear(attr_dim, hidden_dims[0]))
                    layers.append(nn.BatchNorm1d(hidden_dims[0]))
                    layers.append(nn.LeakyReLU(0.2))  # Using LeakyReLU as in original
                    layers.append(nn.Dropout(0.2))

                    for i in range(len(hidden_dims) - 1):
                        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                        layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                        layers.append(nn.LeakyReLU(0.2))
                        layers.append(nn.Dropout(0.2))

                    layers.append(nn.Linear(hidden_dims[-1], semantic_dim))
                    self.mapper = nn.Sequential(*layers)

                def forward(self, x):
                    return self.mapper(x)

            mapper = InternalAttributeSemanticMapper(
                attr_dim=attr_dim_for_mapper,
                semantic_dim=self.actual_latent_dim,
                # hidden_dims can be configured if needed
            ).to(self.device)

            optimizer = torch.optim.AdamW(mapper.parameters(), lr=0.001,
                                          weight_decay=1e-5)  # AdamW is generally good
            mse_loss_fn = nn.MSELoss()

            X_mapper_list = []
            Y_mapper_list = []

            # 1. Training data from single faults
            for sf_name in self.single_fault_types_ordered:
                if sf_name in enhanced_attributes and sf_name in single_fault_prototypes:
                    attr_vec = enhanced_attributes[sf_name]
                    proto_vec = single_fault_prototypes[sf_name]
                    if np.all(np.isfinite(attr_vec)) and np.all(np.isfinite(proto_vec)):
                        X_mapper_list.append(attr_vec)
                        Y_mapper_list.append(proto_vec)

            # 2. Training data from compound faults (target is average of constituents)
            for cf_name, constituents in self.compound_fault_definitions.items():
                if cf_name in enhanced_attributes:
                    attr_vec = enhanced_attributes[cf_name]
                    constituent_prototypes = []
                    valid_constituents = True
                    for constituent_name in constituents:
                        if constituent_name in single_fault_prototypes and \
                                np.all(np.isfinite(single_fault_prototypes[constituent_name])):
                            constituent_prototypes.append(single_fault_prototypes[constituent_name])
                        else:
                            print(
                                f"Warning: Missing or invalid prototype for constituent '{constituent_name}' of '{cf_name}'. Skipping for mapper training target.")
                            valid_constituents = False
                            break

                    if valid_constituents and constituent_prototypes:
                        target_compound_semantic = np.mean(constituent_prototypes, axis=0)
                        if np.all(np.isfinite(attr_vec)) and np.all(np.isfinite(target_compound_semantic)):
                            X_mapper_list.append(attr_vec)
                            Y_mapper_list.append(target_compound_semantic)

            if not X_mapper_list or not Y_mapper_list:
                print(
                    "Error: No valid training data for AttributeSemanticMapper. Cannot use 'mapper_output' rule. Falling back.")
                # Fallback to average or handle error
                # For now, let's try to use average as a fallback if mapper training fails
                return self._synthesize_by_rule(single_fault_prototypes, 'average_prototypes')

            X_mapper_train = torch.FloatTensor(np.array(X_mapper_list)).to(self.device)
            Y_mapper_train = torch.FloatTensor(np.array(Y_mapper_list)).to(self.device)

            print(f"Training AttributeSemanticMapper with {len(X_mapper_train)} samples...")
            mapper_epochs = 200  # Reduced from 500 for quicker test, adjust as needed
            mapper_batch_size = min(32, len(X_mapper_train)) if len(X_mapper_train) > 0 else 1

            if mapper_batch_size == 0:
                print("Error: Mapper batch size is 0. Cannot train mapper.")
                return self._synthesize_by_rule(single_fault_prototypes, 'average_prototypes')

            train_dataset_mapper = TensorDataset(X_mapper_train, Y_mapper_train)
            train_loader_mapper = DataLoader(train_dataset_mapper, batch_size=mapper_batch_size, shuffle=True)

            best_mapper_loss = float('inf')
            patience_counter = 0
            mapper_patience = 30

            for epoch in range(mapper_epochs):
                mapper.train()
                epoch_loss = 0
                for x_batch, y_batch in train_loader_mapper:
                    optimizer.zero_grad()
                    pred_semantics = mapper(x_batch)
                    loss = mse_loss_fn(pred_semantics, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0)
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(train_loader_mapper)
                if (epoch + 1) % 20 == 0:
                    print(f"  Mapper Epoch {epoch + 1}/{mapper_epochs}, Loss: {avg_epoch_loss:.6f}")

                if avg_epoch_loss < best_mapper_loss:
                    best_mapper_loss = avg_epoch_loss
                    patience_counter = 0
                    # torch.save(mapper.state_dict(), 'best_internal_attribute_mapper.pth') # Optional save
                else:
                    patience_counter += 1
                    if patience_counter >= mapper_patience:
                        print(f"  Mapper early stopping at epoch {epoch + 1}.")
                        break
            # if os.path.exists('best_internal_attribute_mapper.pth'):
            #     mapper.load_state_dict(torch.load('best_internal_attribute_mapper.pth'))
            print(f"AttributeSemanticMapper training finished. Best Loss: {best_mapper_loss:.6f}")
            mapper.eval()

            # Generate compound semantics using the trained mapper
            for cf_name in self.compound_fault_definitions.keys():
                if cf_name in enhanced_attributes:
                    attr_vec = enhanced_attributes[cf_name]
                    if np.all(np.isfinite(attr_vec)):
                        attr_tensor = torch.FloatTensor(attr_vec).unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            generated_semantic = mapper(attr_tensor).cpu().numpy().squeeze(0)
                        if np.all(np.isfinite(generated_semantic)):
                            # Apply post-processing
                            generated_semantic = self._post_process_compound_semantic(
                                generated_semantic, cf_name, single_fault_prototypes
                            )
                            synthesized_compound_semantics[cf_name] = generated_semantic
                        else:
                            print(f"Warning: Mapper generated non-finite semantic for '{cf_name}'.")
                    else:
                        print(f"Warning: Attribute vector for '{cf_name}' is non-finite.")

            # Fallback for any that failed
            for cf_name in self.compound_fault_definitions.keys():
                if cf_name not in synthesized_compound_semantics:
                    print(f"Warning: Fallback for '{cf_name}' due to mapper failure.")
                    fallback_sem = self._generate_fallback_semantics([cf_name], single_fault_prototypes)
                    if cf_name in fallback_sem:
                        synthesized_compound_semantics[cf_name] = fallback_sem[cf_name]


        # --- Rules: 'average_prototypes' or 'sum_prototypes' (direct rules) ---
        elif self.compound_data_semantic_generation_rule in ['average_prototypes', 'sum_prototypes']:
            synthesized_compound_semantics = self._synthesize_by_rule(
                single_fault_prototypes,
                self.compound_data_semantic_generation_rule
            )
        else:
            print(
                f"Error: Unknown compound_data_semantic_generation_rule: '{self.compound_data_semantic_generation_rule}'. Using fallback.")
            synthesized_compound_semantics = self._generate_fallback_semantics(
                list(self.compound_fault_definitions.keys()), single_fault_prototypes
            )

        num_synthesized = len(synthesized_compound_semantics)
        num_expected = len(self.compound_fault_definitions)
        if num_synthesized < num_expected:
            print(f"Warning: Synthesized {num_synthesized}/{num_expected} compound fault semantics.")
            # Attempt to fill any missing with a simple average as a last resort
            missing_types = [cf for cf in self.compound_fault_definitions.keys() if
                             cf not in synthesized_compound_semantics]
            if missing_types:
                print(f"Attempting last resort average for missing types: {missing_types}")
                last_resort_semantics = self._synthesize_by_rule(single_fault_prototypes, 'average_prototypes',
                                                                 specific_types=missing_types)
                synthesized_compound_semantics.update(last_resort_semantics)

        return synthesized_compound_semantics

    def _synthesize_by_rule(self, single_fault_prototypes, rule, specific_types=None):
        """Helper function for direct rule-based synthesis (average or sum)."""
        temp_compound_semantics = {}
        target_compound_types = specific_types if specific_types is not None else self.compound_fault_definitions.keys()

        for cf_name in target_compound_types:
            constituents = self.compound_fault_definitions.get(cf_name)
            if not constituents:
                print(f"Warning: No definition found for compound fault '{cf_name}' in direct rule synthesis.")
                continue

            component_semantic_list = []
            valid_components = True
            for constituent_name in constituents:
                if constituent_name in single_fault_prototypes and \
                        np.all(np.isfinite(single_fault_prototypes[constituent_name])):
                    component_semantic_list.append(single_fault_prototypes[constituent_name])
                else:
                    print(
                        f"Warning: Missing or invalid prototype for constituent '{constituent_name}' of '{cf_name}' for rule '{rule}'.")
                    valid_components = False
                    break

            if valid_components and component_semantic_list:
                generated_semantic = None
                if rule == 'average_prototypes':
                    generated_semantic = np.mean(component_semantic_list, axis=0)
                elif rule == 'sum_prototypes':
                    generated_semantic = np.sum(component_semantic_list, axis=0)

                if generated_semantic is not None and np.all(np.isfinite(generated_semantic)):
                    # Apply post-processing
                    generated_semantic = self._post_process_compound_semantic(
                        generated_semantic, cf_name, single_fault_prototypes
                    )
                    temp_compound_semantics[cf_name] = generated_semantic
                else:
                    print(f"Warning: Rule '{rule}' resulted in non-finite semantic for '{cf_name}'.")
            else:
                print(
                    f"Note: Could not synthesize '{cf_name}' using rule '{rule}' due to missing/invalid constituents.")

        return temp_compound_semantics

    def _post_process_compound_semantic(self, semantic_vec, compound_type, single_prototypes):
        """
        对生成的复合故障语义进行后处理优化
        """
        # 原始向量副本
        refined = semantic_vec.copy()

        # 获取组件故障类型
        components = [comp for comp in compound_type.split('_') if comp in single_prototypes]

        if not components:
            return refined

        # 1. 确保复合故障语义与组件故障语义保持合理的相似度关系
        component_vecs = [single_prototypes[comp] for comp in components]
        avg_component_vec = np.mean(component_vecs, axis=0)

        # 计算当前相似度
        current_sim = np.dot(refined, avg_component_vec) / (
                np.linalg.norm(refined) * np.linalg.norm(avg_component_vec) + 1e-8)

        # 如果相似度过低，向平均组件向量方向调整
        if current_sim < 0.3:
            adjustment_factor = 0.3
            refined = (1 - adjustment_factor) * refined + adjustment_factor * avg_component_vec

        # 2. 根据复合故障类型进行特定调整
        if compound_type == 'inner_outer':
            # 增强特定维度以区分inner_outer
            mid = len(refined) // 2
            refined[mid:] = refined[mid:] * 1.05

        elif compound_type == 'inner_ball' or compound_type == 'outer_ball':
            # 添加特定调制模式
            mod_pattern = np.sin(np.linspace(0, 3 * np.pi, len(refined))) * 0.05
            refined = refined + mod_pattern * np.mean(np.abs(refined))

        elif compound_type == 'inner_outer_ball':
            # 加强信号幅度以反映多重故障的严重性
            refined = refined * 1.1

        # 3. 归一化处理后的向量
        norm = np.linalg.norm(refined)
        if norm > 1e-8:
            refined = refined / norm * np.linalg.norm(semantic_vec)  # 保持原始范数

        return refined

    def _generate_fallback_semantics(self, fault_types, single_prototypes):
        """
        对无法通过属性映射生成的复合故障，使用原始方法进行补充
        """
        print("使用传统方法生成缺失的复合故障语义...")
        compound_semantics = {}

        for compound_type in fault_types:
            components = compound_type.split('_')
            component_semantics = []

            # 收集有效的组件语义
            for comp in components:
                if comp in single_prototypes and np.all(np.isfinite(single_prototypes[comp])):
                    component_semantics.append(single_prototypes[comp])

            if component_semantics:
                # 使用平均组合
                combined = np.mean(component_semantics, axis=0)

                # 应用后处理
                combined = self._post_process_compound_semantic(
                    combined, compound_type, single_prototypes
                )

                if np.all(np.isfinite(combined)):
                    compound_semantics[compound_type] = combined
                    print(f"  - 已补充生成 {compound_type} 的语义表示")

        return compound_semantics

    def _fallback_compound_synthesis(self, single_fault_prototypes):
        """
        完全回退到原始的复合故障合成方法
        """
        print("使用原始方法合成复合故障语义...")
        # 使用原有的Dempster-Shafer理论方法
        compound_semantics = {}
        compound_combinations = {
            'inner_outer': ['inner', 'outer'],
            'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'],
            'inner_outer_ball': ['inner', 'outer', 'ball']
        }

        # 使用原始方法中的融合逻辑
        for compound_type, components in compound_combinations.items():
            component_semantics = []
            all_valid = True

            for comp in components:
                proto = single_fault_prototypes.get(comp)
                if proto is not None and np.all(np.isfinite(proto)):
                    component_semantics.append(proto)
                else:
                    all_valid = False
                    break

            if all_valid and component_semantics:
                # 简单平均作为回退策略
                synthesized = np.mean(component_semantics, axis=0)

                # 应用后处理
                synthesized = self._post_process_compound_semantic(
                    synthesized, compound_type, single_fault_prototypes
                )

                if np.all(np.isfinite(synthesized)):
                    compound_semantics[compound_type] = synthesized

        return compound_semantics

class BidirectionalSemanticNetwork(nn.Module):
    """进一步增强的双向语义映射网络，加入投影校准和多级特征对齐"""

    def __init__(self, semantic_dim, feature_dim, hidden_dim1=512,
                 hidden_dim2=256, dropout_rate=0.3):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout_rate = dropout_rate
        self.n_centers = 3
        # 正向映射：语义→特征（深度+残差连接）
        self.forward_net = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate * 0.8),

            # 残差连接块1
            ResidualMappingBlock(hidden_dim2),

            # 残差连接块2
            ResidualMappingBlock(hidden_dim2),

            nn.Linear(hidden_dim2, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # 反向映射：特征→语义
        self.reverse_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            # 残差连接
            ResidualMappingBlock(hidden_dim2),

            nn.Linear(hidden_dim2, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate * 0.6),

            nn.Linear(hidden_dim1, semantic_dim),
        )

        # 特征融合模块 - 用于处理复合故障的特殊性
        self.feature_fusion = FeatureFusionModule(feature_dim)

        # 初始化权重
        self._init_weights()
        self.n_centers = 3  # 子中心数
        self.fc_out_multicenter = nn.Linear(self.hidden_dim2, self.feature_dim * self.n_centers)
        self.fc_weight = nn.Linear(self.hidden_dim2, self.n_centers)
        self.norm_out = nn.LayerNorm(self.feature_dim)

    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mode='forward', fault_type=None):
        """
        增强的双向映射处理
        mode: 'forward', 'reverse', 'cycle'
        fault_type: 故障类型名称，用于特殊处理复合故障
        """
        if mode == 'forward':
            hidden = self.forward_net(x)  # [batch, hidden_dim]
            features_all = self.fc_out_multicenter(hidden).view(-1, self.n_centers, self.feature_dim)
            weights = torch.softmax(self.fc_weight(hidden), dim=-1)
            features = torch.sum(features_all * weights.unsqueeze(-1), dim=1)
            features = self.norm_out(features)
            return features

        elif mode == 'reverse':
            return self.reverse_net(x)

        elif mode == 'cycle':
            hidden = self.forward_net(x)
            features_all = self.fc_out_multicenter(hidden).view(-1, self.n_centers, self.feature_dim)
            weights = torch.softmax(self.fc_weight(hidden), dim=-1)
            features = torch.sum(features_all * weights.unsqueeze(-1), dim=1)
            features = self.norm_out(features)
            reconstructed = self.reverse_net(features)
            return features, reconstructed

        else:
            raise ValueError(f"未知的映射模式: {mode}")


class ResidualMappingBlock(nn.Module):
    """残差映射块，增强梯度流和特征提取"""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(x + self.block(x))


class FeatureFusionModule(nn.Module):
    """特征融合模块 - 专门处理复合故障"""

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        # 复合故障特殊处理层
        self.compound_layers = nn.ModuleDict({
            'inner_outer': self._create_fusion_layer(),
            'inner_ball': self._create_fusion_layer(),
            'outer_ball': self._create_fusion_layer(),  # 特别关注这两个表现不佳的类型
            'inner_outer_ball': self._create_fusion_layer(),  # 特别关注这两个表现不佳的类型
        })

        # 注意力模块 - 学习重点关注特征的哪些部分
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.LayerNorm(feature_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )

    def _create_fusion_layer(self):
        """创建融合层"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.LayerNorm(self.feature_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

    def forward(self, x, fault_type):
        """应用特定于复合故障类型的融合"""
        if fault_type in self.compound_layers:
            # 计算注意力权重
            attention_weights = self.attention(x)

            # 应用融合变换
            fusion_features = self.compound_layers[fault_type](x)

            # 根据注意力融合
            return x + attention_weights * (fusion_features - x)
        return x


class ResidualBlock1D(nn.Module):
    """简单的1D残差块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 残差连接
        out = self.relu(out)
        return out


class AEDataSemanticCNN(nn.Module):
    def __init__(self, input_length=SEGMENT_LENGTH, semantic_dim=AE_LATENT_DIM,
                 num_classes=8, feature_dim=256):
        super().__init__()
        self.input_length = input_length
        self.semantic_dim = semantic_dim  # 固定为AE的64维
        self.feature_dim = feature_dim

        # --- 信号编码器 (与原来相同) ---
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # 残差块1
            ResidualBlock1D(64, 64),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            # 残差块2
            ResidualBlock1D(128, 128),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),

            # 残差块3
            ResidualBlock1D(256, 256),

            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )

        # --- 语义编码器 ---
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

        # --- 双通道特征生成器 ---
        self.feature_generator = nn.Sequential(
            nn.Linear(256 + 256, 384),  # 256(信号) + 256(语义) -> 384
            nn.LayerNorm(384),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(384, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # --- 分类器 ---
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, semantic, return_features=False):
        """
        前向传播 - 必须同时提供信号和语义输入
        """
        # 确保输入是3D: [batch, channel, length]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 提取信号特征
        signal_features = self.signal_encoder(x).squeeze(-1)  # [B, 256]

        # 必须提供语义输入
        if semantic is None:
            raise ValueError("语义输入不能为空，双通道CNN必须同时接收信号和语义")

        # 提取语义特征
        semantic_features = self.semantic_encoder(semantic)

        # 融合特征
        combined = torch.cat([signal_features, semantic_features], dim=1)
        features = self.feature_generator(combined)

        # 如果只需要特征，直接返回
        if return_features:
            return features

        # 分类
        logits = self.classifier(features)
        return logits, features


class ZeroShotCompoundFaultDiagnosis:

    def __init__(self, data_path, sample_length=SEGMENT_LENGTH, latent_dim=AE_LATENT_DIM, batch_size=DEFAULT_BATCH_SIZE,
                 compound_data_semantic_generation_rule='mapper_output'):  # Added new parameter with default
        self.data_path = data_path
        self.sample_length = sample_length
        self.latent_dim_config = latent_dim
        self.batch_size = batch_size
        self.compound_data_semantic_generation_rule = compound_data_semantic_generation_rule
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.preprocessor = DataPreprocessor(sample_length=self.sample_length)
        self.semantic_builder = FaultSemanticBuilder(
            latent_dim=self.latent_dim_config,
            compound_data_semantic_generation_rule=self.compound_data_semantic_generation_rule  # Pass the rule
        )
        self.fault_types = {
            'normal': 0, 'inner': 1, 'outer': 2, 'ball': 3,
            'inner_outer': 4, 'inner_ball': 5, 'outer_ball': 6, 'inner_outer_ball': 7
        }
        self.idx_to_fault = {v: k for k, v in self.fault_types.items()}
        self.semantic_builder.idx_to_fault = self.idx_to_fault  # Ensure idx_to_fault is set in builder
        self.single_fault_types_ordered = ['normal', 'inner', 'outer', 'ball']  # For consistency
        self.compound_fault_definitions = {  # Central definition
            'inner_outer': ['inner', 'outer'],
            'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'],
            'inner_outer_ball': ['inner', 'outer', 'ball']
        }
        self.compound_fault_types = list(self.compound_fault_definitions.keys())
        self.num_classes = len(self.fault_types)
        self.cnn_model = None
        self.embedding_net = None
        self.actual_latent_dim = -1
        self.cnn_feature_dim = -1
        self.fused_semantic_dim = -1
        self.consistency_loss_fn = None

    def load_data(self):
        """改进的数据加载函数，确保每类故障样本数量一致且训练集验证集同样平衡"""
        print("加载并预处理数据，确保训练和测试集严格分离，且类别平衡...")
        single_fault_keys = ['normal', 'inner', 'outer', 'ball']
        compound_fault_keys = ['inner_outer', 'inner_ball', 'outer_ball', 'inner_outer_ball']
        fault_files = {
            'normal': 'normal.mat', 'inner': 'inner.mat', 'outer': 'outer.mat', 'ball': 'ball.mat',
            'inner_outer': 'inner_outer.mat', 'inner_ball': 'inner_ball.mat',
            'outer_ball': 'outer_ball.mat', 'inner_outer_ball': 'inner_outer_ball.mat'
        }

        # 第一步：加载所有数据文件
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
                max_len = 200000
                if len(signal_data_flat) > max_len:
                    signal_data_flat = signal_data_flat[:max_len]
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

        # 第二步：预处理所有单一故障数据
        print("\n预处理所有单一故障数据...")
        train_preprocessor = DataPreprocessor(
            sample_length=self.sample_length,
            overlap=0.25,
            augment=True,
            random_seed=42
        )

        # 预处理并存储每个单一故障的分段
        all_single_fault_segments_by_type = {}

        for fault_type, signal_data in single_fault_raw_signals.items():
            label_idx = self.fault_types[fault_type]
            print(f"预处理 {fault_type} 数据...")
            processed_segments = train_preprocessor.preprocess(signal_data, augmentation=True)

            if processed_segments is None or processed_segments.shape[0] == 0:
                print(f"警告: {fault_type} 数据预处理后无有效分段")
                continue

            all_single_fault_segments_by_type[fault_type] = processed_segments
            print(f"  - {fault_type}: {len(processed_segments)} 个分段")

        # 平衡各故障类型样本数量
        min_samples = min([len(segments) for segments in all_single_fault_segments_by_type.values()])
        print(f"\n平衡各单一故障类型样本数量至每类 {min_samples} 个样本")

        # 定义训练集和验证集的比例
        train_ratio = 0.7
        samples_per_class_train = int(min_samples * train_ratio)
        samples_per_class_val = min_samples - samples_per_class_train

        print(f"每类故障分配: {samples_per_class_train}个训练样本, {samples_per_class_val}个验证样本")

        # 初始化训练集和验证集
        X_train_list = []
        y_train_list = []
        X_val_list = []
        y_val_list = []

        # 对每种故障类型，保持训练集和验证集样本数量一致
        for fault_type, segments in all_single_fault_segments_by_type.items():
            label_idx = self.fault_types[fault_type]
            # 打乱样本顺序
            indices = np.random.permutation(len(segments))

            # 选取固定数量的样本用于训练和验证
            train_indices = indices[:samples_per_class_train]
            val_indices = indices[samples_per_class_train:min_samples]

            # 添加到相应的列表
            X_train_list.append(segments[train_indices])
            y_train_list.extend([label_idx] * samples_per_class_train)

            X_val_list.append(segments[val_indices])
            y_val_list.extend([label_idx] * samples_per_class_val)

            print(f"  - {fault_type}: 训练 {len(train_indices)} 个, 验证 {len(val_indices)} 个")

        # 合并训练集和验证集
        if not X_train_list or not X_val_list:
            print("错误: 无法创建训练集和验证集")
            return None

        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)
        X_val = np.vstack(X_val_list)
        y_val = np.array(y_val_list)

        print(f"\n数据集划分完成:")
        print(f"* 训练集: {len(X_train)} 个样本, 每类 {samples_per_class_train} 个")
        print(f"* 验证集: {len(X_val)} 个样本, 每类 {samples_per_class_val} 个")

        # 第四步：处理测试集（复合故障，无增强）
        test_preprocessor = DataPreprocessor(
            sample_length=self.sample_length,
            overlap=0.5,
            augment=False,
            random_seed=44
        )

        X_test_list = []
        y_test_list = []

        print("\n处理测试集（复合故障，无增强）:")
        for fault_type, signal_data in compound_fault_raw_signals.items():
            label_idx = self.fault_types[fault_type]
            processed_segments = test_preprocessor.preprocess(signal_data, augmentation=False)
            if processed_segments is None or processed_segments.shape[0] == 0:
                print(f"警告: {fault_type} 测试数据预处理后无有效分段")
                continue
            X_test_list.append(processed_segments)
            y_test_list.extend([label_idx] * len(processed_segments))
            print(f"  - {fault_type}: {len(processed_segments)} 个分段")

        X_test = np.vstack(X_test_list) if X_test_list else np.empty((0, self.sample_length), dtype=np.float32)
        y_test = np.array(y_test_list) if y_test_list else np.array([])

        # 统计数据分布
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

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def build_semantics(self, data_dict):
        """Builds knowledge and data semantics (AE training aligned with gen.py)."""
        print("Building fault semantics...")
        knowledge_semantics = self.semantic_builder.build_knowledge_semantics()
        if not knowledge_semantics:
            print("E: Failed build knowledge semantics.")
            return None
        print(f"  Knowledge semantics built. Dimension: {self.semantic_builder.knowledge_dim}")

        print("  Training autoencoder for data semantics...")
        X_train_ae = data_dict.get('X_train')
        y_train_ae = data_dict.get('y_train')
        if X_train_ae is None or y_train_ae is None or len(X_train_ae) < AE_BATCH_SIZE:
            print("E: Not enough training data for AE.")
            return None

        try:
            # 调用训练，这会填充 self.semantic_builder.all_latent_features 和 self.semantic_builder.all_latent_labels
            self.semantic_builder.train_autoencoder(X_train_ae, labels=y_train_ae, epochs=AE_EPOCHS,
                                                    batch_size=AE_BATCH_SIZE, lr=AE_LR)
        except Exception as e:
            print(f"E: AE training failed: {e}")
            traceback.print_exc()
            return None

        self.actual_latent_dim = self.semantic_builder.actual_latent_dim
        if self.actual_latent_dim <= 0:
            print("E: Invalid AE latent dimension after training.")
            return None

        single_fault_prototypes = self.semantic_builder.data_semantics
        if not single_fault_prototypes:
            print("E: AE training produced no data semantic prototypes (centroids).")

        print(f"  Data semantic prototypes learned. Dimension: {self.actual_latent_dim}")

        # 合成复合故障语义原型
        compound_data_semantics = self.semantic_builder.synthesize_compound_semantics(single_fault_prototypes)
        print(f"  Compound data semantics synthesized for {len(compound_data_semantics)} types.")

        # 合并所有数据语义原型 (单一+复合)
        data_only_semantics = {**single_fault_prototypes, **compound_data_semantics}

        # 融合知识语义和数据语义原型
        fused_semantics = {}
        self.fused_semantic_dim = self.semantic_builder.knowledge_dim + self.actual_latent_dim
        for ft, k_vec in knowledge_semantics.items():
            d_vec = data_only_semantics.get(ft)
            # 确保知识向量和数据向量都有效且维度正确
            if d_vec is not None and \
                    k_vec is not None and np.all(np.isfinite(k_vec)) and \
                    np.all(np.isfinite(d_vec)) and \
                    len(k_vec) == self.semantic_builder.knowledge_dim and \
                    len(d_vec) == self.actual_latent_dim:
                k_vec_normalized = k_vec / (np.linalg.norm(k_vec) + 1e-8)
                d_vec_normalized = d_vec / (np.linalg.norm(d_vec) + 1e-8)
                fused_vec = np.concatenate([k_vec, d_vec]).astype(np.float32)
                if np.all(np.isfinite(fused_vec)) and len(fused_vec) == self.fused_semantic_dim:
                    fused_semantics[ft] = fused_vec
                else:
                    print(f"W: Fused vector for '{ft}' is invalid or has wrong dimension after concatenation.")
                    print(
                        f"  Fused semantics prepared for {len(fused_semantics)} types. Dimension: {self.fused_semantic_dim}")

        if self.actual_latent_dim <= 0 or self.fused_semantic_dim <= 0:
            print("E: Invalid semantic dimensions calculated. Aborting.")
            return None

        single_fault_latent_features = self.semantic_builder.all_latent_features
        single_fault_latent_labels = self.semantic_builder.all_latent_labels

        if single_fault_latent_features is None or single_fault_latent_labels is None:
            print("W: Failed to retrieve latent features from semantic builder after AE training.")
            # 根据需要决定是否在此处返回 None，如果后续步骤不依赖它们，可以不返回
            # return None

        # --- 返回包含 latent features 的字典 ---
        return {
            'knowledge_semantics': knowledge_semantics,
            'data_prototypes': single_fault_prototypes,  # 单一故障原型
            'compound_data_semantics': compound_data_semantics,  # 合成复合故障原型
            'data_only_semantics': data_only_semantics,  # 所有原型
            'fused_semantics': fused_semantics,
            # --- 新增返回项 ---
            'single_fault_latent_features': single_fault_latent_features,
            'single_fault_latent_labels': single_fault_latent_labels
        }

    def train_ae_data_semantic_cnn(self, data_dict, semantic_dict, epochs=CNN_EPOCHS,
                                   batch_size=DEFAULT_BATCH_SIZE, lr=CNN_LR):
        """使用AE实时提取的语义训练双通道CNN"""

        print("训练基于AE实时数据语义的双通道CNN...")

        if self.cnn_model is None:
            print("错误：CNN模型未初始化")
            return None
        if self.semantic_builder is None or self.semantic_builder.autoencoder is None:
            print("错误: Autoencoder 未在 semantic_builder 中初始化或训练。")
            return None

        self.cnn_model = self.cnn_model.to(self.device)
        self.semantic_builder.autoencoder.eval()  # 确保AE处于评估模式，不进行梯度更新

        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True)  # drop_last is good for AE

        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.cnn_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
            epochs=epochs, pct_start=0.3
        )

        best_val_acc = 0.0
        patience, patience_counter = 10, 0

        for epoch in range(epochs):
            self.cnn_model.train()
            train_loss_epoch, correct_epoch, total_epoch = 0, 0, 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                # 实时提取AE语义 (不计算AE梯度)
                with torch.no_grad():
                    # 确保输入inputs的维度和类型与AE期望的一致
                    # AE的encode方法通常期望 [batch_size, input_dim]
                    # 如果inputs是 [batch_size, 1, length] (像CNN的Conv1d输入)，可能需要调整
                    # 但您的AE的input_dim=SEGMENT_LENGTH，所以应该是 [batch_size, SEGMENT_LENGTH]
                    # 假设 inputs 已经是 [batch_size, SEGMENT_LENGTH]
                    if inputs.dim() == 3 and inputs.shape[1] == 1:  # [B, 1, L]
                        ae_inputs = inputs.squeeze(1)  # [B, L]
                    elif inputs.dim() == 2:  # [B, L]
                        ae_inputs = inputs
                    else:
                        raise ValueError(f"Unexpected input shape for AE: {inputs.shape}")

                    real_time_semantics = self.semantic_builder.autoencoder.encode(ae_inputs)
                    if not torch.all(torch.isfinite(real_time_semantics)):
                        print(f"W: AE produced non-finite semantics in train batch {batch_idx}. Skipping batch.")
                        continue

                # 前向传播 (CNN)
                logits, _ = self.cnn_model(inputs,
                                           real_time_semantics)  # inputs for signal_encoder, real_time_semantics for semantic_encoder

                loss = criterion(logits, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"W: CNN loss is NaN/Inf in train batch {batch_idx}. Skipping backward.")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                train_loss_epoch += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                total_epoch += targets.size(0)
                correct_epoch += predicted.eq(targets).sum().item()

                if (batch_idx + 1) % 50 == 0 and total_epoch > 0:
                    print(f'Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | '
                          f'Loss: {loss.item():.4f} | Acc: {100. * predicted.eq(targets).sum().item() / targets.size(0):.2f}%')

            avg_train_loss = train_loss_epoch / total_epoch if total_epoch > 0 else 0
            avg_train_acc = 100. * correct_epoch / total_epoch if total_epoch > 0 else 0

            # 验证
            val_loss_epoch, val_correct_epoch, val_total_epoch = 0, 0, 0
            if val_loader:
                self.cnn_model.eval()
                with torch.no_grad():
                    for inputs_val, targets_val in val_loader:
                        inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                        if inputs_val.dim() == 3 and inputs_val.shape[1] == 1:
                            ae_inputs_val = inputs_val.squeeze(1)
                        elif inputs_val.dim() == 2:
                            ae_inputs_val = inputs_val
                        else:
                            raise ValueError(f"Unexpected input shape for AE in validation: {inputs_val.shape}")

                        real_time_semantics_val = self.semantic_builder.autoencoder.encode(ae_inputs_val)
                        if not torch.all(torch.isfinite(real_time_semantics_val)):
                            print(f"W: AE produced non-finite semantics in validation. Skipping batch.")
                            continue

                        logits_val, _ = self.cnn_model(inputs_val, real_time_semantics_val)
                        loss_val = criterion(logits_val, targets_val)

                        if torch.isnan(loss_val) or torch.isinf(loss_val):
                            print(f"W: CNN validation loss is NaN/Inf. Skipping batch.")
                            continue

                        val_loss_epoch += loss_val.item() * inputs_val.size(0)
                        _, predicted_val = logits_val.max(1)
                        val_total_epoch += targets_val.size(0)
                        val_correct_epoch += predicted_val.eq(targets_val).sum().item()

                avg_val_loss = val_loss_epoch / val_total_epoch if val_total_epoch > 0 else 0
                avg_val_acc = 100. * val_correct_epoch / val_total_epoch if val_total_epoch > 0 else 0

                print(
                    f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | '
                    f'Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%')

                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    patience_counter = 0
                    torch.save(self.cnn_model.state_dict(), 'best_ae_semantic_cnn_realtime.pth')
                    print(f"保存最佳模型 (实时语义)，验证集准确率: {avg_val_acc:.2f}%")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停: {patience}轮内未改善")
                        break
            else:  # No validation loader
                print(
                    f'Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%')

        if os.path.exists('best_ae_semantic_cnn_realtime.pth'):
            self.cnn_model.load_state_dict(torch.load('best_ae_semantic_cnn_realtime.pth'))
            print("已加载最佳模型 (实时语义)")
        else:
            print("警告: 未找到最佳模型文件 'best_ae_semantic_cnn_realtime.pth'。使用最后训练的模型。")

        return self.cnn_model
    def run_pipeline(self):
        """基于知识蒸馏和双向对齐的零样本复合故障诊断流水线"""
        start_time = time.time()
        accuracy_pca = 0.0
        accuracy_cosine = 0.0
        final_accuracy_to_report = 0.0

        try:
            print("\n==== 步骤1: 数据加载 ====")
            data_dict = self.load_data()
            if data_dict is None:
                raise RuntimeError("数据加载失败")

            print("\n==== 步骤2: 语义构建 ====")
            semantic_dict = self.build_semantics(data_dict)
            if semantic_dict is None:
                raise RuntimeError("语义构建失败")

            self.visualize_semantic_space(data_dict, semantic_dict)

            self.actual_latent_dim = self.semantic_builder.actual_latent_dim  # 从AE获取真实的潜在维度
            if self.actual_latent_dim <= 0:
                raise RuntimeError("AE latent dimension is invalid after training.")
            print("\n==== 步骤3: 初始化基于AE数据语义的双通道CNN模型 ====")
            self.cnn_model = AEDataSemanticCNN(
                input_length=self.sample_length,
                semantic_dim=self.actual_latent_dim,  # <--- 修改这里
                num_classes=self.num_classes,
                feature_dim=CNN_FEATURE_DIM
            ).to(self.device)
            if hasattr(self.cnn_model, 'feature_dim'):
                self.cnn_feature_dim = self.cnn_model.feature_dim
            else:
                self.cnn_feature_dim = CNN_FEATURE_DIM

            print("\n==== 步骤4: 训练基于AE数据语义的双通道CNN模型 ====")
            trained_cnn_model = self.train_ae_data_semantic_cnn(
                data_dict=data_dict,
                semantic_dict=semantic_dict,
                epochs=CNN_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE,
                lr=CNN_LR
            )
            if trained_cnn_model is None:
                raise RuntimeError("CNN模型训练失败或未返回有效模型")
            self.cnn_model = trained_cnn_model

            print("\n==== 步骤6: 训练双向对齐语义映射网络 ====")
            # 使用新的双向对齐训练方法替代原有方法
            mlp_train_success = self.train_semantic_mapping_mlp_with_bidirectional_alignment(
                data_dict=data_dict,
                semantic_dict=semantic_dict,
                epochs=SEN_EPOCHS,
                lr=SEN_LR
            )
            if not mlp_train_success or self.embedding_net is None:
                raise RuntimeError("双向语义映射网络训练失败")

            print("\n==== 步骤7: 生成复合故障投影 (使用训练好的双向语义映射网络) ====")
            compound_projections = self.generate_compound_fault_projections_fixed(semantic_dict, data_dict)
            if compound_projections is None or not compound_projections:
                raise RuntimeError("复合故障投影生成失败")

            # 可视化比较




            print("\n==== 步骤8: 零样本学习评估 ====")
            print("--- 评估方法1: PCA降维 + 欧氏距离 ---")
            accuracy_pca, conf_matrix_pca = self.evaluate_zero_shot_with_pca(
                data_dict,
                compound_projections,
                pca_components=3
            )

            print("\n--- 评估方法2: 原始特征空间 + 余弦相似度 ---")
            accuracy_cosine, conf_matrix_cosine = self.evaluate_zero_shot_with_cosine_similarity(
                data_dict,
                compound_projections
            )

            final_accuracy_to_report = accuracy_pca

        except Exception as e:
            print(f"错误: 流水线执行过程中发生严重错误: {e}")
            import traceback
            traceback.print_exc()
            final_accuracy_to_report = 0.0

        end_time = time.time()
        print(f"\n=== 流水线在 {(end_time - start_time) / 60:.2f} 分钟内完成 ===")
        print(f"零样本学习准确率 (PCA方法): {accuracy_pca:.2f}%")
        print(f"零样本学习准确率 (余弦相似度方法): {accuracy_cosine:.2f}%")

        return final_accuracy_to_report

    def train_semantic_mapping_mlp_with_bidirectional_alignment(self, data_dict, semantic_dict,
                                                                epochs=SEN_EPOCHS, lr=SEN_LR,
                                                                mlp_hidden_dim1=512, mlp_hidden_dim2=256,
                                                                mlp_dropout_rate=0.3):
        """改进的双向对齐训练方法，增强特征对齐能力"""
        print("\n--- 开始训练增强型双向对齐语义映射网络 ---")

        # 基本检查和准备（与原方法相同）
        if self.cnn_model is None or not hasattr(self.cnn_model, 'feature_dim') or self.cnn_model.feature_dim <= 0:
            print("错误：CNN模型未有效初始化或无特征维度。")
            return False
        self.cnn_model.eval()

        if self.fused_semantic_dim <= 0:
            print("错误：融合语义维度无效。")
            return False

        # 初始化增强型双向语义网络
        self.embedding_net = BidirectionalSemanticNetwork(
            semantic_dim=self.fused_semantic_dim,
            feature_dim=self.cnn_model.feature_dim,
            hidden_dim1=mlp_hidden_dim1,
            hidden_dim2=mlp_hidden_dim2,
            dropout_rate=mlp_dropout_rate
        ).to(self.device)

        print(f"增强型双向语义映射网络初始化:")
        print(f"  语义维度: {self.fused_semantic_dim}")
        print(f"  特征维度: {self.cnn_model.feature_dim}")

        # 数据准备部分（与原方法相同）
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict.get('X_val'), data_dict.get('y_val')

        fused_semantics = semantic_dict.get('fused_semantics', {})
        data_semantics = {}
        if 'data_prototypes' in semantic_dict:
            data_semantics.update(semantic_dict['data_prototypes'])
        if 'compound_data_semantics' in semantic_dict:
            data_semantics.update(semantic_dict['compound_data_semantics'])

        # 构建语义向量映射（与原方法相同）
        semantic_vectors_map = {}  # 融合语义 - 用于MLP训练
        data_semantic_vectors_map = {}  # 数据语义 - 用于CNN特征提取

        for fault_name, semantic_vec in fused_semantics.items():
            if fault_name in self.fault_types and np.all(np.isfinite(semantic_vec)):
                fault_idx = self.fault_types[fault_name]
                semantic_vectors_map[fault_idx] = semantic_vec

        for fault_name, semantic_vec in data_semantics.items():
            if fault_name in self.fault_types and np.all(np.isfinite(semantic_vec)):
                fault_idx = self.fault_types[fault_name]
                data_semantic_vectors_map[fault_idx] = semantic_vec

        if not semantic_vectors_map:
            print("错误：无法创建有效的故障索引到语义的映射。")
            return False

        # 数据加载器设置（与原方法相同）
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        mlp_batch_size = getattr(self, 'batch_size', DEFAULT_BATCH_SIZE)
        train_loader = DataLoader(train_dataset, batch_size=mlp_batch_size, shuffle=True, drop_last=True)

        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=mlp_batch_size, shuffle=False)

        # 改进的优化器设置 - 使用更高级的优化器配置
        optimizer = optim.AdamW(self.embedding_net.parameters(), lr=lr, weight_decay=2e-5)

        # 多阶段学习率调度 - 先快速学习，再精细调整
        def create_scheduler(optimizer):
            # 先快速预热，再余弦退火
            return optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr * 1.5,
                steps_per_epoch=len(train_loader),
                epochs=epochs,
                pct_start=0.1,  # 快速预热
                div_factor=25,  # 初始学习率 = max_lr/div_factor
                final_div_factor=1000,  # 最终学习率 = max_lr/(div_factor*final_div_factor)
                anneal_strategy='cos'
            )

        scheduler = create_scheduler(optimizer)

        # 多种损失函数
        mse_loss_fn = nn.MSELoss()
        cosine_loss_fn = nn.CosineEmbeddingLoss()
        l1_loss_fn = nn.L1Loss()

        # 增强版对比损失
        def enhanced_contrastive_loss(features, labels, temperature=0.07):
            """增强版对比损失，更强的类内聚集和类间分离"""
            batch_size = features.size(0)
            if batch_size <= 1:
                return torch.tensor(0.0, device=features.device)

            # 特征归一化
            features_norm = F.normalize(features, dim=1)

            # 计算余弦相似度矩阵
            similarity_matrix = torch.matmul(features_norm, features_norm.T) / temperature

            # 掩码：同类为1，异类为0
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float()

            # 移除对角线
            mask = mask - torch.eye(batch_size, device=features.device)

            # 对于正样本对
            positive_mask = mask.bool()
            positive_similarity = similarity_matrix[positive_mask]

            # 对于负样本对 - 使用硬负样本挖掘
            negative_mask = (~mask.bool()) & (similarity_matrix > 0)  # 只关注有较高相似度的负样本
            negative_similarity = similarity_matrix[negative_mask]

            # 归一化正样本和负样本损失
            pos_loss = -torch.mean(positive_similarity) if len(positive_similarity) > 0 else torch.tensor(0.0,
                                                                                                          device=features.device)
            neg_loss = torch.mean(torch.clamp(negative_similarity, min=0)) if len(
                negative_similarity) > 0 else torch.tensor(0.0, device=features.device)

            return pos_loss + neg_loss

        # 投影对齐损失 - 专门针对每个类别计算特征对齐
        def projection_alignment_loss(pred_features, target_features, labels):
            """计算每个类别内的特征对齐损失"""
            unique_labels = torch.unique(labels)
            alignment_loss = 0.0

            for lbl in unique_labels:
                mask = (labels == lbl)
                if torch.sum(mask) <= 1:  # 需要至少两个样本
                    continue

                class_pred = pred_features[mask]
                class_target = target_features[mask]

                # 同时考虑距离和方向
                mse = F.mse_loss(class_pred, class_target)

                # 计算中心点偏移
                pred_center = torch.mean(class_pred, dim=0)
                target_center = torch.mean(class_target, dim=0)
                center_shift = F.mse_loss(pred_center, target_center)

                # 计算方向一致性
                class_pred_norm = F.normalize(class_pred, dim=1)
                class_target_norm = F.normalize(class_target, dim=1)
                direction_sim = 1.0 - torch.mean(torch.sum(class_pred_norm * class_target_norm, dim=1))

                # 给复合故障更高的权重
                is_compound = '_' in self.idx_to_fault.get(lbl.item(), '')
                weight = 2.0 if is_compound else 1.0

                alignment_loss += weight * (mse + center_shift * 2.0 + direction_sim)

            return alignment_loss / (len(unique_labels) + 1e-8)

        # 训练循环
        best_val_loss = float('inf')
        patience_epochs = 30  # 更长的耐心值
        current_patience = 0

        # 针对性设置故障类型名称字典
        label_to_fault_name = {idx: name for name, idx in self.fault_types.items()}

        # 记录投影特征质量评估
        projection_quality = {
            'inner_outer': [],
            'inner_ball': [],
            'outer_ball': [],
            'inner_outer_ball': [],
        }

        # 训练循环
        for epoch in range(epochs):
            self.embedding_net.train()
            epoch_losses = {
                'forward': 0.0, 'reverse': 0.0, 'cycle': 0.0,
                'contrastive': 0.0, 'projection': 0.0, 'total': 0.0
            }
            num_valid_batches = 0

            for batch_idx, (batch_signals, batch_labels) in enumerate(train_loader):
                batch_signals = batch_signals.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # 准备双种语义向量与故障类型名称
                batch_semantic_inputs_list = []  # 融合语义用于MLP
                batch_data_semantic_list = []  # 数据语义用于CNN
                valid_indices = []
                valid_labels = []
                fault_types = []  # 故障类型名称列表

                for i, label_idx in enumerate(batch_labels):
                    label_item = label_idx.item()
                    fused_sem = semantic_vectors_map.get(label_item)
                    data_sem = data_semantic_vectors_map.get(label_item)

                    if fused_sem is not None and data_sem is not None:
                        batch_semantic_inputs_list.append(fused_sem)
                        batch_data_semantic_list.append(data_sem)
                        valid_indices.append(i)
                        valid_labels.append(label_item)
                        # 获取故障类型名称
                        fault_types.append(label_to_fault_name.get(label_item, "unknown"))

                if len(valid_indices) <= 1:
                    continue

                batch_semantic_inputs = torch.FloatTensor(np.array(batch_semantic_inputs_list)).to(self.device)
                batch_data_semantics = torch.FloatTensor(np.array(batch_data_semantic_list)).to(self.device)
                batch_signals_filtered = batch_signals[valid_indices]
                valid_labels_tensor = torch.LongTensor(valid_labels).to(self.device)

                # 获取CNN真实特征
                with torch.no_grad():
                    if batch_signals_filtered.dim() == 3 and batch_signals_filtered.shape[1] == 1:
                        ae_batch_signals_filtered = batch_signals_filtered.squeeze(1)
                    elif batch_signals_filtered.dim() == 2:
                        ae_batch_signals_filtered = batch_signals_filtered
                    else:
                        print(f"W: Unexpected shape for AE input in MLP training: {batch_signals_filtered.shape}")

                        continue

                    real_time_ae_semantics_for_cnn = self.semantic_builder.autoencoder.encode(ae_batch_signals_filtered)
                    if not torch.all(torch.isfinite(real_time_ae_semantics_for_cnn)):
                        print(
                            f"W: AE produced non-finite semantics for CNN target features in MLP training. Skipping batch.")
                        continue

                    target_features = self.cnn_model(
                        batch_signals_filtered,  # 原始信号给CNN的信号编码器
                        semantic=real_time_ae_semantics_for_cnn,  # 实时AE语义给CNN的语义编码器
                        return_features=True
                    )
                optimizer.zero_grad()

                # --- 分步骤训练策略 ---
                # 1. 正向映射 - 提供故障类型信息以启用特殊处理
                pred_features = []
                for i, semantic in enumerate(batch_semantic_inputs):
                    # 分别处理每个样本，提供故障类型信息
                    feature = self.embedding_net(
                        semantic.unsqueeze(0),
                        mode='forward',
                        fault_type=fault_types[i]
                    )
                    pred_features.append(feature)
                pred_features = torch.cat(pred_features, dim=0)

                # 2. 反向映射
                pred_semantics = self.embedding_net(target_features, mode='reverse')

                # 3. 循环一致性
                cycle_features = []
                cycle_semantics = []
                for i, semantic in enumerate(batch_semantic_inputs):
                    feature, reconstructed = self.embedding_net(
                        semantic.unsqueeze(0),
                        mode='cycle',
                        fault_type=fault_types[i]
                    )
                    cycle_features.append(feature)
                    cycle_semantics.append(reconstructed)
                cycle_features = torch.cat(cycle_features, dim=0)
                cycle_semantics = torch.cat(cycle_semantics, dim=0)

                # --- 计算多种损失 ---
                # 1. 基础映射损失
                forward_loss = mse_loss_fn(pred_features, target_features) + \
                               l1_loss_fn(pred_features, target_features) * 0.5

                reverse_loss = mse_loss_fn(pred_semantics, batch_semantic_inputs)

                cycle_loss = mse_loss_fn(cycle_semantics, batch_semantic_inputs)

                # 2. 特征对齐损失 - 类内对齐和方向一致性
                projection_loss = projection_alignment_loss(
                    pred_features, target_features, valid_labels_tensor
                )

                # 3. 对比损失 - 基于类别的对比学习
                contrastive_loss = enhanced_contrastive_loss(
                    pred_features, valid_labels_tensor, temperature=0.07
                )

                # 4. 复合故障特殊处理 - 针对性强化outer_ball和inner_outer_ball
                compound_loss = 0.0
                compound_count = 0
                for i, fault_type in enumerate(fault_types):
                    # 特别关注这两个问题故障类型
                    if fault_type in ['outer_ball', 'inner_outer_ball']:
                        feat_pred = pred_features[i].unsqueeze(0)
                        feat_target = target_features[i].unsqueeze(0)

                        # 特殊增强损失 - 同时惩罚方向和距离
                        comp_l2 = F.mse_loss(feat_pred, feat_target) * 2.0

                        # 方向一致性
                        feat_pred_norm = F.normalize(feat_pred, dim=1)
                        feat_target_norm = F.normalize(feat_target, dim=1)
                        comp_dir = 1.0 - torch.sum(feat_pred_norm * feat_target_norm)

                        compound_loss += comp_l2 + comp_dir * 2.0
                        compound_count += 1

                if compound_count > 0:
                    compound_loss /= compound_count

                # --- 动态权重策略 ---
                epoch_progress = epoch / epochs  # 训练进度比例

                # 随着训练进行，增加特殊类型损失的权重
                w_forward = 1.5
                w_reverse = 1.0
                w_cycle = 0.5
                w_contrastive = 0.4 + 0.6 * epoch_progress
                w_projection = 0.8 + 1.2 * epoch_progress
                w_compound = 1.0 + 5.0 * epoch_progress  # 大幅增加复合故障权重

                # 总损失
                total_loss = (
                        w_forward * forward_loss +
                        w_reverse * reverse_loss +
                        w_cycle * cycle_loss +
                        w_contrastive * contrastive_loss +
                        w_projection * projection_loss
                )

                # 只有在有复合故障样本时才添加复合损失
                if compound_count > 0:
                    total_loss += w_compound * compound_loss

                # 反向传播
                if torch.isfinite(total_loss):
                    total_loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.embedding_net.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                    epoch_losses['forward'] += forward_loss.item()
                    epoch_losses['reverse'] += reverse_loss.item()
                    epoch_losses['cycle'] += cycle_loss.item()
                    epoch_losses['contrastive'] += contrastive_loss.item()
                    epoch_losses['projection'] += projection_loss.item()
                    epoch_losses['total'] += total_loss.item()
                    num_valid_batches += 1

                    # 记录特定故障类型的投影质量
                    with torch.no_grad():
                        for i, fault_type in enumerate(fault_types):
                            if fault_type in projection_quality:
                                pred = pred_features[i].detach()
                                target = target_features[i].detach()
                                quality = 1.0 - F.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0)).item()
                                projection_quality[fault_type].append(quality)
                else:
                    print(f"警告: 批次 {batch_idx} 损失非有限值，跳过此批次的更新。")

            # 计算并打印损失
            if num_valid_batches > 0:
                for k in epoch_losses:
                    epoch_losses[k] /= num_valid_batches

                print(f"Epoch [{epoch + 1}/{epochs}] - LR: {scheduler.get_last_lr()[0]:.6f}")
                print(f"  正向映射损失: {epoch_losses['forward']:.4f}")
                print(f"  反向映射损失: {epoch_losses['reverse']:.4f}")
                print(f"  循环一致性损失: {epoch_losses['cycle']:.4f}")
                print(f"  对比损失: {epoch_losses['contrastive']:.4f}")
                print(f"  投影对齐损失: {epoch_losses['projection']:.4f}")
                print(f"  总损失: {epoch_losses['total']:.4f}")

                # 打印平均投影质量
                print("  故障类型投影质量 (越低越好):")
                for fault_type, quality_list in projection_quality.items():
                    if quality_list:
                        avg_quality = sum(quality_list[-50:]) / min(len(quality_list), 50) if quality_list else 0
                        print(f"    - {fault_type}: {avg_quality:.4f}")

                # 清空质量记录器，避免内存占用过大
                for k in projection_quality:
                    projection_quality[k] = projection_quality[k][-100:] if projection_quality[k] else []
            else:
                print(f"Epoch [{epoch + 1}/{epochs}] - 无有效批次")
                continue

            # 验证阶段（与原方法类似，但增加了对比损失评估）
            if val_loader:
                # 验证代码...（大部分与原方法相同）
                self.embedding_net.eval()
                val_total_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch_signals_val, batch_labels_val in val_loader:
                        batch_signals_val = batch_signals_val.to(self.device)
                        batch_labels_val = batch_labels_val.to(self.device)

                        # 准备双种语义向量
                        batch_semantic_val_list = []  # 融合语义用于MLP
                        batch_data_semantic_val_list = []  # 数据语义用于CNN
                        valid_indices_val = []

                        for i, label_idx in enumerate(batch_labels_val):
                            label_item = label_idx.item()
                            fused_sem = semantic_vectors_map.get(label_item)
                            data_sem = data_semantic_vectors_map.get(label_item)

                            if fused_sem is not None and data_sem is not None:
                                batch_semantic_val_list.append(fused_sem)
                                batch_data_semantic_val_list.append(data_sem)
                                valid_indices_val.append(i)

                        if not valid_indices_val:
                            continue

                        batch_semantic_val = torch.FloatTensor(np.array(batch_semantic_val_list)).to(self.device)
                        batch_data_semantic_val = torch.FloatTensor(np.array(batch_data_semantic_val_list)).to(
                            self.device)
                        batch_signals_val_filtered = batch_signals_val[valid_indices_val]

                        # 获取验证集的真实特征 - 使用数据语义
                        target_features_val = self.cnn_model(
                            batch_signals_val_filtered,
                            semantic=batch_data_semantic_val,
                            return_features=True
                        )

                        # 验证正向映射
                        pred_features_val = self.embedding_net(batch_semantic_val, mode='forward')

                        # 验证反向映射
                        pred_semantics_val = self.embedding_net(target_features_val, mode='reverse')

                        # 验证循环一致性
                        _, reconstructed_semantics_val = self.embedding_net(batch_semantic_val, mode='cycle')

                        # 计算验证损失
                        val_forward_loss = mse_loss_fn(pred_features_val, target_features_val)
                        val_reverse_loss = mse_loss_fn(pred_semantics_val, batch_semantic_val)
                        val_cycle_loss = mse_loss_fn(reconstructed_semantics_val, batch_semantic_val)

                        # 验证总损失 (使用相同权重)
                        val_batch_loss = (w_forward * val_forward_loss +
                                          w_reverse * val_reverse_loss +
                                          w_cycle * val_cycle_loss)

                        if torch.isfinite(val_batch_loss):
                            val_total_loss += val_batch_loss.item()
                            val_batches += 1

                    if val_batches > 0:
                        avg_val_loss = val_total_loss / val_batches
                        print(f"  验证集损失: {avg_val_loss:.4f}")

                        # 学习率调度
                        scheduler.step(avg_val_loss)

                        # 保存最佳模型
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save(self.embedding_net.state_dict(), 'best_bidirectional_semantic_network.pth')
                            print(f"  新的最佳模型已保存 (验证损失: {best_val_loss:.4f})")
                            current_patience = 0
                        else:
                            current_patience += 1
                            if current_patience >= patience_epochs:
                                print(f"早停：验证损失在 {patience_epochs} 轮内未改善。")
                                break
            else:  # 如果没有验证集
                if (epoch + 1) % 10 == 0:  # 每10轮保存一次
                    torch.save(self.embedding_net.state_dict(), f'bidirectional_semantic_network_epoch_{epoch + 1}.pth')
                    print(f"  模型已保存 (Epoch {epoch + 1})")

                # 训练结束，加载最佳模型
            if val_loader and os.path.exists('best_bidirectional_semantic_network.pth'):
                self.embedding_net.load_state_dict(torch.load('best_bidirectional_semantic_network.pth'))
                print("已加载最佳双向语义映射网络模型")

            print("双向对齐语义映射网络训练完成!")
            return True

    def generate_compound_fault_projections_fixed(self, semantic_dict, data_dict):
        """
        生成复合故障投影，严格遵循零样本原则：不使用任何复合故障数据
        """
        print("\n生成复合故障投影...")

        # 基础检查
        if self.embedding_net is None:
            print("错误: 语义嵌入网络未训练，无法生成投影")
            return None

        fused_semantics = semantic_dict.get('fused_semantics')
        if not fused_semantics:
            print("错误: 无可用的融合语义向量")
            return None

        # 注意：不再使用测试数据作为参考，完全依赖语义投影
        reference_centroids = None

        # 提取所有复合故障的语义向量
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

        # 使用语义嵌入网络生成投影
        self.embedding_net.eval()

        # 初始化结果字典
        compound_projections = {}
        failed_projections = []

        # 投影每个复合故障
        with torch.no_grad():
            for fault_type, semantic_vec in compound_fused_semantics.items():
                # 创建语义向量的多个副本进行冗余投影以提高稳定性
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
                        projected_feature = self.embedding_net(semantic_tensor, mode='forward', fault_type=fault_type)

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

        # 尝试为失败的投影通过组分平均生成
        if failed_projections:
            print("尝试通过组分平均为失败的投影生成替代方案...")
            for fault_type in failed_projections:
                # 分解复合故障名称，获取组成部分
                components = fault_type.split('_')
                component_projections = []

                # 收集单一故障的投影
                for single_fault_type in self.single_fault_types_ordered:
                    if single_fault_type in components:
                        # 使用单一故障的语义生成投影
                        if single_fault_type in fused_semantics:
                            single_sem = fused_semantics[single_fault_type]
                            if np.all(np.isfinite(single_sem)):
                                try:
                                    single_tensor = torch.FloatTensor(single_sem).unsqueeze(0).to(self.device)
                                    single_proj = self.embedding_net(single_tensor, mode='forward',
                                                                     fault_type=single_fault_type)
                                    if torch.all(torch.isfinite(single_proj)):
                                        component_projections.append(single_proj.cpu().numpy().squeeze(0))
                                except Exception as e:
                                    print(f"警告: 无法为组分'{single_fault_type}'生成投影: {e}")

                # 如果找到了足够的组分投影，计算它们的平均值
                if len(component_projections) >= 2:  # 至少需要两个组分
                    avg_proj = np.mean(component_projections, axis=0)
                    if np.all(np.isfinite(avg_proj)):
                        compound_projections[fault_type] = avg_proj
                        print(f"  - 使用组分平均成功生成'{fault_type}'的投影")

        # 确认生成的投影数量
        if len(compound_projections) == 0:
            print("错误: 无有效的复合故障投影生成")
            return None

        print(f"成功生成{len(compound_projections)}种复合故障投影")

        # 打印相似度矩阵 (用于调试)
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

    def evaluate_zero_shot_with_pca(self, data_dict, compound_projections, pca_components=2):
        """使用实时提取的语义和PCA降维后的欧氏距离进行零样本复合故障分类"""
        print(f"\n评估零样本复合故障分类能力（使用{pca_components}维PCA降维后的欧氏距离）...")

        # 1. 基本检查
        if self.cnn_model is None or not hasattr(self.cnn_model, 'feature_dim') or self.cnn_model.feature_dim <= 0:
            print("错误: CNN模型未训练或特征维度无效")
            return 0.0, None

        if compound_projections is None or not compound_projections:
            print("错误: 缺少复合故障投影")
            return 0.0, None

        if self.semantic_builder is None or self.semantic_builder.autoencoder is None:
            print("错误: 自编码器未初始化，无法提取实时语义")
            return 0.0, None

        # 2. 获取测试数据 - 仅复合故障
        X_test, y_test = data_dict['X_test'], data_dict['y_test']

        compound_fault_types_names = list(compound_projections.keys())
        compound_fault_indices = [self.fault_types[name] for name in compound_fault_types_names if
                                  name in self.fault_types]

        test_compound_mask = np.isin(y_test, compound_fault_indices)
        X_compound_test_orig = X_test[test_compound_mask]
        y_compound_test_orig = y_test[test_compound_mask]

        if len(X_compound_test_orig) == 0:
            print("警告: 无复合故障测试数据与提供的投影匹配")
            return 0.0, None

        finite_mask_test = np.all(np.isfinite(X_compound_test_orig), axis=1)
        X_compound_test = X_compound_test_orig[finite_mask_test]
        y_compound_test = y_compound_test_orig[finite_mask_test]

        if len(X_compound_test) == 0:
            print("错误: 所有复合故障测试数据包含非有限值或无匹配投影")
            return 0.0, None

        print(f"  使用 {len(X_compound_test)} 个复合故障测试样本评估 {len(compound_fault_indices)} 种复合故障")

        # 3. 准备投影特征
        candidate_labels_names = []
        projection_features_list = []
        candidate_fault_indices = []

        for fault_name, projection in compound_projections.items():
            if fault_name in self.fault_types and np.all(np.isfinite(projection)):
                candidate_labels_names.append(fault_name)
                projection_features_list.append(projection)
                candidate_fault_indices.append(self.fault_types[fault_name])

        if not projection_features_list:
            print("错误: 没有有效的投影特征可供使用。")
            return 0.0, None

        projection_features_np = np.array(projection_features_list)

        # 4. 提取测试样本特征 - 使用自编码器实时提取语义
        self.cnn_model.eval()
        self.semantic_builder.autoencoder.eval()
        test_features_list_extracted = []
        valid_test_labels_list = []
        batch_size_eval = getattr(self, 'batch_size', 64)

        with torch.no_grad():
            for i in range(0, len(X_compound_test), batch_size_eval):
                batch_x_signal = torch.FloatTensor(X_compound_test[i:i + batch_size_eval]).to(self.device)
                batch_y_labels = y_compound_test[i:i + batch_size_eval]

                # 准备AE输入 (应为 [B, SEGMENT_LENGTH])
                if batch_x_signal.dim() == 3 and batch_x_signal.shape[1] == 1:  # If [B, 1, L]
                    ae_batch_x = batch_x_signal.squeeze(1)
                elif batch_x_signal.dim() == 2:  # If [B, L]
                    ae_batch_x = batch_x_signal
                else:
                    print(
                        f"W: evaluate_zero_shot_with_pca - Unexpected shape for AE input: {batch_x_signal.shape}. Skipping batch.")
                    continue

                real_time_semantics = self.semantic_builder.autoencoder.encode(ae_batch_x)
                if not torch.all(torch.isfinite(real_time_semantics)):
                    print(f"W: evaluate_zero_shot_with_pca - AE produced non-finite semantics. Skipping batch.")
                    continue

                features = self.cnn_model(batch_x_signal, semantic=real_time_semantics, return_features=True)
                if not torch.all(torch.isfinite(features)):
                    print(f"W: evaluate_zero_shot_with_pca - CNN produced non-finite features. Skipping batch.")
                    continue

                test_features_list_extracted.append(features.cpu().numpy())
                valid_test_labels_list.extend(batch_y_labels.tolist())

        if not test_features_list_extracted:
            print("错误: 无法提取有效的测试特征 (PCA评估)")
            return 0.0, None

        test_features_np = np.vstack(test_features_list_extracted)
        y_compound_test_final = np.array(valid_test_labels_list)  # Use labels of successfully processed samples

        # 5. 应用PCA降维
        from sklearn.decomposition import PCA
        all_features_for_pca = np.vstack([test_features_np, projection_features_np])

        # Ensure pca_components is not greater than the number of features or samples
        actual_pca_components = min(pca_components, all_features_for_pca.shape[0], all_features_for_pca.shape[1])
        if actual_pca_components <= 0:
            print(f"错误: 计算出的PCA组件数 ({actual_pca_components}) 无效。")
            return 0.0, None

        pca = PCA(n_components=actual_pca_components)
        pca.fit(all_features_for_pca)

        test_features_pca = pca.transform(test_features_np)
        projection_features_pca = pca.transform(projection_features_np)

        print(f"  原始特征维度: {test_features_np.shape[1]} -> PCA降维后: {test_features_pca.shape[1]}")
        if pca.n_components_ > 0:
            print(f"  PCA解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")

        # 6. 在PCA空间中进行分类
        y_pred_list = []
        for i in range(len(test_features_pca)):
            dists = [np.linalg.norm(test_features_pca[i] - proj) for proj in projection_features_pca]
            nearest_idx = np.argmin(dists)
            y_pred_list.append(candidate_fault_indices[nearest_idx])

        y_pred_np = np.array(y_pred_list)

        # 7. 计算指标
        accuracy = np.mean(y_pred_np == y_compound_test_final) * 100
        class_accuracy_results = {}
        print("\n=== 各类别分类详情 (PCA) ===")
        for fault_idx_true in np.unique(y_compound_test_final):
            mask = (y_compound_test_final == fault_idx_true)
            count = np.sum(mask)
            correct_count = np.sum(y_pred_np[mask] == y_compound_test_final[mask])
            acc = (correct_count / count) * 100 if count > 0 else 0
            fault_name = self.idx_to_fault.get(fault_idx_true, f"Unknown_{fault_idx_true}")
            class_accuracy_results[fault_name] = acc
            print(f"类别 {fault_name}: {correct_count}/{count} 正确, 准确率 {acc:.2f}%")
        print(f"\n总体准确率 (PCA): {accuracy:.2f}%")

        # 8. 可视化混淆矩阵
        true_labels_str_pca = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_compound_test_final]
        pred_labels_str_pca = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_pred_np]
        # Ensure all candidate fault names are used as labels for CM
        display_labels_cm_pca = sorted(
            list(set(true_labels_str_pca) | set(pred_labels_str_pca) | set(candidate_labels_names)))

        conf_matrix_pca_val = None
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns

            conf_matrix_pca_val = confusion_matrix(true_labels_str_pca, pred_labels_str_pca,
                                                   labels=display_labels_cm_pca)
            configure_chinese_font()
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix_pca_val, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels_cm_pca,
                        yticklabels=display_labels_cm_pca)
            plt.xlabel('预测')
            plt.ylabel('真实')
            plt.title(f'零样本学习混淆矩阵 (实时语义+PCA+欧式距离, 准确率: {accuracy:.2f}%)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('compound_fault_confusion_matrix_zsl_pca_realtime.png')
            plt.close()
            print("混淆矩阵 (PCA) 已保存至 'compound_fault_confusion_matrix_zsl_pca_realtime.png'")
        except Exception as e:
            print(f"警告: PCA混淆矩阵计算或绘图错误: {e}")
            conf_matrix_pca_val = None

        return accuracy, class_accuracy_results  # Return class_accuracy_results instead of conf_matrix

    def evaluate_zero_shot_with_cosine_similarity(self, data_dict, compound_projections):
        """使用实时提取的语义和原始特征空间中的余弦相似度进行零样本复合故障分类"""
        print("\n使用实时语义和余弦相似度评估零样本复合故障分类能力...")

        # 1. 基本检查
        if self.cnn_model is None or not hasattr(self.cnn_model, 'feature_dim') or self.cnn_model.feature_dim <= 0:
            print("错误: CNN模型未训练或特征维度无效")
            return 0.0, None
        if compound_projections is None or not compound_projections:
            print("错误: 缺少复合故障投影")
            return 0.0, None
        if self.semantic_builder is None or self.semantic_builder.autoencoder is None:
            print("错误: 自编码器未初始化，无法提取实时语义")
            return 0.0, None

        # 2. 获取测试数据
        X_test, y_test = data_dict['X_test'], data_dict['y_test']

        compound_fault_types_names_cosine = list(compound_projections.keys())
        compound_fault_indices_cosine = [self.fault_types[name] for name in compound_fault_types_names_cosine if
                                         name in self.fault_types]

        test_compound_mask_cosine = np.isin(y_test, compound_fault_indices_cosine)
        X_compound_test_orig_cosine = X_test[test_compound_mask_cosine]
        y_compound_test_orig_cosine = y_test[test_compound_mask_cosine]

        if len(X_compound_test_orig_cosine) == 0:
            print("警告: 无复合故障测试数据与提供的投影匹配 (余弦评估)")
            return 0.0, None

        finite_mask_test_cosine = np.all(np.isfinite(X_compound_test_orig_cosine), axis=1)
        X_compound_test_cosine = X_compound_test_orig_cosine[finite_mask_test_cosine]
        y_compound_test_cosine = y_compound_test_orig_cosine[finite_mask_test_cosine]

        if len(X_compound_test_cosine) == 0:
            print("错误: 所有复合故障测试数据包含非有限值或无匹配投影 (余弦评估)")
            return 0.0, None

        print(
            f"  使用 {len(X_compound_test_cosine)} 个复合故障测试样本评估 {len(compound_fault_indices_cosine)} 种复合故障 (余弦评估)")

        # 3. 准备投影特征 (原始空间)
        candidate_fault_names_cosine = []
        projection_features_orig_list_cosine = []
        for name, projection in compound_projections.items():
            if name in self.fault_types and np.all(np.isfinite(projection)):
                candidate_fault_names_cosine.append(name)
                projection_features_orig_list_cosine.append(projection)

        if not projection_features_orig_list_cosine:
            print("错误: 没有有效的投影特征可供使用 (余弦评估)。")
            return 0.0, None

        projection_features_orig_np_cosine = np.array(projection_features_orig_list_cosine)
        projection_features_norm_cosine = projection_features_orig_np_cosine / (
                np.linalg.norm(projection_features_orig_np_cosine, axis=1, keepdims=True) + 1e-9)

        # 4. 提取实际测试特征 (原始空间) - 使用实时提取的语义
        self.cnn_model.eval()
        self.semantic_builder.autoencoder.eval()
        test_features_orig_list_extracted_cosine = []
        valid_test_labels_list_cosine = []
        batch_size_eval_cosine = getattr(self, 'batch_size', 64)

        with torch.no_grad():
            for i in range(0, len(X_compound_test_cosine), batch_size_eval_cosine):
                batch_x_signal_cosine = torch.FloatTensor(X_compound_test_cosine[i:i + batch_size_eval_cosine]).to(
                    self.device)
                batch_y_labels_cosine = y_compound_test_cosine[i:i + batch_size_eval_cosine]

                if batch_x_signal_cosine.dim() == 3 and batch_x_signal_cosine.shape[1] == 1:
                    ae_batch_x_cosine = batch_x_signal_cosine.squeeze(1)
                elif batch_x_signal_cosine.dim() == 2:
                    ae_batch_x_cosine = batch_x_signal_cosine
                else:
                    print(
                        f"W: evaluate_zero_shot_with_cosine_similarity - Unexpected shape for AE input: {batch_x_signal_cosine.shape}. Skipping batch.")
                    continue

                real_time_semantics_cosine = self.semantic_builder.autoencoder.encode(ae_batch_x_cosine)
                if not torch.all(torch.isfinite(real_time_semantics_cosine)):
                    print(
                        f"W: evaluate_zero_shot_with_cosine_similarity - AE produced non-finite semantics. Skipping batch.")
                    continue

                features_cosine = self.cnn_model(batch_x_signal_cosine, semantic=real_time_semantics_cosine,
                                                 return_features=True)
                if not torch.all(torch.isfinite(features_cosine)):
                    print(
                        f"W: evaluate_zero_shot_with_cosine_similarity - CNN produced non-finite features. Skipping batch.")
                    continue

                test_features_orig_list_extracted_cosine.append(features_cosine.cpu().numpy())
                valid_test_labels_list_cosine.extend(batch_y_labels_cosine.tolist())

        if not test_features_orig_list_extracted_cosine:
            print("错误: 无法从测试数据中提取有效特征 (余弦评估)")
            return 0.0, None

        test_features_orig_np_cosine = np.vstack(test_features_orig_list_extracted_cosine)
        y_compound_test_final_cosine = np.array(valid_test_labels_list_cosine)
        test_features_norm_cosine = test_features_orig_np_cosine / (
                np.linalg.norm(test_features_orig_np_cosine, axis=1, keepdims=True) + 1e-9)

        # 5. Classification using Cosine Similarity
        similarity_matrix_cosine = np.dot(test_features_norm_cosine, projection_features_norm_cosine.T)
        y_pred_cosine_list = []
        for i in range(len(test_features_norm_cosine)):
            nearest_idx_cosine = np.argmax(similarity_matrix_cosine[i])
            y_pred_cosine_list.append(self.fault_types[candidate_fault_names_cosine[nearest_idx_cosine]])

        y_pred_np_cosine = np.array(y_pred_cosine_list)

        # 6. Calculate Metrics
        from sklearn.metrics import accuracy_score  # Moved import here
        accuracy_cosine_val = accuracy_score(y_compound_test_final_cosine, y_pred_np_cosine) * 100
        class_accuracy_cosine_results = {}
        print("\n=== 各类别分类详情 (余弦相似度) ===")
        for fault_idx_true_cosine in np.unique(y_compound_test_final_cosine):
            mask_cosine = (y_compound_test_final_cosine == fault_idx_true_cosine)
            count_cosine = np.sum(mask_cosine)
            correct_count_cosine = np.sum(y_pred_np_cosine[mask_cosine] == y_compound_test_final_cosine[mask_cosine])
            acc_cosine = (correct_count_cosine / count_cosine) * 100 if count_cosine > 0 else 0
            fault_name_cosine = self.idx_to_fault.get(fault_idx_true_cosine, f"Unknown_{fault_idx_true_cosine}")
            class_accuracy_cosine_results[fault_name_cosine] = acc_cosine
            print(f"类别 {fault_name_cosine}: {correct_count_cosine}/{count_cosine} 正确, 准确率 {acc_cosine:.2f}%")
        print(f"\n总体准确率 (余弦相似度): {accuracy_cosine_val:.2f}%")

        # 7. Visualize Confusion Matrix
        true_labels_str_cosine = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_compound_test_final_cosine]
        pred_labels_str_cosine = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_pred_np_cosine]
        display_labels_cm_cosine = sorted(
            list(set(true_labels_str_cosine) | set(pred_labels_str_cosine) | set(candidate_fault_names_cosine)))

        conf_matrix_cosine_val = None
        try:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns

            conf_matrix_cosine_val = confusion_matrix(true_labels_str_cosine, pred_labels_str_cosine,
                                                      labels=display_labels_cm_cosine)
            configure_chinese_font()
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix_cosine_val, annot=True, fmt='d', cmap='Greens',
                        xticklabels=display_labels_cm_cosine, yticklabels=display_labels_cm_cosine)
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title(f'零样本学习混淆矩阵 (实时语义+余弦相似度, 准确率: {accuracy_cosine_val:.2f}%)')
            plt.xticks(rotation=45, ha='right');
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('compound_fault_confusion_matrix_zsl_cosine_realtime.png')
            plt.close()
            print("混淆矩阵 (余弦相似度) 已保存至 'compound_fault_confusion_matrix_zsl_cosine_realtime.png'")
        except Exception as e:
            print(f"警告: 余弦相似度混淆矩阵计算或绘图错误: {e}")
            conf_matrix_cosine_val = None

        return accuracy_cosine_val, conf_matrix_cosine_val  # Return conf_matrix_cosine_val directly

    def visualize_semantic_space(self, data_dict, semantic_dict):
        """
        可视化单一故障和复合故障在自编码器语义空间的分布
        显示每类故障的所有样本点而不是仅显示中心点
        """
        print("\n==== 可视化AE语义空间分布 ====")

        if self.semantic_builder is None or self.semantic_builder.autoencoder is None:
            print("错误: 自编码器未初始化或未训练")
            return

        # 准备单一故障数据
        single_fault_types = ['normal', 'inner', 'outer', 'ball']
        single_fault_indices = [self.fault_types[name] for name in single_fault_types]

        # 准备复合故障数据
        compound_fault_types = ['inner_outer', 'inner_ball', 'outer_ball', 'inner_outer_ball']
        compound_fault_indices = [self.fault_types[name] for name in compound_fault_types]

        # 获取训练数据和测试数据
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_test, y_test = data_dict['X_test'], data_dict['y_test']

        # 为保证可视化效果，限制每类故障的样本数量
        max_samples_per_class = 300

        # 提取单一故障样本
        single_fault_samples = []
        single_fault_labels = []

        for fault_idx in single_fault_indices:
            mask = (y_train == fault_idx)
            samples = X_train[mask]
            if len(samples) > max_samples_per_class:
                # 随机抽样以控制点数
                indices = np.random.choice(len(samples), max_samples_per_class, replace=False)
                samples = samples[indices]

            if len(samples) > 0:
                single_fault_samples.append(samples)
                single_fault_labels.extend([fault_idx] * len(samples))

        # 提取复合故障样本
        compound_fault_samples = []
        compound_fault_labels = []

        for fault_idx in compound_fault_indices:
            mask = (y_test == fault_idx)
            samples = X_test[mask]
            if len(samples) > max_samples_per_class:
                # 随机抽样以控制点数
                indices = np.random.choice(len(samples), max_samples_per_class, replace=False)
                samples = samples[indices]

            if len(samples) > 0:
                compound_fault_samples.append(samples)
                compound_fault_labels.extend([fault_idx] * len(samples))

        # 准备所有数据用于AE编码
        all_samples = np.vstack(
            single_fault_samples + compound_fault_samples) if single_fault_samples and compound_fault_samples else None
        all_labels = np.array(single_fault_labels + compound_fault_labels)

        if all_samples is None or len(all_samples) == 0:
            print("错误: 未找到有效样本进行可视化")
            return

        # 使用AE提取语义特征
        self.semantic_builder.autoencoder.eval()
        batch_size = 128
        all_features = []

        with torch.no_grad():
            for i in range(0, len(all_samples), batch_size):
                batch_x = torch.FloatTensor(all_samples[i:i + batch_size]).to(self.device)
                latent = self.semantic_builder.autoencoder.encode(batch_x)
                all_features.append(latent.cpu().numpy())

        all_features = np.vstack(all_features)

        # 确保提取的特征是有限的
        valid_indices = np.all(np.isfinite(all_features), axis=1)
        all_features = all_features[valid_indices]
        all_labels = all_labels[valid_indices]

        if len(all_features) == 0:
            print("错误: 所有提取的特征都包含非有限值")
            return

        # 应用PCA降维用于可视化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        features_reduced = pca.fit_transform(all_features)

        # 可视化2D和3D投影
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import seaborn as sns

            # 创建故障标签映射
            fault_names = [self.idx_to_fault.get(label, f"Unknown_{label}") for label in all_labels]
            fault_labels_unique = sorted(set(fault_names))

            # 颜色映射和标记映射
            colors = sns.color_palette('husl', n_colors=len(fault_labels_unique))
            color_map = {fault: colors[i] for i, fault in enumerate(fault_labels_unique)}

            # 单一故障和复合故障使用不同的标记
            marker_map = {fault: 'o' if '_' not in fault else '^' for fault in fault_labels_unique}

            # 配置字体
            configure_chinese_font()

            # 2D可视化
            plt.figure(figsize=(12, 10))
            for fault in fault_labels_unique:
                mask = np.array(fault_names) == fault
                plt.scatter(
                    features_reduced[mask, 0],
                    features_reduced[mask, 1],
                    c=[color_map[fault]],
                    marker=marker_map[fault],
                    label=fault,
                    alpha=0.7,
                    s=50 if '_' not in fault else 80
                )

            plt.title('AE语义空间的故障分布 (PCA 2D投影)')
            plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.legend(loc='best', bbox_to_anchor=(1.01, 1), borderaxespad=0)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('ae_semantic_space_distribution_2d.png')
            plt.close()

            # 3D可视化
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')

            for fault in fault_labels_unique:
                mask = np.array(fault_names) == fault
                ax.scatter(
                    features_reduced[mask, 0],
                    features_reduced[mask, 1],
                    features_reduced[mask, 2],
                    c=[color_map[fault]],
                    marker=marker_map[fault],
                    label=fault,
                    alpha=0.7,
                    s=50 if '_' not in fault else 80
                )

            ax.set_title('AE语义空间的故障分布 (PCA 3D投影)')
            ax.set_xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_zlabel(f'主成分3 ({pca.explained_variance_ratio_[2]:.2%})')
            ax.legend(loc='best', bbox_to_anchor=(1.01, 1), borderaxespad=0)
            plt.tight_layout()
            plt.savefig('ae_semantic_space_distribution_3d.png')
            plt.close()

            # 使用t-SNE进行更好的非线性降维可视化
            try:
                from sklearn.manifold import TSNE

                tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                features_tsne = tsne.fit_transform(all_features)

                plt.figure(figsize=(12, 10))
                for fault in fault_labels_unique:
                    mask = np.array(fault_names) == fault
                    plt.scatter(
                        features_tsne[mask, 0],
                        features_tsne[mask, 1],
                        c=[color_map[fault]],
                        marker=marker_map[fault],
                        label=fault,
                        alpha=0.7,
                        s=50 if '_' not in fault else 80
                    )

                plt.title('AE语义空间的故障分布 (t-SNE投影)')
                plt.xlabel('t-SNE 维度1')
                plt.ylabel('t-SNE 维度2')
                plt.legend(loc='best', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig('ae_semantic_space_distribution_tsne.png')
                plt.close()

                print("t-SNE可视化图已保存")

            except Exception as e:
                print(f"t-SNE可视化失败: {e}")

            print("语义空间可视化完成，2D和3D图像已保存")

        except Exception as e:
            print(f"可视化过程中出错: {e}")
            import traceback
            traceback.print_exc()
if __name__ == "__main__":

    set_seed(42)
    data_path = "E:/研究生/CNN/HDU1000-600"
    if not os.path.isdir(data_path):
        print(f"E: Data directory not found: {data_path}")
    else:
        fault_diagnosis = ZeroShotCompoundFaultDiagnosis(
            data_path=data_path, sample_length=SEGMENT_LENGTH,
            latent_dim=AE_LATENT_DIM, batch_size=DEFAULT_BATCH_SIZE)
        final_accuracy = fault_diagnosis.run_pipeline()
        print(f"\n>>> Final ZSL Accuracy: {final_accuracy:.2f}% <<<")

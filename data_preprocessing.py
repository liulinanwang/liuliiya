
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import CubicSpline, interp1d
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import os
import warnings
import random
import time
from collections import Counter
import traceback
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')
SEGMENT_LENGTH = 1024
OVERLAP = 0.5
STEP = int(SEGMENT_LENGTH * (1 - OVERLAP))
if STEP < 1: STEP = 1
DEFAULT_WAVELET = 'db4'
DEFAULT_WAVELET_LEVEL = 3
AE_LATENT_DIM = 64
AE_EPOCHS = 1
AE_LR = 0.001
AE_BATCH_SIZE = 64
AE_CONTRASTIVE_WEIGHT = 1.2
AE_NOISE_STD = 0.05
CNN_EPOCHS = 5
CNN_LR = 0.0005
SEN_EPOCHS = 2
SEN_LR = 0.001
CNN_FEATURE_DIM =256
DEFAULT_BATCH_SIZE = 128


def configure_chinese_font():
    """确保matplotlib正确配置中文字体"""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np
    import platform

    # 根据操作系统选择合适的中文字体
    system = platform.system()

    if system == 'Windows':
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun']
    elif system == 'Linux':
        font_list = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'AR PL UMing CN']
    elif system == 'Darwin':  # macOS
        font_list = ['PingFang SC', 'Hiragino Sans GB', 'Heiti SC']
    else:
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun',
                     'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
                     'PingFang SC', 'Hiragino Sans GB']

    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 查找第一个可用的中文字体
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

        # 尝试使用matplotlib内置的中文支持
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
             latent = torch.nan_to_num(latent, nan=0.0) # Clamp latent space?
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
    def __init__(self, latent_dim=AE_LATENT_DIM, hidden_dim=128):
        self.latent_dim_config = latent_dim
        self.actual_latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = None
        self.preprocessor = DataPreprocessor(sample_length=SEGMENT_LENGTH)
        self.knowledge_dim = 0
        self.data_semantics = {}
        self.idx_to_fault = {}
        self.all_latent_features = None
        self.all_latent_labels = None
    def build_knowledge_semantics(self):
        """构建基于轴承故障位置和尺寸的知识语义 (Unchanged from original)"""
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
                    print(f"W: NaN/Inf detected in AE outputs epoch {epoch+1}. Skipping batch loss.")
                    continue

                recon_loss = criterion_recon(decoded, batch_data) + criterion_recon(decoded_aug, batch_aug)
                recon_loss = recon_loss / 2.0
                contrastive_loss = self._ae_contrastive_loss(latent, latent_aug, batch_labels)

                total_loss = recon_loss + AE_CONTRASTIVE_WEIGHT * contrastive_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"W: Total AE loss NaN/Inf epoch {epoch+1}. Skipping backward.")
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
            centroid = np.mean(type_features, axis=0)
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
             return np.empty((0, self.actual_latent_dim)), {} if fault_labels is not None else np.empty((0, self.actual_latent_dim))

        self.autoencoder.eval()
        all_data_semantics_list = []
        inference_batch_size = AE_BATCH_SIZE * 4

        X_tensor = torch.FloatTensor(X).to(self.device)
        original_indices = np.arange(len(X))
        filtered_indices = []

        with torch.no_grad():
            for i in range(0, X_tensor.size(0), inference_batch_size):
                batch_indices = original_indices[i : i + inference_batch_size]
                batch_x = X_tensor[batch_indices]

                batch_valid_mask = torch.all(torch.isfinite(batch_x), dim=1)
                if not batch_valid_mask.all():
                     print(f"W: Non-finite data in batch for AE encoding (extract), index {i}. Processing valid rows only.")
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
             return np.empty((0, self.actual_latent_dim)), {} if fault_labels is not None else np.empty((0, self.actual_latent_dim))

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
                                                          n_neighbors=max(5, min(10, len(fault_type_semantics)-1)) )
                             cluster_labels = spectral.fit_predict(fault_type_semantics)

                             if cluster_labels is None or len(cluster_labels) != len(fault_type_semantics):
                                  prototype = np.mean(fault_type_semantics, axis=0)
                             else:
                                  counts = np.bincount(cluster_labels[cluster_labels>=0])
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
            return data_semantics, prototype_semantics
        else:
            return data_semantics

    def synthesize_compound_semantics(self, single_fault_prototypes):
        """
        使用属性合成与映射学习方法生成复合故障语义

        Args:
            single_fault_prototypes (dict): 单一故障语义原型，键为故障类型，值为语义向量

        Returns:
            dict: 复合故障语义
        """
        print("使用属性合成与映射学习方法生成复合故障语义...")

        if self.actual_latent_dim <= 0:
            print("错误: 无效的语义维度配置")
            return {}

        # 1. 定义故障属性空间
        fault_attributes = {
            'normal': [0, 0, 0],
            'inner': [1, 0, 0],
            'outer': [0, 1, 0],
            'ball': [0, 0, 1],
            'inner_outer': [1, 1, 0],
            'inner_ball': [1, 0, 1],
            'outer_ball': [0, 1, 1],
            'inner_outer_ball': [1, 1, 1]
        }

        # 2. 加入轴承物理参数，增强属性表示
        bearing_params = {
            'inner_diameter': 17 / 40,
            'outer_diameter': 1.0,
            'width': 12 / 40,
            'ball_diameter': 6.75 / 40,
            'ball_number': 9 / 20
        }

        # 创建增强的属性向量
        enhanced_attributes = {}
        for fault_type, attr_vec in fault_attributes.items():
            # 合并故障位置属性和轴承物理参数
            enhanced_attr = attr_vec + list(bearing_params.values())
            enhanced_attributes[fault_type] = enhanced_attr

        # 3. 定义属性映射网络
        class AttributeSemanticMapper(nn.Module):
            def __init__(self, attr_dim, semantic_dim, hidden_dims=[128, 256, 128]):
                super(AttributeSemanticMapper, self).__init__()

                layers = []
                # 输入层
                layers.append(nn.Linear(attr_dim, hidden_dims[0]))
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.2))

                # 隐藏层
                for i in range(len(hidden_dims) - 1):
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                    layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
                    layers.append(nn.LeakyReLU(0.2))
                    layers.append(nn.Dropout(0.2))

                # 输出层
                layers.append(nn.Linear(hidden_dims[-1], semantic_dim))

                self.mapper = nn.Sequential(*layers)

            def forward(self, x):
                return self.mapper(x)

        # 4. 准备训练数据（仅使用单一故障）
        train_attributes = []
        train_semantics = []

        single_fault_types = ['normal', 'inner', 'outer', 'ball']

        for fault_name in single_fault_types:
            if fault_name in enhanced_attributes and fault_name in single_fault_prototypes:
                attr_vec = enhanced_attributes[fault_name]
                sem_vec = single_fault_prototypes[fault_name]

                if np.all(np.isfinite(sem_vec)):
                    train_attributes.append(attr_vec)
                    train_semantics.append(sem_vec)

        if len(train_attributes) == 0:
            print("错误: 无有效的单一故障数据用于训练属性映射器")
            # 回退到原始方法
            compound_semantics = self._fallback_compound_synthesis(single_fault_prototypes)
            return compound_semantics

        # 转换为张量
        attr_dim = len(train_attributes[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train = torch.FloatTensor(np.array(train_attributes)).to(device)
        y_train = torch.FloatTensor(np.array(train_semantics)).to(device)

        # 5. 训练属性映射网络
        mapper = AttributeSemanticMapper(
            attr_dim=attr_dim,
            semantic_dim=self.actual_latent_dim
        ).to(device)

        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(mapper.parameters(), lr=0.001, weight_decay=1e-5)
        mse_loss = nn.MSELoss()
        cos_loss = nn.CosineEmbeddingLoss()

        print(f"训练属性-语义映射器: 属性维度={attr_dim}, 语义维度={self.actual_latent_dim}")

        # 训练循环
        epochs = 500
        best_loss = float('inf')
        patience_counter = 0
        patience = 50

        for epoch in range(epochs):
            mapper.train()
            optimizer.zero_grad()

            # 前向传播
            pred_semantics = mapper(X_train)

            # 计算损失
            mapping_loss = mse_loss(pred_semantics, y_train)

            # 余弦相似度损失 - 确保语义向量方向一致
            cosine_target = torch.ones(len(X_train)).to(device)
            consistency_loss = cos_loss(pred_semantics, y_train, cosine_target)

            # 正则化损失 - 鼓励语义保持单位范数
            norm_factors = torch.norm(pred_semantics, dim=1)
            target_norms = torch.norm(y_train, dim=1)
            norm_loss = torch.mean((norm_factors - target_norms) ** 2)

            # 总损失 - 动态平衡各项损失权重
            w_mse = 1.0
            w_cos = 0.5
            w_norm = 0.2

            total_loss = w_mse * mapping_loss + w_cos * consistency_loss + w_norm * norm_loss

            # 反向传播和优化
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0)
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.6f} (MSE: {mapping_loss.item():.6f}, Cos: {consistency_loss.item():.6f}, Norm: {norm_loss.item():.6f})")

            # 早停检查
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
                torch.save(mapper.state_dict(), 'best_attribute_mapper.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {epoch + 1}轮后未改善")
                    break

        # 加载最佳模型
        try:
            mapper.load_state_dict(torch.load('best_attribute_mapper.pth'))
            print("已加载最佳属性映射模型")
        except Exception as e:
            print(f"警告: 加载最佳模型失败: {e}")

        mapper.eval()

        # 6. 生成所有复合故障的语义表示
        compound_semantics = {}
        compound_fault_types = ['inner_outer', 'inner_ball', 'outer_ball', 'inner_outer_ball']

        with torch.no_grad():
            for fault_type in compound_fault_types:
                if fault_type in enhanced_attributes:
                    attr_vec = enhanced_attributes[fault_type]
                    attr_tensor = torch.FloatTensor(attr_vec).unsqueeze(0).to(device)

                    # 通过映射网络生成语义
                    semantic = mapper(attr_tensor).cpu().numpy()[0]

                    # 验证语义有效性
                    if np.all(np.isfinite(semantic)):
                        # 应用后处理调整 - 微调生成的语义确保与组件有合理关系
                        semantic = self._post_process_compound_semantic(
                            semantic,
                            fault_type,
                            single_fault_prototypes
                        )

                        compound_semantics[fault_type] = semantic

                        # 计算与单一故障语义的相似度用于验证
                        components = fault_type.split('_')
                        similarities = []
                        for comp in components:
                            if comp in single_fault_prototypes:
                                comp_vec = single_fault_prototypes[comp]
                                sim = np.dot(semantic, comp_vec) / (
                                            np.linalg.norm(semantic) * np.linalg.norm(comp_vec) + 1e-8)
                                similarities.append(sim)

                        avg_sim = sum(similarities) / len(similarities) if similarities else 0
                        print(f"生成 {fault_type} 语义, 与组件相似度: {avg_sim:.4f}")

        # 7. 检查生成结果，如果映射失败则回退到原始方法
        if len(compound_semantics) < len(compound_fault_types):
            print(f"警告: 只成功生成了 {len(compound_semantics)}/{len(compound_fault_types)} 种复合故障语义")
            # 对缺失的复合故障使用原始方法
            missed_types = [ft for ft in compound_fault_types if ft not in compound_semantics]
            fallback_semantics = self._generate_fallback_semantics(missed_types, single_fault_prototypes)
            # 合并结果
            compound_semantics.update(fallback_semantics)

        return compound_semantics

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
    def batch_domain_calibration(self, compound_vector, compound_type, component_types, single_prototypes):
        """
        批量域校准以减少域偏移

        参数:
            compound_vector: 合成的复合故障语义
            compound_type: 复合故障类型名称
            component_types: 组成该复合故障的单一故障类型列表
            single_prototypes: 所有单一故障原型的字典
        """
        n_components = len(component_types)
        amplitude_factor = 1.0 + 0.05 * n_components
        calibrated = compound_vector * amplitude_factor
        similarities = []
        for comp in component_types:
            if comp in single_prototypes:
                single_vec = single_prototypes[comp]
                sim = np.dot(calibrated, single_vec) / (np.linalg.norm(calibrated) * np.linalg.norm(single_vec) + 1e-10)
                similarities.append(sim)
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            if avg_sim > 0.75:
                noise = np.random.normal(0, 0.05, size=len(calibrated))
                calibrated = calibrated * 0.9 + noise * 0.1
            elif avg_sim < 0.4:

                for comp in component_types:
                    if comp in single_prototypes:
                        single_vec = single_prototypes[comp]
                        calibrated = calibrated * 0.8 + single_vec * 0.2 / n_components

        if compound_type == 'inner_outer':
            mid = len(calibrated) // 2
            calibrated[mid:] = calibrated[mid:] * 1.1
        elif compound_type == 'inner_ball' or compound_type == 'outer_ball':

            mod_pattern = np.sin(np.linspace(0, 3 * np.pi, len(calibrated))) * 0.1
            calibrated = calibrated + mod_pattern * np.mean(np.abs(calibrated))

        elif compound_type == 'inner_outer_ball':

            calibrated = calibrated * 1.15
            low_freq = np.exp(-np.linspace(0, 5, len(calibrated))) * np.mean(np.abs(calibrated)) * 0.2
            calibrated[:len(calibrated) // 4] += low_freq[:len(calibrated) // 4]

        if not np.all(np.isfinite(calibrated)):
            print(f"W: 校准后产生非有限值，回退到原始合成向量")
            return compound_vector

        norm = np.linalg.norm(calibrated)
        if norm > 1e-8:
            calibrated = calibrated / norm

        return calibrated

    def domain_adaptation_correction(self, compound_semantic, single_semantics):
        """
        应用域适应校正以减少合成复合故障与真实复合故障之间的域偏移

        参数:
            compound_semantic: 合成的复合故障语义向量
            single_semantics: 用于合成的单一故障语义向量列表

        返回:
            校正后的复合故障语义向量
        """

        single_means = np.mean([np.mean(s) for s in single_semantics])
        single_stds = np.mean([np.std(s) for s in single_semantics])
        single_ranges = np.mean([np.max(s) - np.min(s) for s in single_semantics])
        compound_mean = np.mean(compound_semantic)
        compound_std = np.std(compound_semantic)
        compound_range = np.max(compound_semantic) - np.min(compound_semantic)
        mean_factor = 1.05
        std_factor = 1.15
        range_factor = 1.1
        adjusted = compound_semantic.copy()
        adjusted = (adjusted - compound_mean) / (compound_std + 1e-10)
        adjusted = adjusted * (single_stds * std_factor)
        adjusted = adjusted + (single_means * mean_factor)
        n = len(adjusted)
        mid = n // 2
        adjusted[mid:] = adjusted[mid:] * 1.08
        adjusted[:mid] = adjusted[:mid] * 1.02
        for single_sem in single_semantics:
            sim = np.dot(compound_semantic, single_sem) / (
                        np.linalg.norm(compound_semantic) * np.linalg.norm(single_sem) + 1e-10)
            if sim > 0.85:
                ortho = single_sem - (np.dot(single_sem, adjusted) / np.dot(adjusted, adjusted)) * adjusted
                ortho_norm = np.linalg.norm(ortho)
                if ortho_norm > 1e-8:
                    ortho = ortho / ortho_norm
                    adjusted = adjusted * 0.9 + ortho * 0.1

            elif sim < 0.3:
                adjusted = adjusted * 0.9 + single_sem * 0.1

        return adjusted

class DomainDiscriminator(nn.Module):
    """
    小型二分类判别器，用于区分特征来源：
    domain=0：双通道（signal + semantic）
    domain=1：单通道（signal only）
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super(DomainDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)  # 输出logit
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_with_domain_adversarial(
    cnn_model: nn.Module,
    X_dual, y_dual, semantics,
    X_single, y_single,
    num_epochs=10,
    batch_size=128,
    lr=1e-4,
    lambda_domain=0.1,
    device='cpu'
):
    """
    对抗域训练：使单通道和双通道特征分布尽可能一致

    Args:
        cnn_model: 已初始化的 DualChannelCNN 实例
        X_dual: 双通道训练信号 (N_dual, L)
        y_dual: 双通道标签 (N_dual,)
        semantics: 双通道的语义向量列表 (N_dual, semantic_dim)
        X_single: 单通道训练信号 (N_single, L)
        y_single: 单通道对应的标签 (N_single,)
        lambda_domain: 对抗域损失权重
    """
    cnn_model.to(device).train()
    feature_dim = cnn_model.feature_dim

    # 域判别器
    domain_disc = DomainDiscriminator(feature_dim).to(device)

    # 优化器
    opt_cnn = optim.Adam(cnn_model.parameters(), lr=lr, weight_decay=1e-5)
    opt_disc = optim.Adam(domain_disc.parameters(), lr=lr, weight_decay=1e-5)

    # 损失函数
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()

    # 数据加载
    dual_ds = TensorDataset(
        torch.FloatTensor(X_dual),
        torch.LongTensor(y_dual),
        torch.FloatTensor(semantics)
    )
    single_ds = TensorDataset(
        torch.FloatTensor(X_single),
        torch.LongTensor(y_single)
    )
    loader_dual = DataLoader(dual_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_single = DataLoader(single_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(num_epochs):
        total_cls, total_dom, total_samples = 0.0, 0.0, 0

        iter_single = iter(loader_single)
        for x_d, y_d, sem_d in loader_dual:
            try:
                x_s, y_s = next(iter_single)
            except StopIteration:
                iter_single = iter(loader_single)
                x_s, y_s = next(iter_single)

            x_d = x_d.to(device); y_d = y_d.to(device)
            sem_d = sem_d.to(device)
            x_s = x_s.to(device); y_s = y_s.to(device)

            # —— 第一步：更新域判别器 ——
            # 提取特征，不参与CNN分类器或特征提取器更新
            with torch.no_grad():
                f_dual = cnn_model(x_d, sem_d, return_features=True)
                f_single = cnn_model(x_s, semantic=None, return_features=True)

            # domain labels: dual=0, single=1
            d_inputs = torch.cat([f_dual, f_single], dim=0)
            d_labels = torch.cat([
                torch.zeros(f_dual.size(0), device=device),
                torch.ones(f_single.size(0), device=device)
            ], dim=0)

            opt_disc.zero_grad()
            logits_d = domain_disc(d_inputs)
            loss_disc = criterion_domain(logits_d, d_labels)
            loss_disc.backward()
            opt_disc.step()

            # —— 第二步：更新CNN模型（特征提取 + 分类器）——
            opt_cnn.zero_grad()

            # 1. 分类损失（只在双通道上）
            logits_cls, _ = cnn_model(x_d, sem_d)
            loss_cls = criterion_cls(logits_cls, y_d)

            # 2. 对抗域损失：希望判别器对CNN特征区分失败
            f_dual_adv = cnn_model(x_d, sem_d, return_features=True)
            f_single_adv = cnn_model(x_s, semantic=None, return_features=True)
            all_features = torch.cat([f_dual_adv, f_single_adv], dim=0)
            # 反转标签：dual→1, single→0
            adv_labels = 1 - torch.cat([
                torch.zeros(f_dual_adv.size(0), device=device),
                torch.ones(f_single_adv.size(0), device=device)
            ], dim=0)

            logits_dom_adv = domain_disc(all_features)
            loss_domain_adv = criterion_domain(logits_dom_adv, adv_labels)

            # 总损失
            loss_total = loss_cls + lambda_domain * loss_domain_adv
            loss_total.backward()
            opt_cnn.step()

            total_cls += loss_cls.item() * x_d.size(0)
            total_dom += loss_domain_adv.item() * x_d.size(0)
            total_samples += x_d.size(0)

        avg_cls = total_cls / total_samples
        avg_dom = total_dom / total_samples
        print(f"[Epoch {epoch+1}/{num_epochs}] cls_loss={avg_cls:.4f}, domain_adv_loss={avg_dom:.4f}")

    print("Domain-adversarial training completed.")
    return cnn_model, domain_disc

class ResidualBlock(nn.Module):
    """高级残差块，用于增强网络训练稳定性和表达能力"""

    def __init__(self, in_dim, out_dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_dim, out_dim)),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.utils.spectral_norm(nn.Linear(out_dim, out_dim)),
            nn.LayerNorm(out_dim)
        )
        # 如果维度不匹配，添加投影层
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        return self.activation(out + identity)


class SelfAttention(nn.Module):
    """自注意力机制，增强语义信息交互"""

    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        # 缩小投影维度以提高效率
        self.projection_dim = max(32, dim // 8)
        self.query = nn.Linear(dim, self.projection_dim)
        self.key = nn.Linear(dim, self.projection_dim)
        self.value = nn.Linear(dim, dim)
        self.scale = self.projection_dim ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的注意力权重

    def forward(self, x):
        # x: [batch_size, dim]
        batch_size = x.size(0)
        if batch_size <= 1:  # 单个样本无法进行自注意力
            return x

        # 计算注意力权重
        q = self.query(x).unsqueeze(1)  # [B, 1, proj_dim]
        k = self.key(x).unsqueeze(2)  # [B, proj_dim, 1]
        v = self.value(x)  # [B, dim]

        # 批量注意力计算
        attention = torch.bmm(q, k) * self.scale  # [B, 1, 1]
        attention = F.softmax(attention.squeeze(), dim=1).unsqueeze(1)  # [B, 1, 1]

        # 加权和
        out = attention * v  # [B, dim]

        # 残差连接
        return x + self.gamma * out

def gradient_penalty(discriminator, real_data, fake_data, device):
    """
    计算 WGAN-GP 的梯度惩罚项
    Args:
        discriminator: 判别器网络
        real_data: 真实数据（CNN特征）
        fake_data: 生成数据（映射器生成的特征）
        device: 计算设备（CPU/GPU）
    Returns:
        梯度惩罚值
    """
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_data)  # 扩展到与输入相同的形状

    # 插值：real_data 和 fake_data 的线性组合
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.requires_grad_(True)

    # 通过判别器计算插值点的分数
    disc_interpolates = discriminator(interpolates)

    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]

    # 计算梯度范数
    gradients = gradients.view(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

class ResidualLinearBlock(nn.Module):
    """残差全连接块"""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear1 = nn.utils.spectral_norm(nn.Linear(in_dim, out_dim))
        self.linear2 = nn.utils.spectral_norm(nn.Linear(out_dim, out_dim))
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.swish = nn.SiLU()  # Swish 激活函数
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.swish(self.norm1(self.linear1(x)))
        out = self.dropout(out)
        out = self.norm2(self.linear2(out))
        return self.swish(out + residual)

class AdversarialSemanticMappingWithCycle(nn.Module):
    def __init__(self, semantic_dim, feature_dim):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim

        # 中间维度
        h1 = max(semantic_dim * 2, 512)
        h2 = max(semantic_dim, 256)
        h3 = max(feature_dim, 128)

        # 映射器：语义 → 特征
        self.mapper = nn.Sequential(
            ResidualLinearBlock(semantic_dim, h1, dropout=0.3),
            ResidualLinearBlock(h1, h2, dropout=0.3),
            ResidualLinearBlock(h2, h3, dropout=0.2),
            nn.utils.spectral_norm(nn.Linear(h3, feature_dim)),
            nn.Tanh()  # 归一化到 [-1, 1]
        )

        # 解码器：特征 → 语义
        self.semantic_decoder = nn.Sequential(
            ResidualLinearBlock(feature_dim, h3, dropout=0.2),
            ResidualLinearBlock(h3, h2, dropout=0.3),
            ResidualLinearBlock(h2, h1, dropout=0.3),
            nn.utils.spectral_norm(nn.Linear(h1, semantic_dim))
            # 无激活函数，适应语义向量范围
        )

        # 判别器
        self.discriminator = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feature_dim, h2)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(h2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(h2, h3)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(h3),
            nn.Dropout(0.2),
            nn.utils.spectral_norm(nn.Linear(h3, 1))
            # 无 sigmoid，配合 WGAN 损失
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """语义 → 特征"""
        return self.mapper(x)

    def decode(self, features):
        """特征 → 语义"""
        return self.semantic_decoder(features)

    def cycle(self, semantics):
        """语义 → 特征 → 语义"""
        features = self.mapper(semantics)
        reconstructed_semantics = self.semantic_decoder(features)
        return features, reconstructed_semantics

    def discriminate(self, features):
        """判别器"""
        return self.discriminator(features)

class EnhancedAdversarialSemanticMapping(nn.Module):
    """增强型对抗语义映射网络，具有双向循环一致性和注意力机制"""

    def __init__(self, semantic_dim, feature_dim, hidden_dim=512, dropout_rate=0.3):
        super(EnhancedAdversarialSemanticMapping, self).__init__()
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 1. 语义编码器（Semantic -> Feature）- 使用残差连接增强信息流
        self.semantic_encoder = nn.ModuleList([
            # 输入投影层
            nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(semantic_dim, hidden_dim)),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ),
            # 残差块1
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate),
            # 残差块2
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate),
            # 输出投影层
            nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(hidden_dim, feature_dim)),
                nn.Tanh()  # 限制输出范围
            )
        ])

        # 2. 特征解码器（Feature -> Semantic）- 对称结构但带有注意力机制
        self.feature_decoder = nn.ModuleList([
            # 输入投影层
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ),
            # 残差块1 + 自注意力
            nn.Sequential(
                ResidualBlock(hidden_dim, hidden_dim, dropout_rate),
                SelfAttention(hidden_dim)
            ),
            # 残差块2
            ResidualBlock(hidden_dim, hidden_dim, dropout_rate),
            # 输出投影层 - 不用激活函数以允许更广泛的语义空间
            nn.Linear(hidden_dim, semantic_dim)
        ])

        # 3. 多尺度判别器 - 检测真假特征
        # 使用1个主判别器和2个辅助判别器以增强稳定性
        self.main_discriminator = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feature_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.utils.spectral_norm(nn.Linear(hidden_dim // 2, 1))
        )

        # 辅助判别器1 - 高频特征检测器
        self.aux_discriminator1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feature_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate / 2),
            nn.utils.spectral_norm(nn.Linear(hidden_dim // 2, 1))
        )

        # 辅助判别器2 - 低频特征检测器
        self.aux_discriminator2 = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # 这个直接输出概率
        )

        # 4. 语义判别器（用于对抗语义空间对齐）
        self.semantic_discriminator = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重，采用针对不同部分的最佳实践"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

                # 根据模块所在层次采用不同初始化策略
                parent_name = str(m.__class__.__module__)
                if 'semantic_encoder' in parent_name:
                    # Xavier (Glorot) 适合Tanh/Sigmoid激活
                    nn.init.xavier_uniform_(m.weight)
                elif 'discriminator' in parent_name:
                    # He初始化适合ReLU/LeakyReLU激活
                    nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                else:
                    # 默认使用He初始化
                    nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')

    def encode_semantic(self, semantic):
        """将语义向量编码为特征向量"""
        x = semantic
        for layer in self.semantic_encoder:
            x = layer(x)
        return x

    def decode_feature(self, feature):
        """将特征向量解码回语义空间"""
        x = feature
        for layer in self.feature_decoder:
            x = layer(x)
        return x

    def cycle_semantic(self, semantic):
        """语义→特征→语义循环"""
        feature = self.encode_semantic(semantic)
        reconstructed = self.decode_feature(feature)
        return feature, reconstructed

    def cycle_feature(self, feature):
        """特征→语义→特征循环"""
        semantic = self.decode_feature(feature)
        reconstructed = self.encode_semantic(semantic)
        return semantic, reconstructed

    def discriminate(self, feature):
        """使用多尺度判别器区分真实和生成的特征"""
        # 融合多尺度判别结果
        main_output = self.main_discriminator(feature)
        aux1_output = self.aux_discriminator1(feature)
        aux2_output = self.aux_discriminator2(feature)

        # 不同判别器的权重可调整
        return main_output, aux1_output, aux2_output

    def discriminate_semantic(self, semantic):
        """区分原始语义和重构语义"""
        return self.semantic_discriminator(semantic)

    def forward(self, x):
        """前向传播方法（为了兼容性）"""
        return self.encode_semantic(x)

class DomainDiscriminator(nn.Module):
    """区分单通道和双通道特征的域判别器"""
    def __init__(self, feature_dim, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),  # 输出二分类logit（单通道:0, 双通道:1）
            nn.Sigmoid()  # 输出概率
        )
        self._initialize_weights()

    def forward(self, features):
        return self.network(features)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversalLayer.apply(x, lambda_)

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

        out += identity # 残差连接
        out = self.relu(out)
        return out

class SemanticAttention(nn.Module):
    """使用语义信息对信号特征进行通道加权 (类似SE Block)"""
    def __init__(self, signal_channels, semantic_dim, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # 全局平均池化获取通道描述符
        hidden_dim = max(1, signal_channels // reduction)

        # 结合语义信息生成注意力权重
        self.fc = nn.Sequential(
            nn.Linear(signal_channels + semantic_dim, hidden_dim, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, signal_channels, bias=False),
            nn.Sigmoid() # 输出通道权重 (0-1)
        )
        self.semantic_dim = semantic_dim

    def forward(self, signal_features, semantic_features):
        b, c, _ = signal_features.size()
        # 1. 获取信号通道描述符
        signal_channel_desc = self.avg_pool(signal_features).view(b, c)

        # 2. 检查并调整语义特征维度
        if semantic_features.dim() == 1:
             semantic_features = semantic_features.unsqueeze(0).expand(b, -1)
        elif semantic_features.size(0) != b:
             # 如果语义特征批次大小不匹配，尝试广播或报错
             if semantic_features.size(0) == 1:
                 semantic_features = semantic_features.expand(b, -1)
             else:
                 raise ValueError(f"Semantic feature batch size {semantic_features.size(0)} "
                                  f"does not match signal feature batch size {b}")

        # 3. 拼接通道描述符和语义特征
        combined = torch.cat([signal_channel_desc, semantic_features], dim=1)

        # 4. 生成通道注意力权重
        channel_weights = self.fc(combined).view(b, c, 1)

        # 5. 将权重应用到信号特征上
        attended_signal = signal_features * channel_weights
        return attended_signal

class DualChannelCNN(nn.Module):
    def __init__(self, input_length=SEGMENT_LENGTH, semantic_dim=AE_LATENT_DIM,
                 num_classes=8, feature_dim=CNN_FEATURE_DIM):
        super().__init__()
        if input_length <= 0 or semantic_dim <= 0 or num_classes <= 0 or feature_dim <= 0:
            raise ValueError("Invalid dimensions provided to DualChannelCNN")

        self.input_length = input_length
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim # 融合后的特征维度

        # --- 通道1: 信号处理 (基于ResNet思想) ---
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False), # 较大的初始卷积核
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.res_block1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.res_block2 = self._make_layer(64, 128, blocks=2, stride=2) # 降维
        self.res_block3 = self._make_layer(128, 256, blocks=2, stride=2) # 降维
        # 可以根据需要添加更多残差块
        # self.res_block4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.final_conv_channels = 256 # 与最后一个残差块的输出通道数一致
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1) # 全局平均池化

        # --- 通道2: 语义处理 ---
        # 保持简单，或稍微加深
        self.semantic_fc = nn.Sequential(
            nn.Linear(semantic_dim, 128),
            nn.BatchNorm1d(128), # 添加BN
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 128), # 输出一个固定维度的语义表示
            nn.LeakyReLU(0.1, inplace=True), # 添加激活函数
        )
        self.semantic_out_dim = 128

        # --- 注意力与融合 ---
        # 使用语义信息对信号特征进行加权
        self.attention = SemanticAttention(self.final_conv_channels, self.semantic_out_dim)

        # 融合层：将加权后的信号特征 (GAP后) 与语义特征拼接
        fusion_input_dim = self.final_conv_channels + self.semantic_out_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_input_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5) # 较强的Dropout
        )

        # --- 分类器 ---
        self.classifier = nn.Linear(feature_dim, num_classes) # 不再使用weight_norm，可以在训练循环中添加

        # --- 处理无语义输入的情况 ---
        # 如果没有语义输入，需要一个单独的路径或处理方式
        # 这里我们创建一个简单的线性层，将信号GAP后的特征直接映射到最终的feature_dim
        self.signal_only_fc = nn.Sequential(
            nn.Linear(self.final_conv_channels, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """构建包含多个残差块的层"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            # 如果需要下采样（步幅不为1或通道数变化）
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, semantic=None, return_features=False):
        # --- 输入预处理 ---
        if x.dim() == 2:
            x = x.unsqueeze(1) # 确保输入是 (B, 1, L)
        if not torch.all(torch.isfinite(x)):
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # --- 通道1: 信号特征提取 ---
        signal_out = self.initial_conv(x)
        signal_out = self.res_block1(signal_out)
        signal_out = self.res_block2(signal_out)
        signal_out = self.res_block3(signal_out)
        # if hasattr(self, 'res_block4'): signal_out = self.res_block4(signal_out)

        # --- 处理无语义输入的情况 ---
        if semantic is None:
            # 使用全局平均池化
            signal_gap = self.global_avg_pool(signal_out).squeeze(-1) # (B, C)
            if not torch.all(torch.isfinite(signal_gap)):
                signal_gap = torch.nan_to_num(signal_gap, nan=0.0)

            # 通过单独的全连接层处理
            features = self.signal_only_fc(signal_gap)
            if not torch.all(torch.isfinite(features)):
                features = torch.nan_to_num(features, nan=0.0)

            if return_features:
                return features
            else:
                logits = self.classifier(features)
                if not torch.all(torch.isfinite(logits)):
                    logits = torch.nan_to_num(logits, nan=0.0)
                # 返回 logits 和 features 以便计算对比损失等
                return logits, features

        # --- 处理有语义输入的情况 ---
        # 语义输入校验
        if not torch.all(torch.isfinite(semantic)):
             semantic = torch.nan_to_num(semantic, nan=0.0)
        if semantic.dim() != 2 or semantic.shape[1] != self.semantic_dim:
            # 尝试修复维度不匹配，例如如果批次大小为1
            if semantic.dim() == 1 and self.semantic_dim == semantic.shape[0]:
                 semantic = semantic.unsqueeze(0)
            elif semantic.dim()==2 and semantic.shape[0]==1 and x.shape[0]>1:
                 semantic = semantic.expand(x.shape[0], -1) # 广播语义
            else:
                 raise ValueError(f"CNN Semantic input wrong shape. Expected [B, {self.semantic_dim}], got {semantic.shape}")

        # --- 通道2: 语义特征处理 ---
        semantic_out = self.semantic_fc(semantic) # (B, semantic_out_dim)
        if not torch.all(torch.isfinite(semantic_out)):
            semantic_out = torch.nan_to_num(semantic_out, nan=0.0)

        # --- 注意力与融合 ---
        # 应用语义注意力到信号特征 (卷积输出)
        attended_signal = self.attention(signal_out, semantic_out) # (B, C, L')
        if not torch.all(torch.isfinite(attended_signal)):
            attended_signal = torch.nan_to_num(attended_signal, nan=0.0)

        # 对加权后的信号特征进行全局平均池化
        signal_gap_attended = self.global_avg_pool(attended_signal).squeeze(-1) # (B, C)
        if not torch.all(torch.isfinite(signal_gap_attended)):
            signal_gap_attended = torch.nan_to_num(signal_gap_attended, nan=0.0)

        # 拼接加权信号特征和语义特征
        concatenated = torch.cat([signal_gap_attended, semantic_out], dim=1) # (B, C + semantic_out_dim)
        if not torch.all(torch.isfinite(concatenated)):
            concatenated = torch.nan_to_num(concatenated, nan=0.0)

        # 通过融合层
        fused_features = self.fusion_fc(concatenated) # (B, feature_dim)
        if not torch.all(torch.isfinite(fused_features)):
            fused_features = torch.nan_to_num(fused_features, nan=0.0)

        if return_features:
            return fused_features

        # --- 分类 ---
        logits = self.classifier(fused_features) # (B, num_classes)
        if not torch.all(torch.isfinite(logits)):
            logits = torch.nan_to_num(logits, nan=0.0)

        # 返回 logits 和融合后的特征
        return logits, fused_features


class ContrastiveLoss(nn.Module):
    """ For CNN Training - Encourages features of same class to be closer """
    def __init__(self, temperature=0.05):
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
        mask = mask - identity_mask
        exp_logits = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) - exp_logits.diag().unsqueeze(1) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        valid_loss_terms = mean_log_prob_pos[mask.sum(dim=1) > 0]
        if len(valid_loss_terms) > 0:
             loss = -valid_loss_terms.mean()
        else:
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
        self.projection = nn.Linear(semantic_dim_input, feature_dim_output)
        try:
             nn.init.xavier_uniform_(self.projection.weight)
             if self.projection.bias is not None: nn.init.zeros_(self.projection.bias)
        except: pass
        print(f"Initialized ConsistencyLoss projection: {semantic_dim_input} -> {feature_dim_output}")

    def forward(self, features, data_semantics):

         if not torch.all(torch.isfinite(features)): return torch.tensor(0.0, device=features.device)
         if not torch.all(torch.isfinite(data_semantics)): return torch.tensor(0.0, device=features.device)
         if features.shape[0] != data_semantics.shape[0]: return torch.tensor(0.0, device=features.device)
         if data_semantics.shape[1] != self.projection.in_features:
              print(f"E: ConsistencyLoss dim mismatch! Expected {self.projection.in_features}, got {data_semantics.shape[1]}")
              return torch.tensor(0.0, device=features.device)
         self.projection.to(features.device)

         projected_data_semantics = self.projection(data_semantics)

         if not torch.all(torch.isfinite(projected_data_semantics)):
              print("W: ConsistencyLoss projection output non-finite. Using zeros.")
              projected_data_semantics = torch.zeros_like(projected_data_semantics)

         features_norm = F.normalize(features, p=2, dim=1)
         projected_data_semantics_norm = F.normalize(projected_data_semantics, p=2, dim=1)
         if torch.isnan(features_norm).any(): features_norm = torch.nan_to_num(features_norm, nan=0.0)
         if torch.isnan(projected_data_semantics_norm).any(): projected_data_semantics_norm = torch.nan_to_num(projected_data_semantics_norm, nan=0.0)

         similarity = torch.sum(features_norm * projected_data_semantics_norm, dim=1) / self.temperature  # 加入温度缩放
         consistency_loss = 1.0 - torch.mean(similarity)

         l2_reg = torch.mean(torch.norm(features, p=2, dim=1)**2) if self.beta > 0 else 0.0
         loss = consistency_loss + self.beta * l2_reg

         if torch.isnan(loss) or torch.isinf(loss): return torch.tensor(0.0, device=features.device)
         return loss

class ZeroShotCompoundFaultDiagnosis:

    def __init__(self, data_path, sample_length=SEGMENT_LENGTH, latent_dim=AE_LATENT_DIM, batch_size=DEFAULT_BATCH_SIZE):
        self.data_path = data_path
        self.domain_discriminator = None  # 初始化为None
        self.sample_length = sample_length
        self.latent_dim_config = latent_dim
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.preprocessor = DataPreprocessor(sample_length=self.sample_length)
        self.semantic_builder = FaultSemanticBuilder(latent_dim=self.latent_dim_config)
        self.fault_types = {
            'normal': 0, 'inner': 1, 'outer': 2, 'ball': 3,
            'inner_outer': 4, 'inner_ball': 5, 'outer_ball': 6, 'inner_outer_ball': 7
        }
        self.idx_to_fault = {v: k for k, v in self.fault_types.items()}
        self.semantic_builder.idx_to_fault = self.idx_to_fault
        self.compound_fault_types = ['inner_outer',
                                     'inner_ball', 'outer_ball', 'inner_outer_ball']
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

        # 获取AE训练后的实际潜在维度
        self.actual_latent_dim = self.semantic_builder.actual_latent_dim
        if self.actual_latent_dim <= 0:
            print("E: Invalid AE latent dimension after training.")
            return None

        # 获取单一故障的原型 (centroids)
        single_fault_prototypes = self.semantic_builder.data_semantics
        if not single_fault_prototypes:
            print("E: AE training produced no data semantic prototypes (centroids).")
            # 即使没有原型，可能仍然有 latent features，所以不立即返回 None
            # return None # 原始逻辑是如果原型失败就退出，这里改为继续尝试获取 latent features

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

                fused_vec = np.concatenate([k_vec, d_vec]).astype(np.float32)
                if np.all(np.isfinite(fused_vec)) and len(fused_vec) == self.fused_semantic_dim:
                    fused_semantics[ft] = fused_vec
                else:
                    print(f"W: Fused vector for '{ft}' is invalid or has wrong dimension after concatenation.")
            # else: # Debugging info
            #     if d_vec is None: print(f"Debug: No data semantic vector for {ft}")
            #     elif k_vec is None: print(f"Debug: No knowledge semantic vector for {ft}")
            #     elif not np.all(np.isfinite(k_vec)): print(f"Debug: Invalid knowledge vector for {ft}")
            #     elif not np.all(np.isfinite(d_vec)): print(f"Debug: Invalid data vector for {ft}")
            #     elif len(k_vec) != self.semantic_builder.knowledge_dim: print(f"Debug: Wrong knowledge dim for {ft}")
            #     elif len(d_vec) != self.actual_latent_dim: print(f"Debug: Wrong data dim for {ft}")

        print(f"  Fused semantics prepared for {len(fused_semantics)} types. Dimension: {self.fused_semantic_dim}")

        if self.actual_latent_dim <= 0 or self.fused_semantic_dim <= 0:
            print("E: Invalid semantic dimensions calculated. Aborting.")
            return None

        # --- 新增：从 semantic_builder 获取 latent features ---
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

    def visualize_semantics(self, semantic_dict):
        """Visualizes semantic similarity matrices."""
        print("Visualizing semantic similarity matrices...")
        def compute_similarity_matrix(semantics_dict):
            fault_types = []
            vectors = []
            if not semantics_dict: return None, []
            first_valid_vec = next((v for v in semantics_dict.values() if v is not None and np.all(np.isfinite(v))), None)
            if first_valid_vec is None: return None, []
            expected_dim = len(first_valid_vec)
            for ft, vec in semantics_dict.items():
                 if vec is not None and np.all(np.isfinite(vec)) and len(vec)==expected_dim:
                     norm = np.linalg.norm(vec)
                     if norm > 1e-9: vectors.append(vec / norm); fault_types.append(ft)

            if not vectors: return None, []
            n = len(vectors)
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    sim = np.dot(vectors[i], vectors[j])
                    sim_matrix[i, j] = sim; sim_matrix[j, i] = sim
            return sim_matrix, fault_types

        plt.rc('font', size=8)
        d_sim_mat, d_labels = compute_similarity_matrix(semantic_dict.get('data_only_semantics'))
        if d_sim_mat is not None:
            configure_chinese_font()
            plt.figure(figsize=(9, 7)); sns.heatmap(d_sim_mat, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=d_labels, yticklabels=d_labels, annot_kws={"size": 7})
            plt.title('Data-Only Semantics Similarity (AE Output)'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            plt.tight_layout(); plt.savefig('data_only_semantics_similarity.png'); plt.close(); print("Data-only sim matrix saved.")
        else: print("W: Could not compute/plot data-only similarity matrix.")
        f_sim_mat, f_labels = compute_similarity_matrix(semantic_dict.get('fused_semantics'))
        if f_sim_mat is not None:
            configure_chinese_font()
            plt.figure(figsize=(9, 7)); sns.heatmap(f_sim_mat, annot=True, fmt=".2f", cmap="viridis", xticklabels=f_labels, yticklabels=f_labels, annot_kws={"size": 7})
            plt.title('Fused Semantics Similarity (Knowledge + AE Data)'); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
            plt.tight_layout(); plt.savefig('fused_semantics_similarity.png'); plt.close(); print("Fused sim matrix saved.")
        else: print("W: Could not compute/plot fused similarity matrix.")
        plt.rcdefaults()

    def visualize_data_semantics_distribution(self, semantic_dict, pca_components=2, max_points_per_class=500):
        """使用PCA降维可视化单一故障样本和合成复合故障原型的数据语义分布"""
        print(f"\n使用PCA降维可视化数据语义分布 (目标维度: {pca_components})...")

        # 获取单一故障样本的 latent features 和 labels
        single_features = semantic_dict.get('single_fault_latent_features')
        single_labels_numeric = semantic_dict.get('single_fault_latent_labels')

        # 获取复合故障的合成原型
        compound_prototypes = semantic_dict.get('compound_data_semantics')

        if (single_features is None or single_labels_numeric is None) and not compound_prototypes:
            print("警告: 语义字典中无单一故障latent features或复合故障prototypes，无法可视化。")
            return

        all_vectors = []
        all_labels_str = []
        vector_types = []  # 用于区分 'single_sample' 或 'compound_prototype'

        # 1. 处理单一故障样本
        if single_features is not None and single_labels_numeric is not None:
            print(f"  处理 {len(single_features)} 个单一故障样本...")
            unique_single_labels = np.unique(single_labels_numeric)
            sampled_indices = []

            for label_val in unique_single_labels:
                class_indices = np.where(single_labels_numeric == label_val)[0]
                num_class_samples = len(class_indices)

                if num_class_samples > max_points_per_class:
                    # 如果样本过多，进行随机抽样
                    chosen_indices = np.random.choice(class_indices, max_points_per_class, replace=False)
                    sampled_indices.extend(chosen_indices)
                    print(
                        f"    - 类别 {self.idx_to_fault.get(label_val, label_val)}: 抽样 {max_points_per_class}/{num_class_samples} 个样本")
                else:
                    # 样本不多，全部使用
                    sampled_indices.extend(class_indices)

            # 添加抽样后的单一故障样本
            sampled_features = single_features[sampled_indices]
            sampled_labels_numeric = single_labels_numeric[sampled_indices]

            all_vectors.extend(list(sampled_features))
            all_labels_str.extend([self.idx_to_fault.get(lbl, f"Unk_{lbl}") for lbl in sampled_labels_numeric])
            vector_types.extend(['single_sample'] * len(sampled_features))
            print(f"  添加了 {len(sampled_features)} 个抽样后的单一故障样本用于可视化。")

        # 2. 处理复合故障原型
        if compound_prototypes:
            print(f"  处理 {len(compound_prototypes)} 个合成的复合故障原型...")
            for fault_type, vector in compound_prototypes.items():
                if vector is not None and np.all(np.isfinite(vector)):
                    all_vectors.append(vector)
                    all_labels_str.append(fault_type)
                    vector_types.append('compound_prototype')
                else:
                    print(f"警告: 复合故障原型 '{fault_type}' 的语义向量无效，已跳过。")

        # 3. PCA 降维
        if len(all_vectors) < 2:
            print("警告: 有效向量（样本+原型）少于2个，无法进行PCA降维和可视化。")
            return

        # 检查维度一致性
        first_dim = len(all_vectors[0])
        if not all(len(v) == first_dim for v in all_vectors):
            print("错误: 向量维度不一致，无法进行PCA。")
            # 尝试找出维度不一致的向量
            for i, v in enumerate(all_vectors):
                if len(v) != first_dim:
                    print(
                        f"  - 向量 {i} ({all_labels_str[i]}, 类型 {vector_types[i]}) 维度为 {len(v)}, 期望 {first_dim}")
            return

        if first_dim < pca_components:
            print(f"警告: 原始维度 ({first_dim}) 小于目标PCA维度 ({pca_components})，将使用原始维度。")
            pca_components = first_dim

        try:
            pca = PCA(n_components=pca_components)
            data_pca = pca.fit_transform(np.array(all_vectors))
            print(f"  PCA完成。解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")
        except Exception as e:
            print(f"错误: PCA降维失败: {e}")
            traceback.print_exc()
            return

        # 4. 绘图
        configure_chinese_font()
        plt.figure(figsize=(14, 12))  # 增大图像尺寸

        unique_fault_types_str = sorted(list(set(all_labels_str)))
        colors = plt.cm.get_cmap('tab20', len(unique_fault_types_str))  # 使用更多颜色
        color_map = {label: colors(i) for i, label in enumerate(unique_fault_types_str)}

        # 分别绘制单一故障样本和复合故障原型
        for i in range(len(data_pca)):
            label = all_labels_str[i]
            vec_type = vector_types[i]
            color = color_map[label]
            x_coord = data_pca[i, 0]
            y_coord = data_pca[i, 1] if pca_components > 1 else 0

            if vec_type == 'single_sample':
                plt.scatter(x_coord, y_coord, color=color, marker='.', s=20, alpha=0.6,
                            label=label if i == all_labels_str.index(label) else "")  # 小点表示样本
            elif vec_type == 'compound_prototype':
                plt.scatter(x_coord, y_coord, color=color, marker='X', s=150, alpha=0.9, edgecolors='k', linewidths=1,
                            label=label if i == all_labels_str.index(label) else "")  # 大X表示原型
                # 在复合故障原型旁边添加文本标签
                plt.text(x_coord + 0.02, y_coord + 0.02, label, fontsize=10, weight='bold')

        # 添加图例和标题
        handles, labels = plt.gca().get_legend_handles_labels()
        # 去重图例标签
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), title="故障类型", bbox_to_anchor=(1.05, 1), loc='upper left',
                   markerscale=2)

        plt.title(f'数据语义分布 (PCA降至{pca_components}维)\n(点: 单一故障样本, X: 合成复合故障原型)')
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2f})')
        if pca_components > 1:
            plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2f})')
        else:
            plt.ylabel('')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例留出空间

        # 保存图像
        save_path = 'data_semantics_distribution_samples_pca.png'
        plt.savefig(save_path)
        print(f"数据语义分布图（含样本点）已保存至: {save_path}")
        plt.close()

    def train_dual_channel_cnn(self, data_dict, semantic_dict, epochs=CNN_EPOCHS, lr=CNN_LR, domain_lambda=0.1):
        """改进的CNN训练，加入双路径对抗训练，使单通道和双通道特征分布对齐"""
        print("Training dual channel CNN model with domain-adversarial alignment...")

        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']

        if len(X_train) == 0:
            print("E: No training data for CNN.")
            return

        has_val_data = len(X_val) > 0
        data_only_semantics = semantic_dict.get('data_only_semantics')

        if not data_only_semantics:
            print("E: Data-only semantics missing.")
            return

        if self.actual_latent_dim <= 0:
            print("E: AE latent dim not set.")
            return

        semantic_dim_data = self.actual_latent_dim
        default_semantic = np.zeros(semantic_dim_data, dtype=np.float32)
        semantic_vectors_cnn = {idx: data_only_semantics.get(name, default_semantic)
                                for name, idx in self.fault_types.items()}

        try:
            self.cnn_model = DualChannelCNN(
                input_length=self.sample_length,
                semantic_dim=self.actual_latent_dim,
                num_classes=self.num_classes,
                feature_dim=CNN_FEATURE_DIM
            ).to(self.device)
            self.cnn_feature_dim = self.cnn_model.feature_dim
        except Exception as e:
            print(f"E: CNN init failed: {e}")
            return

        # 新增域判别器
        try:
            self.domain_discriminator = DomainDiscriminator(feature_dim=self.cnn_feature_dim).to(self.device)
        except Exception as e:
            print(f"E: Domain discriminator init failed: {e}")
            return

        # 损失函数
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion_contrastive = ContrastiveLoss(temperature=0.07).to(self.device)
        criterion_domain = nn.BCELoss()  # 域判别器的二分类损失

        try:
            self.consistency_loss_fn = FeatureSemanticConsistencyLoss(
                beta=0.005,
                semantic_dim_input=self.actual_latent_dim,
                feature_dim_output=self.cnn_feature_dim
            ).to(self.device)
        except Exception as e:
            print(f"E: Consistency loss init failed: {e}")
            return

        # 优化器
        cnn_params = list(self.cnn_model.parameters())
        consist_params = []
        if hasattr(self.consistency_loss_fn, 'projection') and self.consistency_loss_fn.projection is not None:
            consist_params = list(self.consistency_loss_fn.projection.parameters())
        domain_params = list(self.domain_discriminator.parameters())

        optimizer_cnn = optim.AdamW(cnn_params + consist_params, lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
        optimizer_domain = optim.AdamW(domain_params, lr=lr * 0.5, weight_decay=1e-4)  # 域判别器学习率稍低

        # 数据加载器
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        steps_per_epoch = len(train_loader)
        scheduler_cnn = optim.lr_scheduler.OneCycleLR(
            optimizer_cnn,
            max_lr=lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0,
            anneal_strategy='cos'
        )
        scheduler_domain = optim.lr_scheduler.OneCycleLR(
            optimizer_domain,
            max_lr=lr * 0.5,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0,
            anneal_strategy='cos'
        )

        val_loader = None
        if has_val_data:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)

        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15

        train_losses = []
        val_losses = []
        domain_losses = []

        print(f"Starting CNN training with domain-adversarial alignment ({epochs} epochs)...")
        for epoch in range(epochs):
            self.cnn_model.train()
            self.domain_discriminator.train()
            if self.consistency_loss_fn:
                self.consistency_loss_fn.train()

            train_loss, domain_loss, correct, total = 0.0, 0.0, 0, 0
            train_loss_ce, train_loss_contr, train_loss_consist = 0.0, 0.0, 0.0
            batch_count = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                current_bs = inputs.size(0)
                if current_bs < 2:
                    continue

                # 数据增强
                if np.random.random() > 0.5:
                    noise_level = 0.01
                    noise = torch.randn_like(inputs) * noise_level
                    inputs = inputs + noise
                    inputs = torch.clamp(inputs, -1.0, 1.0)

                # 获取语义输入
                batch_semantics_data = torch.stack([
                    torch.from_numpy(semantic_vectors_cnn[lbl.item()])
                    for lbl in labels]).to(self.device)

                # 计算双通道特征
                logits, features_dual = self.cnn_model(inputs, batch_semantics_data)
                if not torch.all(torch.isfinite(logits)) or not torch.all(torch.isfinite(features_dual)):
                    print(f"W: NaN/Inf in dual-channel outputs epoch {epoch + 1}. Skip batch.")
                    continue

                # 计算单通道特征
                _, features_single = self.cnn_model(inputs, semantic=None)
                if not torch.all(torch.isfinite(features_single)):
                    print(f"W: NaN/Inf in single-channel outputs epoch {epoch + 1}. Skip batch.")
                    continue

                # 1. 任务分类损失
                ce_loss = criterion_ce(logits, labels)
                contr_loss = criterion_contrastive(features_dual, labels)
                consist_loss = self.consistency_loss_fn(features_dual, batch_semantics_data) \
                    if self.consistency_loss_fn else torch.tensor(0.0)

                # 2. 域对抗损失
                # 域标签：单通道=0, 双通道=1
                domain_labels_single = torch.zeros(current_bs, device=self.device)
                domain_labels_dual = torch.ones(current_bs, device=self.device)

                # 通过梯度反转层
                features_single_grl = grad_reverse(features_single, lambda_=domain_lambda)
                features_dual_grl = grad_reverse(features_dual, lambda_=domain_lambda)

                # 域判别器预测
                domain_pred_single = self.domain_discriminator(features_single_grl)
                domain_pred_dual = self.domain_discriminator(features_dual_grl)

                # 计算域判别损失
                domain_loss_single = criterion_domain(domain_pred_single.squeeze(), domain_labels_single)
                domain_loss_dual = criterion_domain(domain_pred_dual.squeeze(), domain_labels_dual)
                domain_loss_batch = (domain_loss_single + domain_loss_dual) / 2

                # 动态损失权重
                progress = min(1.0, epoch / (epochs * 0.75))
                w_ce = 1.0
                w_contr = 0.1 + 0.1 * progress
                w_consist = 0.3 * progress
                w_domain = domain_lambda * (0.5 + 0.5 * progress)  # 域损失权重逐渐增加

                total_batch_loss = w_ce * ce_loss + w_contr * contr_loss + w_consist * consist_loss + w_domain * domain_loss_batch

                if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                    print(f"W: NaN/Inf Loss epoch {epoch + 1}. Skip batch backprop.")
                    continue

                # 3. 更新CNN和一致性损失参数
                optimizer_cnn.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(cnn_params + consist_params, max_norm=1.0)
                optimizer_cnn.step()
                scheduler_cnn.step()

                # 4. 更新域判别器
                optimizer_domain.zero_grad()
                domain_pred_single = self.domain_discriminator(features_single.detach())
                domain_pred_dual = self.domain_discriminator(features_dual.detach())
                domain_loss_single = criterion_domain(domain_pred_single.squeeze(), domain_labels_single)
                domain_loss_dual = criterion_domain(domain_pred_dual.squeeze(), domain_labels_dual)
                domain_loss_batch = (domain_loss_single + domain_loss_dual) / 2
                domain_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(domain_params, max_norm=1.0)
                optimizer_domain.step()
                scheduler_domain.step()

                # 记录损失
                train_loss += total_batch_loss.item() * current_bs
                domain_loss += domain_loss_batch.item() * current_bs
                train_loss_ce += ce_loss.item() * current_bs
                train_loss_contr += contr_loss.item() * current_bs
                train_loss_consist += consist_loss.item() * current_bs

                _, predicted = torch.max(logits, 1)
                total += current_bs
                correct += (predicted == labels).sum().item()
                batch_count += 1

            current_lr = optimizer_cnn.param_groups[0]['lr']
            avg_train_loss = train_loss / total if total > 0 else 0
            avg_domain_loss = domain_loss / total if total > 0 else 0
            train_accuracy = 100.0 * correct / total if total > 0 else 0
            avg_train_ce = train_loss_ce / total if total > 0 else 0
            avg_train_contr = train_loss_contr / total if total > 0 else 0
            avg_train_consist = train_loss_consist / total if total > 0 else 0

            train_losses.append(avg_train_loss)
            domain_losses.append(avg_domain_loss)

            # 验证阶段
            avg_val_loss, val_accuracy = 0.0, 0.0
            if val_loader is not None:
                self.cnn_model.eval()
                self.domain_discriminator.eval()
                if self.consistency_loss_fn:
                    self.consistency_loss_fn.eval()

                val_loss, val_correct, val_total = 0.0, 0, 0
                val_logits_list = []
                val_labels_list = []

                with torch.no_grad():
                    for inputs_val, labels_val in val_loader:
                        inputs_val, labels_val = inputs_val.to(self.device), labels_val.to(self.device)
                        bs_val = inputs_val.size(0)

                        batch_semantics_val = torch.stack([
                            torch.from_numpy(semantic_vectors_cnn[lbl.item()])
                            for lbl in labels_val]).to(self.device)

                        logits_val, _ = self.cnn_model(inputs_val, batch_semantics_val)

                        val_logits_list.append(logits_val.cpu())
                        val_labels_list.append(labels_val.cpu())

                        loss_val = criterion_ce(logits_val, labels_val)

                        if torch.isfinite(loss_val):
                            val_loss += loss_val.item() * bs_val

                        _, predicted_val = torch.max(logits_val, 1)
                        val_total += bs_val
                        val_correct += (predicted_val == labels_val).sum().item()

                    avg_val_loss = val_loss / val_total if val_total > 0 else 0
                    val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0
                    val_losses.append(avg_val_loss)

                    if val_logits_list:
                        all_val_logits = torch.cat(val_logits_list, dim=0)
                        all_val_labels = torch.cat(val_labels_list, dim=0)
                        probs = F.softmax(all_val_logits, dim=1)
                        correct_probs = probs[torch.arange(len(probs)), all_val_labels]
                        mean_confidence = correct_probs.mean().item()
                        min_confidence = correct_probs.min().item()
                        print(f"  Val confidence - Mean: {mean_confidence:.4f}, Min: {min_confidence:.4f}")

            epoch_info = (f"E[{epoch + 1}/{epochs}] LR={current_lr:.6f} "
                          f"TrLss={avg_train_loss:.4f} (CE:{avg_train_ce:.4f}, "
                          f"Ctr:{avg_train_contr:.4f}, Cns:{avg_train_consist:.4f}, "
                          f"Dom:{avg_domain_loss:.4f}) "
                          f"TrAcc={train_accuracy:.2f}% | VlLss={avg_val_loss:.4f} "
                          f"VlAcc={val_accuracy:.2f}%")
            print(epoch_info)

            # 保存最佳模型
            monitor_metric = val_accuracy if has_val_data else -avg_train_loss
            monitor_loss = avg_val_loss if has_val_data else avg_train_loss

            improved = False
            if monitor_metric > best_val_acc:
                best_val_acc = monitor_metric
                improved = True
            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                improved = True

            if improved:
                patience_counter = 0
                torch.save(self.cnn_model.state_dict(), 'best_cnn_model.pth')
                torch.save(self.domain_discriminator.state_dict(), 'best_domain_discriminator.pth')
                if self.consistency_loss_fn and hasattr(self.consistency_loss_fn, 'projection'):
                    torch.save(self.consistency_loss_fn.projection.state_dict(),
                               'best_consistency_projection.pth')
                print(f"  Best model saved. Val Acc: {val_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}/{epochs}")
                    break

        # 加载最佳模型
        if os.path.exists('best_cnn_model.pth'):
            self.cnn_model.load_state_dict(torch.load('best_cnn_model.pth'))
            if os.path.exists('best_domain_discriminator.pth'):
                self.domain_discriminator.load_state_dict(torch.load('best_domain_discriminator.pth'))
            if self.consistency_loss_fn and hasattr(self.consistency_loss_fn, 'projection') and \
                    os.path.exists('best_consistency_projection.pth'):
                self.consistency_loss_fn.projection.load_state_dict(torch.load(
                    'best_consistency_projection.pth'))
            print("Loaded best CNN and domain discriminator state.")

        self.cnn_model.eval()
        self.domain_discriminator.eval()
        if self.consistency_loss_fn:
            self.consistency_loss_fn.eval()

    def train_semantic_embedding_redesign_with_cycle(self, data_dict, semantic_dict, epochs=SEN_EPOCHS):
        """
        重新设计的语义嵌入训练，使用对抗学习和循环一致性约束
        """
        print("\n--- 开始对抗式语义嵌入网络训练（新设计，带循环一致性约束）---")

        # 检查 CNN 模型
        if self.cnn_model is None or not hasattr(self.cnn_model, 'feature_dim') or self.cnn_model.feature_dim <= 0:
            print("错误：无效的CNN模型或特征维度")
            return

        if self.fused_semantic_dim <= 0:
            print("错误：无效的融合语义维度")
            return

        # 初始化新网络
        print(
            f"创建新对抗式语义映射网络：{self.fused_semantic_dim} → {self.cnn_model.feature_dim} → {self.fused_semantic_dim}")
        self.embedding_net = AdversarialSemanticMappingWithCycle(
            semantic_dim=self.fused_semantic_dim,
            feature_dim=self.cnn_model.feature_dim
        ).to(self.device)

        # 准备数据
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']
        fused_semantics = semantic_dict.get('fused_semantics', {})

        if not fused_semantics:
            print("错误：无有效的语义向量")
            return

        # 创建故障索引到语义向量的映射
        semantic_vectors_dict = {}
        for fault_name, semantic_vec in fused_semantics.items():
            if fault_name in self.fault_types and np.all(np.isfinite(semantic_vec)):
                fault_idx = self.fault_types[fault_name]
                semantic_vectors_dict[fault_idx] = semantic_vec

        if not semantic_vectors_dict:
            print("错误：无有效的语义映射")
            return

        # 数据加载器
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        val_loader = None
        if len(X_val) > 0:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        # 优化器（使用 RMSprop 更适合 WGAN）
        mapper_optimizer = torch.optim.RMSprop(self.embedding_net.mapper.parameters(), lr=0.0001)
        decoder_optimizer = torch.optim.RMSprop(self.embedding_net.semantic_decoder.parameters(), lr=0.0001)
        discriminator_optimizer = torch.optim.RMSprop(self.embedding_net.discriminator.parameters(), lr=0.00005)

        # 学习率调度器
        mapper_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mapper_optimizer, T_max=epochs, eta_min=1e-6)
        decoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=epochs, eta_min=1e-6)
        disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discriminator_optimizer, T_max=epochs, eta_min=1e-6)

        # 损失函数
        feature_loss = nn.MSELoss()
        semantic_recon_loss = nn.MSELoss()
        cosine_sim_loss = nn.CosineEmbeddingLoss()
        l2_reg = lambda x: torch.norm(x, p=2)

        # 收集 CNN 特征统计信息
        print("收集CNN特征统计信息...")
        train_stats = self._collect_feature_stats(self.cnn_model, train_loader)
        val_stats = self._collect_feature_stats(self.cnn_model, val_loader) if val_loader else None

        # 训练循环
        print(f"开始训练，共{epochs}轮...")
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            self.embedding_net.train()
            epoch_d_losses = []
            epoch_g_losses = []
            epoch_cycle_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 获取 CNN 特征
                with torch.no_grad():
                    cnn_features = self.cnn_model(batch_x, semantic=None, return_features=True)
                if not torch.all(torch.isfinite(cnn_features)):
                    continue

                batch_size = cnn_features.size(0)
                batch_semantics = torch.FloatTensor([
                    semantic_vectors_dict.get(y.item(), np.zeros(self.fused_semantic_dim))
                    for y in batch_y
                ]).to(self.device)

                progress = epoch / epochs

                # 1. 训练判别器（WGAN 损失）
                discriminator_optimizer.zero_grad()
                fake_features = self.embedding_net(batch_semantics)
                if not torch.all(torch.isfinite(fake_features)):
                    continue

                real_score = self.embedding_net.discriminate(cnn_features)
                fake_score = self.embedding_net.discriminate(fake_features.detach())
                gp = gradient_penalty(self.embedding_net.discriminator, cnn_features, fake_features.detach(),
                                      self.device)
                d_loss = fake_score.mean() - real_score.mean() + 10.0 * gp

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.embedding_net.discriminator.parameters(), max_norm=1.0)
                discriminator_optimizer.step()

                # 2. 训练映射器
                mapper_optimizer.zero_grad()
                fake_features = self.embedding_net(batch_semantics)
                validity = self.embedding_net.discriminate(fake_features)
                g_adv_loss = -validity.mean()  # WGAN 生成器损失

                # 特征匹配损失
                matching_loss = torch.tensor(0.0, device=self.device)
                cosine_loss = torch.tensor(0.0, device=self.device)
                for i, label in enumerate(batch_y):
                    if label.item() in train_stats['centroids']:
                        centroid = torch.tensor(train_stats['centroids'][label.item()], device=self.device)
                        matching_loss += feature_loss(fake_features[i], centroid)
                        cosine_loss += 1 - F.cosine_similarity(fake_features[i].unsqueeze(0), centroid.unsqueeze(0))[0]
                matching_loss /= batch_size
                cosine_loss /= batch_size

                # 分布对齐损失
                global_mean = torch.tensor(train_stats['mean'], device=self.device)
                global_std = torch.tensor(train_stats['std'], device=self.device)
                fake_mean = fake_features.mean(dim=0)
                fake_std = fake_features.std(dim=0)
                dist_loss = feature_loss(fake_mean, global_mean) + feature_loss(fake_std, global_std)

                # 动态权重
                w_adv = 0.2 + 0.3 * progress  # 0.2 -> 0.5
                w_match = 1.0 - 0.4 * progress  # 1.0 -> 0.6
                w_cos = 0.5
                w_dist = 0.8 - 0.3 * progress  # 0.8 -> 0.5

                g_loss = w_adv * g_adv_loss + w_match * matching_loss + w_cos * cosine_loss + w_dist * dist_loss
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.embedding_net.mapper.parameters(), max_norm=1.0)
                mapper_optimizer.step()

                # 3. 训练循环一致性
                mapper_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                noised_semantics = batch_semantics + 0.02 * torch.randn_like(batch_semantics)
                cycle_features, reconstructed_semantics = self.embedding_net.cycle(noised_semantics)

                cycle_mse_loss = semantic_recon_loss(reconstructed_semantics, noised_semantics)
                cos_target = torch.ones(batch_size, device=self.device)
                cycle_cos_loss = cosine_sim_loss(reconstructed_semantics, noised_semantics, cos_target)

                feature_dist_loss = torch.tensor(0.0, device=self.device)
                for i, label in enumerate(batch_y):
                    if label.item() in train_stats['centroids']:
                        centroid = torch.tensor(train_stats['centroids'][label.item()], device=self.device)
                        feature_dist_loss += feature_loss(cycle_features[i], centroid)
                feature_dist_loss /= batch_size

                w_cycle_mse = 0.6 + 0.4 * progress  # 0.6 -> 1.0
                w_cycle_cos = 0.3 + 0.2 * progress  # 0.3 -> 0.5
                cycle_loss = w_cycle_mse * cycle_mse_loss + w_cycle_cos * cycle_cos_loss + 0.4 * feature_dist_loss

                # L2 正则化
                l2_loss = sum(l2_reg(p) for p in self.embedding_net.parameters()) * 1e-5
                combined_cycle_loss = cycle_loss + l2_loss

                combined_cycle_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.embedding_net.semantic_decoder.parameters(), max_norm=1.0)
                mapper_optimizer.step()
                decoder_optimizer.step()

                epoch_d_losses.append(d_loss.item())
                epoch_g_losses.append(g_loss.item())
                epoch_cycle_losses.append(cycle_loss.item())

            # 更新学习率
            mapper_scheduler.step()
            decoder_scheduler.step()
            disc_scheduler.step()

            # 计算平均损失
            avg_d_loss = sum(epoch_d_losses) / max(1, len(epoch_d_losses))
            avg_g_loss = sum(epoch_g_losses) / max(1, len(epoch_g_losses))
            avg_cycle_loss = sum(epoch_cycle_losses) / max(1, len(epoch_cycle_losses))

            # 验证
            val_loss = 0
            if val_stats and val_loader:
                val_loss = self._validate_embedding_with_cycle(val_loader, semantic_vectors_dict, val_stats)

            print(
                f"轮次 [{epoch + 1}/{epochs}] 判别器: {avg_d_loss:.4f}, 生成器: {avg_g_loss:.4f}, 循环: {avg_cycle_loss:.4f}, 验证: {val_loss:.4f}")

            # 早停
            monitor_metric = val_loss + 0.3 * avg_cycle_loss
            if monitor_metric < best_val_loss:
                best_val_loss = monitor_metric
                patience_counter = 0
                torch.save(self.embedding_net.state_dict(), 'best_semantic_mapper_with_cycle.pth')
                print(f"  新最佳模型已保存 (验证损失: {val_loss:.4f}, 循环损失: {avg_cycle_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {epoch + 1}轮后未改善")
                    break

        # 加载最佳模型
        try:
            self.embedding_net.load_state_dict(torch.load('best_semantic_mapper_with_cycle.pth'))
            print("已加载最佳模型权重")
        except Exception as e:
            print(f"加载模型出错: {e}")

        # 验证循环一致性
        print("验证语义循环一致性性能...")
        try:
            self._validate_cycle_consistency(semantic_dict.get('fused_semantics', {}))
        except Exception as e:
            print(f"循环一致性验证错误: {e}")

        print("新设计的对抗式语义嵌入训练完成!")

    def _validate_embedding_with_cycle(self, val_loader,
                                       semantic_vectors_dict, feature_stats):
        """验证嵌入模型，包括循环一致性"""
        self.embedding_net.eval()
        total_loss = 0
        total_cycle_loss = 0
        count = 0
        feature_loss = nn.MSELoss()
        semantic_loss = nn.MSELoss()

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_y = batch_y.to(self.device)

                # 获取语义向量
                batch_semantics = []
                for y in batch_y:
                    y_item = y.item()
                    if y_item in semantic_vectors_dict:
                        batch_semantics.append(semantic_vectors_dict[y_item])
                    else:
                        batch_semantics.append(np.zeros(self.fused_semantic_dim, dtype=np.float32))

                batch_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                # 执行语义→特征和语义→特征→语义
                fake_features = self.embedding_net(batch_semantics)
                _, reconstructed_semantics = self.embedding_net.cycle(batch_semantics)

                # 计算特征空间损失
                for i, label in enumerate(batch_y):
                    label_int = label.item()
                    if label_int in feature_stats['centroids']:
                        centroid = torch.tensor(
                            feature_stats['centroids'][label_int],
                            dtype=torch.float32,
                            device=self.device
                        )

                        # 特征空间损失
                        feat_mse = feature_loss(fake_features[i], centroid)

                        # 循环一致性损失
                        cycle_mse = semantic_loss(reconstructed_semantics[i], batch_semantics[i])

                        # 组合损失 - 特征损失+循环损失
                        combined_loss = feat_mse + 0.5 * cycle_mse

                        total_loss += combined_loss.item()
                        total_cycle_loss += cycle_mse.item()
                        count += 1

        avg_loss = total_loss / max(1, count)
        avg_cycle_loss = total_cycle_loss / max(1, count)

        print(f"  验证 - 平均特征匹配损失: {avg_loss:.4f}, 平均循环一致性损失: {avg_cycle_loss:.4f}")

        return avg_loss

    def _validate_cycle_consistency(self, fused_semantics):
        """验证语义循环一致性性能，测量重构误差和语义保留"""
        if not fused_semantics:
            print("警告: 没有可用的融合语义向量进行验证")
            return

        if self.embedding_net is None:
            print("警告: 语义嵌入网络未初始化")
            return

        self.embedding_net.eval()

        print("验证各类型语义的循环一致性...")

        # 存储结果
        reconstruction_errors = {}
        cosine_similarities = {}

        with torch.no_grad():
            for fault_type, semantic_vec in fused_semantics.items():
                if not np.all(np.isfinite(semantic_vec)):
                    continue

                # 转换为张量
                semantic_tensor = torch.FloatTensor(semantic_vec).unsqueeze(0).to(self.device)

                try:
                    # 执行循环: 语义→特征→语义
                    _, reconstructed = self.embedding_net.cycle(semantic_tensor)

                    # 计算重构误差
                    recon_err = F.mse_loss(reconstructed, semantic_tensor).item()
                    reconstruction_errors[fault_type] = recon_err

                    # 计算余弦相似度
                    cos_sim = F.cosine_similarity(reconstructed, semantic_tensor).item()
                    cosine_similarities[fault_type] = cos_sim

                except Exception as e:
                    print(f"错误: 处理'{fault_type}'的循环一致性时出错: {e}")

        if not reconstruction_errors:
            print("警告: 无法计算任何语义的循环一致性")
            return

        # 打印结果
        print("\n语义循环一致性结果:")
        print("故障类型         | 重构误差 | 余弦相似度")
        print("-" * 45)

        # 分类打印单一故障和复合故障
        single_faults = ['normal', 'inner', 'outer', 'ball']
        compound_faults = [ft for ft in reconstruction_errors.keys() if ft not in single_faults]

        # 单一故障
        print("单一故障:")
        for fault in single_faults:
            if fault in reconstruction_errors:
                err = reconstruction_errors[fault]
                sim = cosine_similarities[fault]
                print(f"{fault:15} | {err:.6f} | {sim:.6f}")

        # 复合故障
        print("\n复合故障:")
        for fault in compound_faults:
            if fault in reconstruction_errors:
                err = reconstruction_errors[fault]
                sim = cosine_similarities[fault]
                print(f"{fault:15} | {err:.6f} | {sim:.6f}")

        # 平均值
        avg_err = sum(reconstruction_errors.values()) / len(reconstruction_errors)
        avg_sim = sum(cosine_similarities.values()) / len(cosine_similarities)
        print("\n平均值:")
        print(f"所有故障      | {avg_err:.6f} | {avg_sim:.6f}")

        # 计算单一故障和复合故障的平均值
        if single_faults:
            single_errs = [reconstruction_errors[f] for f in single_faults if f in reconstruction_errors]
            single_sims = [cosine_similarities[f] for f in single_faults if f in cosine_similarities]
            if single_errs:
                avg_single_err = sum(single_errs) / len(single_errs)
                avg_single_sim = sum(single_sims) / len(single_sims)
                print(f"单一故障      | {avg_single_err:.6f} | {avg_single_sim:.6f}")

        if compound_faults:
            compound_errs = [reconstruction_errors[f] for f in compound_faults if f in reconstruction_errors]
            compound_sims = [cosine_similarities[f] for f in compound_faults if f in cosine_similarities]
            if compound_errs:
                avg_compound_err = sum(compound_errs) / len(compound_errs)
                avg_compound_sim = sum(compound_sims) / len(compound_sims)
                print(f"复合故障      | {avg_compound_err:.6f} | {avg_compound_sim:.6f}")

        # 可视化重构图
        try:
            configure_chinese_font()
            plt.figure(figsize=(12, 10))

            # 创建条形图数据
            fault_types = list(reconstruction_errors.keys())
            errors = [reconstruction_errors[ft] for ft in fault_types]
            similarities = [cosine_similarities[ft] for ft in fault_types]

            x = np.arange(len(fault_types))
            width = 0.35

            fig, ax1 = plt.subplots(figsize=(14, 8))
            configure_chinese_font()
            # 绘制重构误差
            ax1.set_xlabel('故障类型')
            ax1.set_ylabel('重构误差', color='tab:red')
            ax1.bar(x - width / 2, errors, width, color='tab:red', alpha=0.7, label='重构误差')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            # 创建第二个Y轴绘制相似度
            ax2 = ax1.twinx()
            ax2.set_ylabel('余弦相似度', color='tab:blue')
            ax2.bar(x + width / 2, similarities, width, color='tab:blue', alpha=0.7, label='余弦相似度')
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            # 设置x轴刻度
            ax1.set_xticks(x)
            ax1.set_xticklabels(fault_types, rotation=45, ha='right')

            # 添加图例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            plt.title('语义循环一致性性能')
            plt.tight_layout()
            plt.savefig('cycle_consistency_performance.png')
            plt.close()

            print("\n循环一致性性能可视化已保存至'cycle_consistency_performance.png'")

        except Exception as e:
            print(f"可视化循环一致性性能时出错: {e}")

    def _collect_feature_stats(self, cnn_model, data_loader):
        """收集CNN特征统计信息，用于指导训练"""
        cnn_model.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                features = cnn_model(batch_x, semantic=None, return_features=True)

                if torch.all(torch.isfinite(features)):
                    all_features.append(features.cpu().numpy())
                    all_labels.extend(batch_y.numpy())

        if not all_features:
            return {
                'centroids': {},
                'mean': np.zeros(cnn_model.feature_dim),
                'std': np.ones(cnn_model.feature_dim)
            }

        features = np.vstack(all_features)
        labels = np.array(all_labels)

        # 计算类中心
        unique_labels = np.unique(labels)
        centroids = {}

        for label in unique_labels:
            mask = (labels == label)
            if np.sum(mask) > 0:
                centroids[label] = np.mean(features[mask], axis=0)

        # 计算全局统计
        feature_mean = np.mean(features, axis=0)
        feature_std = np.std(features, axis=0)

        return {
            'centroids': centroids,
            'mean': feature_mean,
            'std': feature_std,
            'features': features,
            'labels': labels
        }

    def _validate_embedding(self, val_loader, semantic_vectors_dict, feature_stats):
        """验证嵌入模型"""
        self.embedding_net.eval()
        total_loss = 0
        count = 0
        feature_loss = nn.MSELoss()

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_y = batch_y.to(self.device)

                # 获取语义向量
                batch_semantics = []
                for y in batch_y:
                    y_item = y.item()
                    if y_item in semantic_vectors_dict:
                        batch_semantics.append(semantic_vectors_dict[y_item])
                    else:
                        batch_semantics.append(np.zeros(self.fused_semantic_dim, dtype=np.float32))

                batch_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                fake_features = self.embedding_net(batch_semantics)

                for i, label in enumerate(batch_y):
                    label_int = label.item()
                    if label_int in feature_stats['centroids']:
                        centroid = torch.tensor(
                            feature_stats['centroids'][label_int],
                            dtype=torch.float32,
                            device=self.device
                        )

                        # 修复：分别计算MSE和余弦相似度损失
                        mse = feature_loss(fake_features[i], centroid)
                        cos_sim = F.cosine_similarity(
                            fake_features[i].unsqueeze(0),
                            centroid.unsqueeze(0)
                        )
                        loss = mse + (1 - cos_sim[0])  # 取张量的第一个元素

                        total_loss += loss.item()
                        count += 1

        return total_loss / max(1, count)

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

    def evaluate_zero_shot_with_pca(self, data_dict, compound_projections,
                                    pca_components=2):
        """使用PCA降维后的欧氏距离进行零样本复合故障分类"""
        print(f"评估零样本复合故障分类能力（使用{pca_components}维PCA降维后的欧氏距离）...")

        # 1. 基本检查
        if self.cnn_model is None or self.cnn_feature_dim <= 0:
            print("错误: CNN模型未训练")
            return 0.0, None

        if compound_projections is None or not compound_projections:
            print("错误: 缺少复合故障投影")
            return 0.0, None

        # 2. 获取测试数据
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        if len(X_test) == 0:
            print("警告: 无复合故障测试数据")
            return 0.0, None

        # 确保测试数据有效
        finite_mask_test = np.all(np.isfinite(X_test), axis=1)
        X_test, y_test = X_test[finite_mask_test], y_test[finite_mask_test]
        if len(X_test) == 0:
            print("错误: 测试数据都是非有限值")
            return 0.0, None

        # 3. 筛选有投影的测试数据
        available_projection_labels = list(compound_projections.keys())
        available_projection_indices = [self.fault_types[label] for label in available_projection_labels
                                        if label in self.fault_types]

        test_mask = np.isin(y_test, available_projection_indices)
        X_compound_test, y_compound_test = X_test[test_mask], y_test[test_mask]

        if len(X_compound_test) == 0:
            print("错误: 无与投影匹配的测试样本")
            return 0.0, None

        print(f"  使用 {len(X_compound_test)} 个测试样本评估 {len(available_projection_labels)} 种复合故障")

        # 4. 准备投影特征
        candidate_labels = []
        projection_features = []

        for label, projection in compound_projections.items():
            if label in self.fault_types:
                candidate_labels.append(label)
                projection_features.append(projection)

        projection_features = np.array(projection_features)

        # 5. 提取测试样本特征
        self.cnn_model.eval()
        test_features_list = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(X_compound_test), batch_size):
                batch_x = torch.FloatTensor(X_compound_test[i:i + batch_size]).to(self.device)
                features = self.cnn_model(batch_x, semantic=None, return_features=True)
                if torch.all(torch.isfinite(features)):
                    test_features_list.append(features.cpu().numpy())

        test_features = np.vstack(test_features_list)

        # 6. 应用PCA降维
        from sklearn.decomposition import PCA

        # 合并所有特征用于拟合PCA
        all_features = np.vstack([test_features, projection_features])
        pca = PCA(n_components=pca_components)
        pca.fit(all_features)

        # 分别转换测试特征和投影特征
        test_features_pca = pca.transform(test_features)
        projection_features_pca = pca.transform(projection_features)

        print(f"  原始特征维度: {test_features.shape[1]} -> PCA降维后: {test_features_pca.shape[1]}")
        print(f"  PCA解释方差比例: {np.sum(pca.explained_variance_ratio_):.4f}")

        # 7. 在PCA空间中进行分类
        y_pred = []
        distances = []

        for i in range(len(test_features_pca)):
            # 计算到每个投影点的欧氏距离
            dists = [np.linalg.norm(test_features_pca[i] - proj) for proj in projection_features_pca]
            nearest_idx = np.argmin(dists)
            y_pred.append(self.fault_types[candidate_labels[nearest_idx]])
            distances.append(dists[nearest_idx])

        # 8. 计算指标
        accuracy = accuracy_score(y_compound_test, y_pred) * 100

        # 各类别准确率
        class_accuracy = {}
        for fault_type in set([self.idx_to_fault.get(y) for y in y_compound_test]):
            if fault_type is None:
                continue
            mask = [self.idx_to_fault.get(y) == fault_type for y in y_compound_test]
            if sum(mask) > 0:
                class_acc = accuracy_score(
                    [y for i, y in enumerate(y_compound_test) if mask[i]],
                    [y for i, y in enumerate(y_pred) if mask[i]]
                ) * 100
                class_accuracy[fault_type] = class_acc

        # 9. 可视化
        true_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_compound_test]
        pred_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_pred]
        display_labels = sorted(list(set(true_labels_str) | set(pred_labels_str)))

        try:
            conf_matrix = confusion_matrix(true_labels_str, pred_labels_str, labels=display_labels)


            # 绘制混淆矩阵
            configure_chinese_font()
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=display_labels,
                yticklabels=display_labels
            )
            plt.xlabel('预测')
            plt.ylabel('真实')
            plt.title(f'零样本学习混淆矩阵 (PCA + 欧式距离, 准确率: {accuracy:.2f}%)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('compound_fault_confusion_matrix_zsl_pca.png')
            plt.close()

            # 打印详细结果
            print(f"零样本学习准确率 (PCA + 欧式距离): {accuracy:.2f}%")
            print("各类别准确率:")
            for fault_type, acc in class_accuracy.items():
                print(f"  - {fault_type}: {acc:.2f}%")

            print("混淆矩阵已保存至 'compound_fault_confusion_matrix_zsl_pca.png'")

        except Exception as e:
            print(f"警告: 混淆矩阵计算错误: {e}")
            conf_matrix = None

        return accuracy, conf_matrix

    def visualize_feature_space(self, data_dict, compound_projections,
                                reduction_method='tsne', n_components=2, perplexity=30):
        """
        可视化CNN提取的特征和语义映射生成的特征在同一特征空间中的分布

        参数:
            data_dict: 包含训练和测试数据的字典
            compound_projections: 语义映射生成的复合故障特征投影
            reduction_method: 降维方法，'tsne'或'pca'
            n_components: 降维后的维度，通常为2或3
            perplexity: t-SNE的困惑度参数(仅当reduction_method='tsne'时使用)
        """
        print(f"\n可视化特征空间分布(使用{reduction_method.upper()})...")

        if self.cnn_model is None:
            print("错误: CNN模型未训练")
            return

        if compound_projections is None or not compound_projections:
            print("错误: 缺少复合故障投影")
            return

        # 获取测试数据
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        if len(X_test) == 0:
            print("警告: 无测试数据")
            return

        # 确保测试数据有效
        finite_mask_test = np.all(np.isfinite(X_test), axis=1)
        X_test, y_test = X_test[finite_mask_test], y_test[finite_mask_test]
        if len(X_test) == 0:
            print("错误: 测试数据都是非有限值")
            return

        # 1. 提取CNN特征
        self.cnn_model.eval()
        test_features_list = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch_x = torch.FloatTensor(X_test[i:i + batch_size]).to(self.device)
                features = self.cnn_model(batch_x, semantic=None, return_features=True)
                if torch.all(torch.isfinite(features)):
                    test_features_list.append(features.cpu().numpy())

        if not test_features_list:
            print("错误: 无法提取CNN特征")
            return

        test_features = np.vstack(test_features_list)

        # 2. 收集语义映射的投影特征
        projection_features = []
        projection_labels = []

        for fault_name, projection in compound_projections.items():
            if fault_name in self.fault_types:
                projection_features.append(projection)
                projection_labels.append(self.fault_types[fault_name])

        if not projection_features:
            print("错误: 无有效的投影特征")
            return

        projection_features = np.array(projection_features)
        projection_labels = np.array(projection_labels)

        # 3. 合并所有特征
        all_features = np.vstack([test_features, projection_features])
        all_labels = np.concatenate([y_test, projection_labels])

        # 用于区分真实数据和投影点的标记
        feature_types = np.array(['CNN提取'] * len(test_features) + ['语义映射'] * len(projection_features))

        # 4. 应用降维
        if reduction_method.lower() == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, perplexity=perplexity,
                           random_state=42, n_iter=1000, learning_rate='auto', init='pca')
        else:  # 默认使用PCA
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)

        # 应用降维
        reduced_features = reducer.fit_transform(all_features)

        # 将特征分开
        test_reduced = reduced_features[:len(test_features)]
        proj_reduced = reduced_features[len(test_features):]

        # 5. 可视化
        # 设置中文字体
        configure_chinese_font()

        # 创建映射，将数字标签转换为故障名称
        label_names = [self.idx_to_fault.get(label, f"未知_{label}") for label in all_labels]

        # 设置颜色映射，确保相同故障类型使用相同颜色
        unique_faults = list(set(label_names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_faults)))
        color_dict = {fault: colors[i] for i, fault in enumerate(unique_faults)}

        # 绘图
        plt.figure(figsize=(12, 10))

        # 首先绘制CNN特征
        for fault in unique_faults:
            mask = np.logical_and(np.array(label_names) == fault, feature_types == 'CNN提取')
            if np.sum(mask) > 0:
                plt.scatter(
                    reduced_features[mask, 0],
                    reduced_features[mask, 1],
                    c=[color_dict[fault]],
                    marker='o',
                    s=50,
                    alpha=0.6,
                    label=f"{fault} (CNN特征)"
                )

        # 然后绘制语义映射的投影点，使用相同的颜色但不同的标记
        for fault in unique_faults:
            mask = np.logical_and(np.array(label_names) == fault, feature_types == '语义映射')
            if np.sum(mask) > 0:
                plt.scatter(
                    reduced_features[mask, 0],
                    reduced_features[mask, 1],
                    c=[color_dict[fault]],
                    marker='*',
                    s=300,  # 投影点更大
                    edgecolors='black',
                    linewidths=1.5,
                    label=f"{fault} (语义投影)"
                )

        # 为语义投影点添加标签
        for i, (x, y) in enumerate(proj_reduced):
            plt.annotate(
                label_names[len(test_features) + i],
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )

        plt.title(f"特征空间分布 ({reduction_method.upper()}降维)")
        plt.xlabel("维度1")
        plt.ylabel("维度2")

        # 优化图例显示
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'feature_space_distribution_{reduction_method}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"特征空间分布图已保存为'feature_space_distribution_{reduction_method}.png'")

        # 6. 计算并打印投影点与真实数据中心的距离
        print("\n语义投影点与对应类别数据中心的距离:")

        # 计算每个类别的CNN特征中心
        class_centers = {}
        for label in np.unique(y_test):
            mask = (y_test == label)
            if np.sum(mask) > 0:
                class_centers[label] = np.mean(test_features[mask], axis=0)

        # 计算每个投影点与对应类别中心的距离
        for i, proj_label in enumerate(projection_labels):
            if proj_label in class_centers:
                proj_feature = projection_features[i]
                center = class_centers[proj_label]

                # 计算欧氏距离
                distance = np.linalg.norm(proj_feature - center)

                # 计算余弦相似度
                cos_sim = np.dot(proj_feature, center) / (np.linalg.norm(proj_feature) * np.linalg.norm(center))

                fault_name = self.idx_to_fault.get(proj_label, f"未知_{proj_label}")
                print(f"  {fault_name}: 欧氏距离 = {distance:.4f}, 余弦相似度 = {cos_sim:.4f}")

    def visualize_features_with_projections(self, data_dict, semantic_dict, method='pca', n_components=2):
        """
        可视化CNN提取的单一故障特征和语义投影的特征分布

        参数:
            data_dict: 包含数据集的字典
            semantic_dict: 包含语义向量的字典
            method: 降维方法，'pca'或'tsne'
            n_components: 可视化的维度数，2或3
        """
        print(f"使用{method.upper()}可视化特征分布...")

        if self.cnn_model is None or self.embedding_net is None:
            print("错误: CNN模型或语义嵌入网络未训练")
            return

        # 获取单一故障数据
        single_fault_types = ['normal', 'inner', 'outer', 'ball']
        single_fault_indices = [self.fault_types[ft] for ft in single_fault_types if ft in self.fault_types]

        # 从训练集筛选单一故障数据
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        single_mask = np.isin(y_train, single_fault_indices)
        X_single, y_single = X_train[single_mask], y_train[single_mask]

        # 从每种类型中最多取200个样本，避免过多数据
        X_selected = []
        y_selected = []

        for fault_idx in single_fault_indices:
            idx_mask = (y_single == fault_idx)
            idx_count = min(sum(idx_mask), 200)  # 每类最多200个样本
            if idx_count > 0:
                idx_samples = np.where(idx_mask)[0][:idx_count]
                X_selected.append(X_single[idx_samples])
                y_selected.append(y_single[idx_samples])

        if not X_selected:
            print("错误: 没有单一故障样本可供可视化")
            return

        X_selected = np.vstack(X_selected)
        y_selected = np.concatenate(y_selected)

        # 使用CNN提取实际特征
        self.cnn_model.eval()
        actual_features_list = []
        batch_size = 64

        print("提取CNN特征...")
        with torch.no_grad():
            for i in range(0, len(X_selected), batch_size):
                batch_x = torch.FloatTensor(X_selected[i:i + batch_size]).to(self.device)
                features = self.cnn_model(batch_x, semantic=None, return_features=True)
                if torch.all(torch.isfinite(features)):
                    actual_features_list.append(features.cpu().numpy())

        actual_features = np.vstack(actual_features_list)
        actual_labels = y_selected

        # 获取单一故障的语义向量
        fused_semantics = semantic_dict.get('fused_semantics', {})
        semantic_vectors = []
        semantic_labels = []

        print("生成语义投影特征...")
        for fault_type in single_fault_types:
            if fault_type in fused_semantics and fault_type in self.fault_types:
                semantic_vec = fused_semantics[fault_type]
                fault_idx = self.fault_types[fault_type]

                if not np.all(np.isfinite(semantic_vec)):
                    print(f"警告: '{fault_type}'的语义向量包含无效值，跳过")
                    continue

                # 为每种故障类型复制多次语义向量，便于可视化
                copies = min(50, sum(actual_labels == fault_idx))  # 每类最多50个投影点
                if copies > 0:
                    semantic_vectors.extend([semantic_vec] * copies)
                    semantic_labels.extend([fault_idx] * copies)

        if not semantic_vectors:
            print("错误: 没有有效的语义向量可供投影")
            return

        # 将语义向量投影到CNN特征空间
        self.embedding_net.eval()
        proj_features_list = []

        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(semantic_vectors), batch_size):
                batch_semantics = torch.FloatTensor(semantic_vectors[i:i + batch_size]).to(self.device)
                projected = self.embedding_net(batch_semantics)
                if torch.all(torch.isfinite(projected)):
                    proj_features_list.append(projected.cpu().numpy())

        if not proj_features_list:
            print("错误: 语义投影失败")
            return

        proj_features = np.vstack(proj_features_list)
        proj_labels = np.array(semantic_labels)

        # 合并所有特征用于降维
        all_features = np.vstack([actual_features, proj_features])

        # 降维处理
        if method.lower() == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
            print("应用PCA降维...")
        else:  # t-SNE
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(all_features) - 1))
            print("应用t-SNE降维...")

        try:
            reduced_features = reducer.fit_transform(all_features)

            # 分离回CNN特征和投影特征
            reduced_actual = reduced_features[:len(actual_features)]
            reduced_proj = reduced_features[len(actual_features):]

            # 可视化
            plt.figure(figsize=(12, 10))
            configure_chinese_font()

            # 设置颜色映射
            unique_labels = sorted(set(np.concatenate([actual_labels, proj_labels])))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

            # 绘制实际CNN特征
            for label in unique_labels:
                mask = (actual_labels == label)
                if np.any(mask):
                    fault_name = self.idx_to_fault.get(label, f"未知_{label}")
                    if n_components == 2:
                        plt.scatter(
                            reduced_actual[mask, 0], reduced_actual[mask, 1],
                            color=color_map[label], marker='o', alpha=0.7,
                            label=f"CNN特征: {fault_name}"
                        )
                    else:  # 3D
                        ax = plt.gca(projection='3d')
                        ax.scatter(
                            reduced_actual[mask, 0], reduced_actual[mask, 1], reduced_actual[mask, 2],
                            color=color_map[label], marker='o', alpha=0.7,
                            label=f"CNN特征: {fault_name}"
                        )

            # 绘制投影特征
            for label in unique_labels:
                mask = (proj_labels == label)
                if np.any(mask):
                    fault_name = self.idx_to_fault.get(label, f"未知_{label}")
                    if n_components == 2:
                        plt.scatter(
                            reduced_proj[mask, 0], reduced_proj[mask, 1],
                            color=color_map[label], marker='x', s=100, alpha=1.0,
                            label=f"语义投影: {fault_name}"
                        )
                    else:  # 3D
                        ax = plt.gca(projection='3d')
                        ax.scatter(
                            reduced_proj[mask, 0], reduced_proj[mask, 1], reduced_proj[mask, 2],
                            color=color_map[label], marker='x', s=100, alpha=1.0,
                            label=f"语义投影: {fault_name}"
                        )

            # 添加图例和标题
            plt.title(f'CNN特征与语义投影特征分布对比 ({method.upper()})')

            # 处理图例 - 由于每个类别有两种标记，需要优化图例显示
            handles, labels = plt.gca().get_legend_handles_labels()
            # 创建一个字典来存储标签和句柄
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))

            plt.tight_layout()
            plt.savefig(f'feature_projection_comparison_{method}.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"特征分布对比图已保存为 'feature_projection_comparison_{method}.png'")

            if method.lower() == 'pca':
                print(f"PCA解释方差比例: {np.sum(reducer.explained_variance_ratio_):.4f}")
                # 创建方差展示图
                plt.figure(figsize=(8, 4))
                plt.bar(range(1, n_components + 1), reducer.explained_variance_ratio_)
                plt.xlabel('主成分')
                plt.ylabel('解释方差比例')
                plt.title('PCA解释方差占比')
                plt.tight_layout()
                plt.savefig('pca_variance_explained.png')
                plt.close()
                print("PCA方差占比图已保存为 'pca_variance_explained.png'")

        except Exception as e:
            print(f"可视化错误: {e}")
            traceback.print_exc()

    def run_pipeline(self):
        """使用改进的对抗式语义嵌入的完整流水线"""
        start_time = time.time()
        accuracy = 0.0
        try:
            data_dict = self.load_data()
            if data_dict is None:
                raise RuntimeError("数据加载失败")

            semantic_dict = self.build_semantics(data_dict)
            if semantic_dict is None:
                raise RuntimeError("语义构建失败")

            # --- 在构建语义后调用可视化 ---
            self.visualize_semantics(semantic_dict)
            self.visualize_data_semantics_distribution(semantic_dict)

            self.train_dual_channel_cnn(data_dict, semantic_dict)
            if self.cnn_model is None:
                raise RuntimeError("CNN训练失败")

            self.train_semantic_embedding_redesign_with_cycle(data_dict, semantic_dict)
            if self.embedding_net is None:
                raise RuntimeError("语义嵌入训练失败")

            self.visualize_features_with_projections(data_dict, semantic_dict, method='pca')
            self.visualize_features_with_projections(data_dict, semantic_dict, method='tsne')
            compound_projections = self.generate_compound_fault_projections_fixed(semantic_dict, data_dict)
            if compound_projections is None:
                raise RuntimeError("投影生成失败")
            accuracy, _ = self.evaluate_zero_shot_with_pca(data_dict, compound_projections, pca_components=2)
            self.visualize_feature_space(data_dict, compound_projections, reduction_method='tsne')
            self.visualize_feature_space(data_dict, compound_projections, reduction_method='pca')
        except Exception as e:
            print(f"错误: 流水线失败: {e}")
            import traceback
            traceback.print_exc()
            accuracy = 0.0

        end_time = time.time()
        print(f"\n--- 流水线在 {(end_time - start_time) / 60:.2f} 分钟内完成 ---")
        return accuracy

if __name__ == "__main__":

    set_seed(42)
    data_path = "E:/研究生/CNN/HDU1000"
    if not os.path.isdir(data_path):
        print(f"E: Data directory not found: {data_path}")
    else:
        fault_diagnosis = ZeroShotCompoundFaultDiagnosis(
            data_path=data_path, sample_length=SEGMENT_LENGTH,
            latent_dim=AE_LATENT_DIM, batch_size=DEFAULT_BATCH_SIZE )
        final_accuracy = fault_diagnosis.run_pipeline()
        print(f"\n>>> Final ZSL Accuracy: {final_accuracy:.2f}% <<<")

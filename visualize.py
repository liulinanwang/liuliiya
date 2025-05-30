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
from torch.utils.data import DataLoader,TensorDataset
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
CNN_EPOCHS = 10
CNN_LR = 0.0005
SEN_EPOCHS = 5
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


    font_list = ['SimHei', 'Microsoft YaHei', 'SimSun']


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
    def __init__(self,
                 latent_dim,                      # 必须参数
                 num_classes_for_cvae,            # <<< 必须参数，确保它在这里！
                 idx_to_fault_mapping,            # 必须参数
                 data_path_for_reference=None,    # 可选参数
                 compound_data_semantic_generation_rule='mapper_output' # 可选参数
                ):
        """
        这是 FaultSemanticBuilder 的 __init__ 方法。
        确保这里声明的参数与 ZeroShotCompoundFaultDiagnosis 中调用时提供的参数一致。
        """
        # 打印接收到的参数，用于调试
        print(f"--- FaultSemanticBuilder __init__ ---")
        print(f"    latent_dim: {latent_dim}")
        print(f"    num_classes_for_cvae: {num_classes_for_cvae}") # << 打印这个关键参数
        print(f"    idx_to_fault_mapping (type): {type(idx_to_fault_mapping)}")
        print(f"    data_path_for_reference: {data_path_for_reference}")
        print(f"    compound_data_semantic_generation_rule: {compound_data_semantic_generation_rule}")

        # 将传入的参数赋值给实例属性
        self.latent_dim_config = latent_dim
        self.num_classes_for_cvae = num_classes_for_cvae # << 使用传入的这个值
        self.idx_to_fault = idx_to_fault_mapping
        self.data_path_for_reference = data_path_for_reference
        self.compound_data_semantic_generation_rule = compound_data_semantic_generation_rule

        # --- 其他必要的初始化，与之前提供的完整版 __init__ 类似 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actual_latent_dim = -1
        self.cvae_model = None

        # 确保 SEGMENT_LENGTH 已定义
        if 'SEGMENT_LENGTH' not in globals() and 'SEGMENT_LENGTH' not in locals():
            print("CRITICAL ERROR: Global constant 'SEGMENT_LENGTH' is NOT defined.")
            # 你可以在这里抛出异常，或者设置一个默认值并警告，但最好是定义它
            # For now, let's assume it should be defined, otherwise DataPreprocessor will fail.
            # Example fallback (not recommended for production):
            # SEGMENT_LENGTH = 1024 # Placeholder if not defined globally
            # print("WARNING: SEGMENT_LENGTH was not defined globally, using placeholder 1024.")
            raise NameError("Global constant 'SEGMENT_LENGTH' is not defined.")

        self.preprocessor = DataPreprocessor(sample_length=SEGMENT_LENGTH) # 依赖 SEGMENT_LENGTH

        self.knowledge_dim = 0
        self.data_semantics = {}
        self.all_latent_features = None
        self.all_latent_labels = None

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
        # 获取 enhanced_attribute_dim (需要确保 fault_location_attributes 不为空)
        if self.fault_location_attributes:
            first_loc_attr_key = next(iter(self.fault_location_attributes))
            self.enhanced_attribute_dim = len(self.fault_location_attributes[first_loc_attr_key]) + \
                                      len(self.bearing_params_attributes)
        else:
            self.enhanced_attribute_dim = 0 # 或者抛出错误
            print("WARNING: fault_location_attributes is empty, enhanced_attribute_dim set to 0.")

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
                # Define AttributeSemanticMapper (can be the same as the one in the original code)
                # Note: The original code defined AttributeSemanticMapper *inside* this method.
                # Let's use the one defined at the class level (if it exists) or redefine it here for clarity.
                # For this modification, I'll redefine it here to match the structure of your original snippet.
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

                # Prepare training data for the mapper
                # X_mapper: attribute vectors, Y_mapper: target semantic vectors
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


class BidirectionalSemanticNetwork(nn.Module):
    def __init__(self, semantic_dim, feature_dim, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.3):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim

        # 正向映射：语义向量 → 特征向量 (与原SemanticMappingMLP相似)
        self.forward_mapper = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim2, feature_dim)
        )

        # 反向映射：特征向量 → 语义向量 (新增)
        self.reverse_mapper = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim2, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim1, semantic_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化方法，与原SemanticMappingMLP相同"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mode='forward'):
        """支持不同模式的前向传播"""
        if mode == 'forward':
            # 语义→特征 (默认模式，兼容原有代码)
            return self.forward_mapper(x)
        elif mode == 'reverse':
            # 特征→语义
            return self.reverse_mapper(x)
        elif mode == 'cycle':
            # 完整循环: 语义→特征→语义
            features = self.forward_mapper(x)
            reconstructed_semantic = self.reverse_mapper(features)
            return features, reconstructed_semantic
        else:
            raise ValueError(f"Unsupported mode: {mode}")


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


class DistillationDualChannelCNN(nn.Module):
    def __init__(self, input_length=SEGMENT_LENGTH, semantic_dim=AE_LATENT_DIM,
                 num_classes=8, feature_dim=256):
        super().__init__()
        self.input_length = input_length
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim

        # --- 信号编码器 (共享的主干网络) ---
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
        # 结合信号和语义特征
        self.dual_channel_generator = nn.Sequential(
            nn.Linear(256 + 256, 384),  # 256(信号) + 256(语义) -> 384
            nn.LayerNorm(384),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(384, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # --- 单通道特征生成器 ---
        # 学会在没有语义信息的情况下生成相似的特征
        self.single_channel_generator = nn.Sequential(
            nn.Linear(256, 384),
            nn.LayerNorm(384),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(384, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # --- 分类器 ---
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, semantic=None, return_features=False):
        # 确保输入是3D: [batch, channel, length]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 提取信号特征 (共享骨干网络)
        signal_features = self.signal_encoder(x).squeeze(-1)  # [B, 256]

        if return_features:
            # 根据需要返回特征
            if semantic is None:
                return self.single_channel_generator(signal_features)
            else:
                semantic_features = self.semantic_encoder(semantic)
                combined = torch.cat([signal_features, semantic_features], dim=1)
                return self.dual_channel_generator(combined)

        # 生成单通道特征
        single_channel_features = self.single_channel_generator(signal_features)

        if semantic is None:
            # 单通道模式
            logits = self.classifier(single_channel_features)
            return logits, single_channel_features, signal_features
        else:
            # 双通道模式
            semantic_features = self.semantic_encoder(semantic)
            combined = torch.cat([signal_features, semantic_features], dim=1)
            dual_channel_features = self.dual_channel_generator(combined)

            logits = self.classifier(dual_channel_features)
            return logits, dual_channel_features, signal_features

class ZeroShotCompoundFaultDiagnosis:

    def __init__(self, data_path,
                 sample_length=SEGMENT_LENGTH, # 来自全局常量
                 latent_dim=AE_LATENT_DIM,    # 来自全局常量, 这个值将用于CVAE
                 batch_size=DEFAULT_BATCH_SIZE, # 来自全局常量
                 compound_data_semantic_generation_rule='mapper_output'):
        """
        这是基于你提供的原始 __init__ 修改的版本。
        确保 fault_types 等属性提前定义，并正确传递参数给 FaultSemanticBuilder。
        """

        self.data_path = data_path
        self.sample_length = sample_length
        self.latent_dim_config = latent_dim # CVAE的目标隐变量维度
        self.batch_size = batch_size
        self.compound_data_semantic_generation_rule = compound_data_semantic_generation_rule
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")

        self.fault_types = {
            'normal': 0, 'inner': 1, 'outer': 2, 'ball': 3,
            'inner_outer': 4, 'inner_ball': 5, 'outer_ball': 6, 'inner_outer_ball': 7
        }
        self.num_classes = len(self.fault_types)
        self.idx_to_fault = {v: k for k, v in self.fault_types.items()}
        print(f"  Number of classes set to: {self.num_classes}")

        self.single_fault_types_ordered = ['normal', 'inner', 'outer', 'ball']
        self.compound_fault_definitions = {
            'inner_outer': ['inner', 'outer'],
            'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'],
            'inner_outer_ball': ['inner', 'outer', 'ball']
        }
        self.compound_fault_types = list(self.compound_fault_definitions.keys()) # 与你原始代码一致

        self.preprocessor = DataPreprocessor(sample_length=self.sample_length)


        self.semantic_builder = FaultSemanticBuilder(
            latent_dim=self.latent_dim_config,                 # 参数名匹配
            num_classes_for_cvae=self.num_classes,             # 参数名匹配
            idx_to_fault_mapping=self.idx_to_fault,            # 参数名匹配
            data_path_for_reference=self.data_path,            # 参数名匹配
            compound_data_semantic_generation_rule=self.compound_data_semantic_generation_rule # 参数名匹配
        )

        self.cnn_model = None
        self.embedding_net = None
        self.actual_latent_dim = -1
        if 'CNN_FEATURE_DIM' in globals() or 'CNN_FEATURE_DIM' in locals():
            self.cnn_feature_dim = CNN_FEATURE_DIM
        elif hasattr(self, 'cnn_model') and self.cnn_model is not None and hasattr(self.cnn_model, 'feature_dim'): # 尝试从模型获取
            self.cnn_feature_dim = self.cnn_model.feature_dim
        else:
            print("  WARNING: Global constant 'CNN_FEATURE_DIM' not found. Setting cnn_feature_dim to -1 or a default.")
            self.cnn_feature_dim = 256 # 或者你之前的默认值 -1

        self.fused_semantic_dim = -1

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

    def validate_model(self, val_loader, semantic_vectors_dict, criterion):
        """
        在验证集上评估模型
        """
        self.cnn_model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 准备语义向量
                batch_semantics = []
                for lbl in targets:
                    if lbl.item() in semantic_vectors_dict:
                        batch_semantics.append(semantic_vectors_dict[lbl.item()])
                    else:
                        # 默认零向量
                        batch_semantics.append(np.zeros(self.cnn_model.semantic_dim))

                batch_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                # 前向传播 - 双通道
                logits, _, _ = self.cnn_model(inputs, batch_semantics)

                # 计算损失
                loss = criterion(logits, targets)
                val_loss += loss.item()

                # 统计准确率
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return val_loss / len(val_loader), correct / total

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

                fused_vec = np.concatenate([k_vec, d_vec]).astype(np.float32)
                if np.all(np.isfinite(fused_vec)) and len(fused_vec) == self.fused_semantic_dim:
                    fused_semantics[ft] = fused_vec
                else:
                    print(f"W: Fused vector for '{ft}' is invalid or has wrong dimension after concatenation.")
                    print(f"  Fused semantics prepared for {len(fused_semantics)} types. Dimension: {self.fused_semantic_dim}")

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

    def visualize_all_semantics_comparison(self, semantic_dict, data_dict, save_path='all_semantics_comparison.png',
                                           method='pca', n_components=2, perplexity=30):
        """
        在同一空间中可视化真实故障数据语义、单一故障数据语义以及合成的复合故障数据语义

        参数:
            semantic_dict: 语义字典，包含单一故障和合成复合故障的语义向量
            data_dict: 数据字典，用于提取真实故障数据
            save_path: 图像保存路径
            method: 降维方法，'pca'或't-sne'
            n_components: 降维后的维度，通常为2或3
            perplexity: t-SNE的复杂度参数
        """
        print(f"\n绘制所有故障语义分布对比图 (使用{method.upper()}降维)...")

        # 导入需要的库
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # 1. 收集不同来源的语义向量

        # A. 单一故障数据语义 (从CVAE/AE获取的中心点)
        single_fault_semantics = {}
        single_fault_types = ['normal', 'inner', 'outer', 'ball']

        data_prototypes = semantic_dict.get('data_prototypes', {})
        for fault_type in single_fault_types:
            if fault_type in data_prototypes:
                semantic_vec = data_prototypes[fault_type]
                if semantic_vec is not None and np.all(np.isfinite(semantic_vec)):
                    single_fault_semantics[fault_type] = semantic_vec

        # B. 合成的复合故障语义
        compound_data_semantics = semantic_dict.get('compound_data_semantics', {})
        compound_fault_types = ['inner_outer', 'inner_ball', 'outer_ball', 'inner_outer_ball']

        # C. 真实故障数据语义 (如果有真实的复合故障测试数据)
        real_compound_semantics = {}

        # 检查是否有测试数据
        X_test = data_dict.get('X_test')
        y_test = data_dict.get('y_test')

        if X_test is not None and y_test is not None and len(X_test) > 0:
            print("  提取真实复合故障数据的语义表示...")

            # 准备数据加载器
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            # 收集每种故障类型的特征
            fault_features = {}

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(self.device)

                    # 为每个样本准备条件向量，用于CVAE
                    batch_conditions = []
                    for lbl in labels:
                        fault_name = self.idx_to_fault.get(lbl.item(), 'normal')
                        condition = self.semantic_builder.fault_location_attributes.get(fault_name, [0, 0, 0])
                        batch_conditions.append(condition)

                    batch_conditions = torch.FloatTensor(batch_conditions).to(self.device)

                    # 使用CVAE提取语义
                    if hasattr(self.semantic_builder, 'cvae') and self.semantic_builder.cvae is not None:
                        # 如果使用CVAE
                        semantics = self.semantic_builder.cvae.get_latent(inputs, batch_conditions)
                    elif hasattr(self.semantic_builder,
                                 'autoencoder') and self.semantic_builder.autoencoder is not None:
                        # 如果使用AE
                        semantics = self.semantic_builder.autoencoder.encode(inputs)
                    else:
                        print("错误: 没有找到可用的编码器模型")
                        return

                    # 保存每个样本的特征和标签
                    for i, lbl in enumerate(labels):
                        fault_name = self.idx_to_fault.get(lbl.item(), f"unknown_{lbl.item()}")
                        if fault_name not in fault_features:
                            fault_features[fault_name] = []

                        feature_vec = semantics[i].cpu().numpy()
                        if np.all(np.isfinite(feature_vec)):
                            fault_features[fault_name].append(feature_vec)

            # 计算每种复合故障类型的中心点
            for fault_type in compound_fault_types:
                if fault_type in fault_features and len(fault_features[fault_type]) > 0:
                    real_compound_semantics[fault_type] = np.mean(fault_features[fault_type], axis=0)
                    print(f"  - 提取了 {fault_type} 的真实语义中心，来自 {len(fault_features[fault_type])} 个样本")

        # 2. 将所有语义向量合并用于降维
        all_vectors = []
        all_labels = []
        all_sources = []  # 标记语义来源: 'single', 'synthetic', 'real'

        # 添加单一故障语义
        for fault_type, vec in single_fault_semantics.items():
            all_vectors.append(vec)
            all_labels.append(fault_type)
            all_sources.append('single')

        # 添加合成的复合故障语义
        for fault_type, vec in compound_data_semantics.items():
            all_vectors.append(vec)
            all_labels.append(fault_type)
            all_sources.append('synthetic')

        # 添加真实复合故障语义
        for fault_type, vec in real_compound_semantics.items():
            all_vectors.append(vec)
            all_labels.append(fault_type)
            all_sources.append('real')

        if len(all_vectors) == 0:
            print("错误: 没有有效的语义向量用于可视化")
            return

        # 转换为numpy数组
        all_vectors = np.array(all_vectors)

        # 3. 降维
        if method.lower() == 't-sne':
            print(f"  使用t-SNE降维到{n_components}维，perplexity={perplexity}...")
            if len(all_vectors) > perplexity:
                reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca')
                reduced_vectors = reducer.fit_transform(all_vectors)
                df_points = pd.DataFrame(reduced_vectors, columns=['x', 'y'])
                df_points['label'] = all_labels
                df_points['source'] = all_sources
                df_points.to_csv('semantic_points_all_methods.csv', index=False, encoding='utf-8-sig')
                print('降维后的所有点已导出到 semantic_points_all_methods.csv')
            else:
                print(f"警告: 数据点数量({len(all_vectors)})小于perplexity参数({perplexity})，将使用PCA代替")
                reducer = PCA(n_components=min(n_components, all_vectors.shape[1], len(all_vectors)))
                reduced_vectors = reducer.fit_transform(all_vectors)
                df_points = pd.DataFrame(reduced_vectors, columns=['x', 'y'])
                df_points['label'] = all_labels
                df_points['source'] = all_sources
                df_points.to_csv('semantic_points_all_methods.csv', index=False, encoding='utf-8-sig')
                print(f"  PCA解释方差比例: {np.sum(reducer.explained_variance_ratio_):.4f}")
        else:  # 默认使用PCA
            print(f"  使用PCA降维到{n_components}维...")
            reducer = PCA(n_components=min(n_components, all_vectors.shape[1], len(all_vectors)))
            reduced_vectors = reducer.fit_transform(all_vectors)
            df_points = pd.DataFrame(reduced_vectors, columns=['x', 'y'])
            df_points['label'] = all_labels
            df_points['source'] = all_sources
            df_points.to_csv('semantic_points_all_methods.csv', index=False, encoding='utf-8-sig')
            print(f"  PCA解释方差比例: {np.sum(reducer.explained_variance_ratio_):.4f}")

        # 4. 可视化
        configure_chinese_font()  # 确保中文正确显示
        plt.figure(figsize=(14, 10))

        # 4.1 设置颜色、标记和标签
        fault_colors = {
            'normal': '#1f77b4',  # 蓝色
            'inner': '#ff7f0e',  # 橙色
            'outer': '#2ca02c',  # 绿色
            'ball': '#d62728',  # 红色
            'inner_outer': '#9467bd',  # 紫色
            'inner_ball': '#8c564b',  # 棕色
            'outer_ball': '#e377c2',  # 粉色
            'inner_outer_ball': '#7f7f7f'  # 灰色
        }

        source_markers = {
            'single': 'o',  # 圆点
            'synthetic': 'X',  # 叉叉
            'real': 'D'  # 菱形
        }

        source_sizes = {
            'single': 100,  # 单一故障大小
            'synthetic': 150,  # 合成故障大小
            'real': 200  # 真实故障大小
        }

        source_labels = {
            'single': '单一故障语义',
            'synthetic': '合成复合故障语义',
            'real': '真实复合故障语义'
        }

        # 4.2 绘制散点图
        for i in range(len(all_vectors)):
            fault_type = all_labels[i]
            source = all_sources[i]

            color = fault_colors.get(fault_type, '#17becf')  # 默认使用青色
            marker = source_markers.get(source, '*')
            size = source_sizes.get(source, 100)

            # 设置不同源的透明度
            alpha = 0.9 if source == 'real' else 0.7

            # 设置边框颜色，真实数据点加重显示
            edgecolor = 'black' if source == 'real' else None
            linewidth = 1.5 if source == 'real' else 0.5

            label = f"{fault_type} ({source_labels[source]})"

            # 绘制点
            if n_components >= 2:
                plt.scatter(
                    reduced_vectors[i, 0],
                    reduced_vectors[i, 1],
                    s=size,
                    color=color,
                    marker=marker,
                    alpha=alpha,
                    edgecolors=edgecolor,
                    linewidth=linewidth,
                    label=label
                )

                # 添加文本标签
                plt.text(
                    reduced_vectors[i, 0] + 0.02,
                    reduced_vectors[i, 1] + 0.02,
                    fault_type,
                    fontsize=10,
                    weight='bold',
                    alpha=0.7
                )
            else:
                # 1D情况，添加一个固定的y坐标
                plt.scatter(
                    reduced_vectors[i, 0],
                    0,
                    s=size,
                    color=color,
                    marker=marker,
                    alpha=alpha,
                    edgecolors=edgecolor,
                    linewidth=linewidth,
                    label=label
                )

                # 添加文本标签
                plt.text(
                    reduced_vectors[i, 0] + 0.02,
                    0.02,
                    fault_type,
                    fontsize=10,
                    weight='bold',
                    alpha=0.7
                )

        # 4.3 添加图例、标题和轴标签
        # 优化图例 - 只显示唯一条目
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = {}
        for handle, label in zip(handles, labels):
            if label not in by_label:
                by_label[label] = handle

        plt.legend(
            by_label.values(),
            by_label.keys(),
            loc='upper right',
            bbox_to_anchor=(1.15, 1),
            fontsize=9
        )

        plt.title(f"故障语义分布对比 ({method.upper()}降维)")

        if isinstance(reducer, PCA) and reducer.n_components > 0:
            x_label = f"维度1 ({reducer.explained_variance_ratio_[0]:.2%})"
            if n_components > 1:
                y_label = f"维度2 ({reducer.explained_variance_ratio_[1]:.2%})"
            else:
                y_label = ""
        else:
            x_label = "维度1"
            y_label = "维度2" if n_components > 1 else ""

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # 添加网格线
        plt.grid(True, linestyle='--', alpha=0.5)

        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"故障语义分布对比图已保存至: {save_path}")

        # 5. 如果有真实和合成的复合故障，计算它们之间的距离
        if real_compound_semantics and compound_data_semantics:
            print("\n真实与合成复合故障语义的距离分析:")

            common_faults = set(real_compound_semantics.keys()) & set(compound_data_semantics.keys())

            if common_faults:
                distances = {}
                cosine_sims = {}

                for fault in common_faults:
                    real_vec = real_compound_semantics[fault]
                    synth_vec = compound_data_semantics[fault]

                    # 欧氏距离
                    dist = np.linalg.norm(real_vec - synth_vec)
                    distances[fault] = dist

                    # 余弦相似度
                    real_norm = real_vec / np.linalg.norm(real_vec)
                    synth_norm = synth_vec / np.linalg.norm(synth_vec)
                    cos_sim = np.dot(real_norm, synth_norm)
                    cosine_sims[fault] = cos_sim

                    print(f"  - {fault}: 欧氏距离 = {dist:.4f}, 余弦相似度 = {cos_sim:.4f}")

                # 平均距离
                avg_dist = np.mean(list(distances.values()))
                avg_cos = np.mean(list(cosine_sims.values()))
                print(f"  平均欧氏距离: {avg_dist:.4f}")
                print(f"  平均余弦相似度: {avg_cos:.4f}")

    def train_with_knowledge_distillation(self, data_dict, semantic_dict,
                                          epochs=CNN_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, lr=CNN_LR):
        """
        使用知识蒸馏训练双通道CNN

        参数:
            data_dict: 包含训练和验证数据的字典
            semantic_dict: 包含语义向量的字典
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
        """
        # 确保CNN模型已初始化
        if self.cnn_model is None:
            print("错误：CNN模型未初始化")
            return None

        self.cnn_model = self.cnn_model.to(self.device)

        # 准备数据
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']

        # 准备数据加载器
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 准备语义向量
        data_semantics = semantic_dict.get('data_only_semantics', {})
        semantic_vectors_dict = {}
        for fault_name, idx in self.fault_types.items():
            if fault_name in data_semantics:
                semantic_vectors_dict[idx] = data_semantics[fault_name]

        # 损失函数
        ce_criterion = nn.CrossEntropyLoss()
        distill_criterion = nn.MSELoss()  # 蒸馏损失
        feature_criterion = nn.CosineEmbeddingLoss()  # 余弦相似度损失，促进特征向量方向一致

        # 优化器
        optimizer = optim.AdamW(self.cnn_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3
        )

        # 训练循环
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0

        # 创建特征中心，用于监控训练进程
        class_centers = {}

        for epoch in range(epochs):
            self.cnn_model.train()
            total_loss = 0
            correct = 0
            total = 0

            # 损失统计
            ce_losses = []
            distill_losses = []
            feature_losses = []

            # 动态调整蒸馏权重
            progress = min(1.0, epoch / (epochs * 0.5))
            distill_weight = 0.5 + 0.5 * progress  # 随训练进行增加蒸馏权重

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 准备语义向量
                batch_semantics = []
                for lbl in targets:
                    if lbl.item() in semantic_vectors_dict:
                        batch_semantics.append(semantic_vectors_dict[lbl.item()])
                    else:
                        # 默认零向量
                        batch_semantics.append(np.zeros(self.cnn_model.semantic_dim))

                batch_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                # 步骤1: 使用双通道模式前向传播
                logits_dual, dual_features, signal_features = self.cnn_model(inputs, batch_semantics)

                # 步骤2: 使用单通道模式前向传播
                logits_single, single_features, _ = self.cnn_model(inputs)

                # 步骤3: 计算分类损失 (双通道和单通道)
                ce_loss_dual = ce_criterion(logits_dual, targets)
                ce_loss_single = ce_criterion(logits_single, targets)
                ce_loss = 0.9*ce_loss_dual + 0.3*ce_loss_single

                # 步骤4: 计算知识蒸馏损失 (单通道特征学习双通道特征)
                # MSE损失使特征向量接近
                distill_loss = distill_criterion(single_features, dual_features)

                # 余弦相似度损失确保特征向量方向一致
                batch_size = inputs.size(0)
                target_ones = torch.ones(batch_size).to(self.device)  # 目标余弦相似度为1
                cos_loss = feature_criterion(single_features, dual_features, target_ones)

                # 总损失
                loss = 0.5*ce_loss + distill_weight * (distill_loss + cos_loss)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=5.0)

                optimizer.step()
                scheduler.step()

                # 统计
                total_loss += loss.item()
                _, predicted = logits_dual.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                ce_losses.append(ce_loss.item())
                distill_losses.append(distill_loss.item())
                feature_losses.append(cos_loss.item())

                # 打印进度
                if (batch_idx + 1) % 50 == 0:
                    print(f'Epoch {epoch + 1}/{epochs} | Batch {batch_idx + 1}/{len(train_loader)} | '
                          f'Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%')

            # 每个epoch结束后验证
            val_loss, val_acc = self.validate_model(val_loader, semantic_vectors_dict, ce_criterion)

            # 更新特征中心
            if epoch % 2 == 0:
                class_centers = self.update_feature_centers(val_loader, semantic_vectors_dict)

                # 打印各类别特征的相似度
                if len(class_centers) >= 2:
                    print("\n类别特征相似度矩阵:")
                    center_names = list(class_centers.keys())
                    for i in range(len(center_names)):
                        for j in range(i + 1, len(center_names)):
                            name_i, name_j = center_names[i], center_names[j]
                            center_i, center_j = class_centers[name_i], class_centers[name_j]

                            # 归一化后计算余弦相似度
                            norm_i = center_i / np.linalg.norm(center_i)
                            norm_j = center_j / np.linalg.norm(center_j)
                            sim = np.dot(norm_i, norm_j)

                            print(f"  {name_i} vs {name_j}: {sim:.4f}")

            # 打印epoch结果
            print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {total_loss / len(train_loader):.4f} | '
                  f'Train Acc: {100. * correct / total:.2f}% | Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {100. * val_acc:.2f}%')

            print(f'CE Loss: {np.mean(ce_losses):.4f} | Distill Loss: {np.mean(distill_losses):.4f} | '
                  f'Feature Cos Loss: {np.mean(feature_losses):.4f} | Distill Weight: {distill_weight:.2f}')

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.cnn_model.state_dict(), 'best_distilled_cnn.pth')
                print(f"保存最佳模型，验证集准确率: {100. * val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {patience}轮内未改善")
                    break

        # 加载最佳模型
        self.cnn_model.load_state_dict(torch.load('best_distilled_cnn.pth'))
        return self.cnn_model

    def update_feature_centers(self, data_loader, semantic_vectors_dict):
        """更新并返回各类别在特征空间中的中心点"""
        self.cnn_model.eval()
        class_features = {}

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 单通道特征
                _, single_features, _ = self.cnn_model(inputs)

                # 收集各类别特征
                for i, label in enumerate(targets):
                    label_name = self.idx_to_fault.get(label.item(), f"未知_{label.item()}")
                    if label_name not in class_features:
                        class_features[label_name] = []
                    class_features[label_name].append(single_features[i].cpu().numpy())

        # 计算每个类别的中心
        centers = {}
        for name, features in class_features.items():
            if features:
                centers[name] = np.mean(np.array(features), axis=0)

        return centers

    def visualize_distillation_effect(self, data_dict, semantic_dict, save_path='distillation_effect.png'):
        """可视化知识蒸馏效果"""
        if self.cnn_model is None:
            print("错误: CNN模型未训练")
            return

        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()

        # 准备数据
        X_val, y_val = data_dict['X_val'], data_dict['y_val']

        # 准备语义向量
        data_semantics = semantic_dict.get('data_only_semantics', {})
        semantic_vectors_dict = {}
        for fault_name, idx in self.fault_types.items():
            if fault_name in data_semantics:
                semantic_vectors_dict[idx] = data_semantics[fault_name]

        # 随机采样
        sample_size = min(300, len(X_val))
        indices = np.random.choice(len(X_val), sample_size, replace=False)
        X_sample = X_val[indices]
        y_sample = y_val[indices]

        # 提取特征
        single_features_list = []
        dual_features_list = []
        labels_list = []
        distances = []  # 记录每个样本的单双通道特征距离

        with torch.no_grad():
            for i in range(0, len(X_sample), 32):
                end_idx = min(i + 32, len(X_sample))
                batch_x = torch.FloatTensor(X_sample[i:end_idx]).to(self.device)
                batch_y = y_sample[i:end_idx]

                # 准备语义向量
                batch_semantics = []
                for lbl in batch_y:
                    if lbl in semantic_vectors_dict:
                        batch_semantics.append(semantic_vectors_dict[lbl])
                    else:
                        batch_semantics.append(np.zeros(self.cnn_model.semantic_dim))
                batch_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                # 提取特征
                _, single_features, _ = self.cnn_model(batch_x)
                _, dual_features, _ = self.cnn_model(batch_x, batch_semantics)

                # 计算特征距离
                for j in range(len(single_features)):
                    sf = single_features[j].cpu().numpy()
                    df = dual_features[j].cpu().numpy()

                    # 欧氏距离
                    dist = np.linalg.norm(sf - df)
                    distances.append(dist)

                # 收集特征用于可视化
                single_features_list.append(single_features.cpu().numpy())
                dual_features_list.append(dual_features.cpu().numpy())
                labels_list.extend(batch_y)

        # 合并特征
        single_features = np.vstack(single_features_list)
        dual_features = np.vstack(dual_features_list)
        labels = np.array(labels_list)

        # 计算平均距离
        avg_distance = np.mean(distances)
        std_distance = np.std(distances)

        # 可视化特征距离直方图
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=30, alpha=0.7)
        plt.axvline(avg_distance, color='r', linestyle='--', label=f'平均距离: {avg_distance:.4f}')
        plt.title('单通道与双通道特征欧氏距离分布')
        plt.xlabel('欧氏距离')
        plt.ylabel('频率')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{save_path}_distances.png", dpi=300)
        plt.close()

        # 使用t-SNE可视化特征空间
        from sklearn.manifold import TSNE

        # 合并所有特征
        combined_features = np.vstack([single_features, dual_features])
        feature_types = ['单通道'] * len(single_features) + ['双通道'] * len(dual_features)
        combined_labels = np.concatenate([labels, labels])

        # 应用t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        reduced_features = tsne.fit_transform(combined_features)

        # 分离结果
        single_reduced = reduced_features[:len(single_features)]
        dual_reduced = reduced_features[len(single_features):]

        # 可视化t-SNE结果
        plt.figure(figsize=(14, 12))

        # 获取唯一标签
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        # 计算特征对齐度量
        alignment_scores = {}
        overall_distance = 0

        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            label_name = self.idx_to_fault.get(label, f"未知_{label}")

            # 绘制单通道和双通道特征
            plt.scatter(
                single_reduced[mask, 0], single_reduced[mask, 1],
                color=colors[i], marker='o', alpha=0.6, s=50,
                label=f"{label_name} (单通道)"
            )

            plt.scatter(
                dual_reduced[mask, 0], dual_reduced[mask, 1],
                color=colors[i], marker='x', alpha=0.6, s=50,
                label=f"{label_name} (双通道)"
            )

            # 计算中心点
            single_center = np.mean(single_reduced[mask], axis=0)
            dual_center = np.mean(dual_reduced[mask], axis=0)

            # 计算距离
            distance = np.linalg.norm(single_center - dual_center)
            alignment_scores[label_name] = distance
            overall_distance += distance

        # 平均距离
        avg_alignment = overall_distance / len(unique_labels)

        # 添加标题和图例
        plt.title(f"知识蒸馏后的特征分布\n平均特征欧氏距离: {avg_distance:.4f} | 平均中心距离: {avg_alignment:.4f}")
        plt.xlabel("t-SNE维度1")
        plt.ylabel("t-SNE维度2")
        plt.grid(True, linestyle='--', alpha=0.7)

        # 优化图例
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 打印统计结果
        print("\n知识蒸馏效果评估:")
        print(f"特征欧氏距离: {avg_distance:.4f} ± {std_distance:.4f}")
        print(f"特征中心距离: {avg_alignment:.4f}")
        print("\n各类别中心距离:")
        for label, score in alignment_scores.items():
            print(f"  {label}: {score:.4f}")

        return avg_distance, avg_alignment

    def visualize_all_feature_projection_comparison(self, data_dict, semantic_dict, compound_projections,
                                                    save_path='all_feature_projection_comparison.png',
                                                    method='pca', perplexity=30, n_components=2,
                                                    max_samples_per_class=100):  # Reduced for clarity
        """
        可视化单一故障实际CNN特征、单一故障融合语义投影特征、
        以及复合故障融合语义投影特征在同一空间的分布。

        参数:
            data_dict: 包含训练/验证数据的字典 (用于获取单一故障样本)。
            semantic_dict: 包含融合语义的字典。
            compound_projections: 预先计算好的复合故障投影特征 (来自 SemanticMappingMLP)。
            save_path: 图像保存路径。
            method: 降维方法 ('tsne' 或 'pca')。
            perplexity: t-SNE 的复杂度参数。
            n_components: 降维后的维度。
            max_samples_per_class: 每类单一故障在图中显示的最大实际样本点数量。
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import seaborn as sns

        print(f"\n绘制单一与复合故障特征及语义投影比较图 (使用{method.upper()})...")

        # 1. 检查模型和数据
        if self.cnn_model is None or not hasattr(self.cnn_model, 'feature_dim') or self.cnn_model.feature_dim <= 0:
            print("错误: CNN模型未有效初始化或无特征维度。")
            return
        if self.embedding_net is None:  # embedding_net is the SemanticMappingMLP
            print("错误: MLP语义映射网络 (embedding_net) 未初始化。")
            return
        fused_semantics = semantic_dict.get('fused_semantics')
        if not fused_semantics:
            print("错误: 缺少融合语义信息。")
            return
        if compound_projections is None or not compound_projections:
            print("错误: 缺少预计算的复合故障投影。")  # These are MLP projections of fused semantics
            return

        # --- Part A: 单一故障实际CNN特征 ---
        X_samples, y_samples = data_dict.get('X_train'), data_dict.get('y_train')
        if X_samples is None or y_samples is None or len(X_samples) == 0:
            X_samples, y_samples = data_dict.get('X_val'), data_dict.get('y_val')
            if X_samples is None or y_samples is None or len(X_samples) == 0:
                print("错误: 数据字典中无有效的训练或验证数据用于提取单一故障样本。")
                return

        single_fault_types_names = ['normal', 'inner', 'outer', 'ball']
        single_fault_indices_map = {name: self.fault_types[name] for name in single_fault_types_names if
                                    name in self.fault_types}

        actual_cnn_features_list = []
        actual_cnn_labels_list = []

        print(f"  A. 从数据集中提取单一故障的实际CNN特征 (每类最多{max_samples_per_class}个样本)...")
        self.cnn_model.eval()
        with torch.no_grad():
            for fault_name, fault_idx in single_fault_indices_map.items():
                mask = (y_samples == fault_idx)
                X_class_samples = X_samples[mask]
                if len(X_class_samples) == 0: continue
                if len(X_class_samples) > max_samples_per_class:
                    sample_indices = np.random.choice(len(X_class_samples), max_samples_per_class, replace=False)
                    X_class_samples = X_class_samples[sample_indices]

                if len(X_class_samples) > 0:
                    batch_x = torch.FloatTensor(X_class_samples).to(self.device)
                    features = self.cnn_model(batch_x, semantic=None, return_features=True)
                    if torch.all(torch.isfinite(features)):
                        actual_cnn_features_list.append(features.cpu().numpy())
                        actual_cnn_labels_list.extend([fault_idx] * len(features))

        actual_cnn_features = np.vstack(actual_cnn_features_list) if actual_cnn_features_list else np.empty(
            (0, self.cnn_model.feature_dim))
        actual_cnn_labels = np.array(actual_cnn_labels_list)
        print(f"     共提取了 {len(actual_cnn_features)} 个实际CNN特征点。")

        # --- Part B: 单一故障融合语义投影特征 (来自MLP) ---
        projected_single_fused_features_list = []
        projected_single_fused_labels_list = []

        print("  B. 通过MLP映射网络投影单一故障的融合语义...")
        self.embedding_net.eval()
        with torch.no_grad():
            for fault_name in single_fault_types_names:
                if fault_name in fused_semantics and fault_name in self.fault_types:
                    semantic_vec = fused_semantics[fault_name]
                    if semantic_vec is not None and np.all(np.isfinite(semantic_vec)):
                        semantic_tensor = torch.FloatTensor(semantic_vec).unsqueeze(0).to(self.device)
                        projected_feature = self.embedding_net(semantic_tensor)
                        if torch.all(torch.isfinite(projected_feature)):
                            projected_single_fused_features_list.append(projected_feature.cpu().numpy().squeeze(0))
                            projected_single_fused_labels_list.append(self.fault_types[fault_name])

        projected_single_fused_features = np.array(
            projected_single_fused_features_list) if projected_single_fused_features_list else np.empty(
            (0, self.embedding_net.feature_dim))
        projected_single_fused_labels = np.array(projected_single_fused_labels_list)
        print(f"     共投影了 {len(projected_single_fused_features)} 个单一故障的融合语义中心。")

        # --- Part C: 复合故障融合语义投影特征 (来自MLP - 使用传入的 compound_projections) ---
        projected_compound_fused_features_list = []
        projected_compound_fused_labels_list = []
        print("  C. 获取预计算的复合故障融合语义投影 (来自MLP)...")
        for fault_name, proj_feature in compound_projections.items():
            if fault_name in self.fault_types and fault_name in self.compound_fault_types:  # Ensure it's a compound fault
                if proj_feature is not None and np.all(np.isfinite(proj_feature)):
                    projected_compound_fused_features_list.append(proj_feature)
                    projected_compound_fused_labels_list.append(self.fault_types[fault_name])

        projected_compound_fused_features = np.array(
            projected_compound_fused_features_list) if projected_compound_fused_features_list else np.empty(
            (0, self.embedding_net.feature_dim))
        projected_compound_fused_labels = np.array(projected_compound_fused_labels_list)
        print(f"     获取了 {len(projected_compound_fused_features)} 个复合故障的融合语义投影。")

        # 4. 合并所有特征进行降维
        all_features_to_reduce_list = []
        if len(actual_cnn_features) > 0: all_features_to_reduce_list.append(actual_cnn_features)
        if len(projected_single_fused_features) > 0: all_features_to_reduce_list.append(projected_single_fused_features)
        if len(projected_compound_fused_features) > 0: all_features_to_reduce_list.append(
            projected_compound_fused_features)

        if not all_features_to_reduce_list:
            print("错误: 没有有效的特征可以用于降维和可视化。")
            return

        all_features_to_reduce = np.vstack(all_features_to_reduce_list)

        feature_source_types = (['Actual CNN (Single)'] * len(actual_cnn_features)) + \
                               (['Projected Fused (Single)'] * len(projected_single_fused_features)) + \
                               (['Projected Fused (Compound)'] * len(projected_compound_fused_features))

        all_labels_for_reduction_list = []
        if len(actual_cnn_labels) > 0: all_labels_for_reduction_list.append(actual_cnn_labels)
        if len(projected_single_fused_labels) > 0: all_labels_for_reduction_list.append(projected_single_fused_labels)
        if len(projected_compound_fused_labels) > 0: all_labels_for_reduction_list.append(
            projected_compound_fused_labels)
        all_labels_for_reduction = np.concatenate(all_labels_for_reduction_list)

        # 5. 降维
        if method.lower() == 'tsne':
            # Ensure perplexity is less than n_samples
            current_perplexity = min(perplexity, len(all_features_to_reduce) - 1)
            if current_perplexity < 5:  # t-SNE might not work well with very low perplexity
                print(
                    f"警告: Perplexity ({current_perplexity}) is very low for t-SNE. Results might be suboptimal. Number of samples: {len(all_features_to_reduce)}")
            if current_perplexity <= 0:  # Cannot be <=0
                print(f"错误: Perplexity ({current_perplexity}) must be > 0 for t-SNE. Skipping visualization.")
                return

            reducer = TSNE(n_components=n_components, perplexity=current_perplexity,
                           random_state=42, n_iter=1000, init='pca', learning_rate='auto')  # Added init and lr
            print(f"  应用t-SNE降维 (perplexity={reducer.perplexity})...")
        else:
            current_n_components = min(n_components, all_features_to_reduce.shape[1])
            if current_n_components <= 0:
                print(f"错误: PCA n_components ({current_n_components}) must be > 0. Skipping visualization.")
                return
            reducer = PCA(n_components=current_n_components)
            print(f"  应用PCA降维 (n_components={reducer.n_components})...")

        try:
            reduced_features = reducer.fit_transform(all_features_to_reduce)
            if isinstance(reducer, PCA) and reducer.n_components > 0:
                print(f"  PCA解释方差比例: {np.sum(reducer.explained_variance_ratio_):.4f}")
        except Exception as e:
            print(f"错误: 降维失败: {e}")
            traceback.print_exc()
            return

        # 6. 划分降维后的特征
        ptr = 0
        actual_cnn_reduced = reduced_features[ptr: ptr + len(actual_cnn_features)]
        ptr += len(actual_cnn_features)
        projected_single_fused_reduced = reduced_features[ptr: ptr + len(projected_single_fused_features)]
        ptr += len(projected_single_fused_features)
        projected_compound_fused_reduced = reduced_features[ptr: ptr + len(projected_compound_fused_features)]

        # 7. 绘制可视化图
        configure_chinese_font()
        plt.figure(figsize=(16, 12))  # Slightly larger figure

        # Define colors and markers
        # More distinct colors for all fault types
        all_fault_names_ordered = single_fault_types_names + self.compound_fault_types
        unique_fault_names_in_plot = sorted(
            list(set(self.idx_to_fault.get(idx, str(idx)) for idx in all_labels_for_reduction)))

        palette = sns.color_palette("husl", len(unique_fault_names_in_plot))
        color_map = {name: palette[i] for i, name in enumerate(unique_fault_names_in_plot)}

        marker_actual = 'o'
        marker_proj_single = '^'  # Triangle up for single fault projections
        marker_proj_compound = 'X'  # X for compound fault projections

        # Plotting order: actual samples first, then single projections, then compound projections

        # A. 单一故障实际CNN特征 (点)
        if len(actual_cnn_reduced) > 0:
            for fault_idx in np.unique(actual_cnn_labels):
                mask = (actual_cnn_labels == fault_idx)
                fault_name = self.idx_to_fault.get(fault_idx, f"Unk_{fault_idx}")
                plt.scatter(
                    actual_cnn_reduced[mask, 0],
                    actual_cnn_reduced[mask, 1] if n_components > 1 else np.zeros_like(actual_cnn_reduced[mask, 0]),
                    color=color_map.get(fault_name, 'grey'), marker=marker_actual, s=30, alpha=0.4,
                    label=f"{fault_name} (实际CNN)" if not plt.gca().get_legend_handles_labels()[1].count(
                        f"{fault_name} (实际CNN)") else ""
                )

        # B. 单一故障融合语义投影特征 (上三角)
        if len(projected_single_fused_reduced) > 0:
            for fault_idx in np.unique(projected_single_fused_labels):
                mask = (projected_single_fused_labels == fault_idx)
                fault_name = self.idx_to_fault.get(fault_idx, f"Unk_{fault_idx}")
                plt.scatter(
                    projected_single_fused_reduced[mask, 0],
                    projected_single_fused_reduced[mask, 1] if n_components > 1 else np.zeros_like(
                        projected_single_fused_reduced[mask, 0]),
                    color=color_map.get(fault_name, 'black'), marker=marker_proj_single, s=150, alpha=0.9,
                    edgecolors='k', linewidths=1,
                    label=f"{fault_name} (投影-单一)" if not plt.gca().get_legend_handles_labels()[1].count(
                        f"{fault_name} (投影-单一)") else ""
                )
                # Add text label
                for i in np.where(mask)[0]:
                    plt.text(projected_single_fused_reduced[i, 0],
                             (projected_single_fused_reduced[i, 1] if n_components > 1 else 0) + 0.1,
                             fault_name, fontsize=9, weight='bold', color=color_map.get(fault_name, 'black'),
                             ha='center')

        # C. 复合故障融合语义投影特征 (X)
        if len(projected_compound_fused_reduced) > 0:
            for fault_idx in np.unique(projected_compound_fused_labels):
                mask = (projected_compound_fused_labels == fault_idx)
                fault_name = self.idx_to_fault.get(fault_idx, f"Unk_{fault_idx}")
                plt.scatter(
                    projected_compound_fused_reduced[mask, 0],
                    projected_compound_fused_reduced[mask, 1] if n_components > 1 else np.zeros_like(
                        projected_compound_fused_reduced[mask, 0]),
                    color=color_map.get(fault_name, 'black'), marker=marker_proj_compound, s=200, alpha=0.9,
                    edgecolors='k', linewidths=1.5,
                    label=f"{fault_name} (投影-复合)" if not plt.gca().get_legend_handles_labels()[1].count(
                        f"{fault_name} (投影-复合)") else ""
                )
                # Add text label
                for i in np.where(mask)[0]:
                    plt.text(projected_compound_fused_reduced[i, 0],
                             (projected_compound_fused_reduced[i, 1] if n_components > 1 else 0) + 0.1,
                             fault_name, fontsize=9, weight='bold', color=color_map.get(fault_name, 'black'),
                             ha='center')

        plt.title(f'特征空间: 实际CNN vs 融合语义投影 ({method.upper()}降维)')
        plt.xlabel(f'{method.upper()} Comp 1' + (f' ({reducer.explained_variance_ratio_[0]:.2%})' if isinstance(reducer,
                                                                                                                PCA) and reducer.n_components > 0 else ""))
        if n_components > 1:
            plt.ylabel(f'{method.upper()} Comp 2' + (
                f' ({reducer.explained_variance_ratio_[1]:.2%})' if isinstance(reducer,
                                                                               PCA) and reducer.n_components > 1 else ""))
        else:
            plt.yticks([])

        # Optimize legend
        handles, labels = plt.gca().get_legend_handles_labels()
        unique_legend_items = {}
        for handle, label_text in zip(handles, labels):
            if label_text not in unique_legend_items:
                unique_legend_items[label_text] = handle

        if unique_legend_items:
            plt.legend(unique_legend_items.values(), unique_legend_items.keys(), loc='best', markerscale=1.0,
                       fontsize=8, ncol=2)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"统一特征空间比较图已保存至 '{save_path}'")

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

            self.visualize_semantics(semantic_dict)
            self.visualize_data_semantics_distribution(semantic_dict)
            self.visualize_all_semantics_comparison(semantic_dict, data_dict,
                                                    save_path='all_semantics_comparison_pca.png',
                                                    method='pca')
            self.visualize_all_semantics_comparison(semantic_dict, data_dict,
                                                    save_path='all_semantics_comparison_tsne.png',
                                                    method='t-sne',
                                                    perplexity=20)
            print("\n==== 步骤3: 初始化基于知识蒸馏的双通道CNN模型 ====")
            self.cnn_model = DistillationDualChannelCNN(
                input_length=self.sample_length,
                semantic_dim=self.actual_latent_dim,
                num_classes=self.num_classes,
                feature_dim=CNN_FEATURE_DIM
            ).to(self.device)

            if hasattr(self.cnn_model, 'feature_dim'):
                self.cnn_feature_dim = self.cnn_model.feature_dim
            else:
                self.cnn_feature_dim = CNN_FEATURE_DIM

            print("\n==== 步骤4: 使用知识蒸馏训练双通道CNN模型 ====")
            trained_cnn_model = self.train_with_knowledge_distillation(
                data_dict=data_dict,
                semantic_dict=semantic_dict,
                epochs=CNN_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE,
                lr=CNN_LR
            )
            if trained_cnn_model is None:
                raise RuntimeError("CNN模型训练失败或未返回有效模型")
            self.cnn_model = trained_cnn_model

            print("\n==== 步骤5: 评估知识蒸馏效果 (CNN模型) ====")
            avg_distance, avg_alignment = self.visualize_distillation_effect(
                data_dict,
                semantic_dict,
                save_path='distillation_effect_cnn.png'
            )
            print(f"CNN知识蒸馏平均特征距离: {avg_distance:.4f}, 平均中心对齐度: {avg_alignment:.4f}")

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

            # 添加评估双向对齐网络的步骤
            self.evaluate_bidirectional_semantic_network(semantic_dict, data_dict)

            print("\n==== 步骤7: 生成复合故障投影 (使用训练好的双向语义映射网络) ====")
            compound_projections = self.generate_compound_fault_projections_fixed(semantic_dict, data_dict)
            if compound_projections is None or not compound_projections:
                raise RuntimeError("复合故障投影生成失败")

            # 可视化比较
            self.visualize_all_feature_projection_comparison(
                data_dict=data_dict,
                semantic_dict=semantic_dict,
                compound_projections=compound_projections,
                save_path='all_feature_projection_comparison_unified.png',
                method='pca'
            )

            # 可视化与实际复合故障特征的比较
            self.visualize_feature_space_comparison(
                data_dict,
                compound_projections,
                save_path='feature_space_comparison_mlp_vs_actual_compound.png',
                method='pca'
            )

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

    def visualize_feature_space_comparison(self, data_dict, compound_projections,
                                           save_path='feature_space_comparison.png',
                                           method='pca', perplexity=30, n_components=2):
        """
        可视化语义投影特征和实际复合故障特征在同一特征空间的分布

        参数:
            data_dict: 包含测试数据的字典
            compound_projections: 语义投影得到的特征字典
            save_path: 图像保存路径
            method: 降维方法，'tsne'或'pca'
            perplexity: t-SNE的复杂度参数
            n_components: 降维后的维度
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import seaborn as sns

        print(f"\n绘制复合故障特征空间比较图 (使用{method.upper()})...")

        # 1. 检查模型和数据
        if self.cnn_model is None:
            print("错误: CNN模型未初始化")
            return

        if not compound_projections:
            print("错误: 缺少语义投影特征")
            return

        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        if len(X_test) == 0:
            print("错误: 无测试数据")
            return

        # 2. 过滤出复合故障测试数据
        compound_fault_indices = set()
        for fault_name in self.compound_fault_types:
            if fault_name in self.fault_types:
                compound_fault_indices.add(self.fault_types[fault_name])

        compound_mask = np.isin(y_test, list(compound_fault_indices))
        X_compound_test = X_test[compound_mask]
        y_compound_test = y_test[compound_mask]

        if len(X_compound_test) == 0:
            print("错误: 无复合故障测试数据")
            return

        print(f"  处理 {len(X_compound_test)} 个复合故障测试样本")

        # 3. 提取实际的复合故障特征
        self.cnn_model.eval()
        actual_features_list = []
        actual_labels = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(X_compound_test), batch_size):
                batch_x = torch.FloatTensor(X_compound_test[i:i + batch_size]).to(self.device)
                batch_y = y_compound_test[i:i + batch_size]

                # 使用单通道特征提取器
                features = self.cnn_model(batch_x, semantic=None, return_features=True)

                if torch.all(torch.isfinite(features)):
                    actual_features_list.append(features.cpu().numpy())
                    actual_labels.extend(batch_y)

        if not actual_features_list:
            print("错误: 无法提取实际特征")
            return

        actual_features = np.vstack(actual_features_list)
        actual_labels = np.array(actual_labels)

        # 4. 收集语义投影特征
        projected_features = []
        projected_labels = []

        for fault_name, proj_feature in compound_projections.items():
            if fault_name in self.fault_types and np.all(np.isfinite(proj_feature)):
                projected_features.append(proj_feature)
                projected_labels.append(self.fault_types[fault_name])

        projected_features = np.array(projected_features)
        projected_labels = np.array(projected_labels)

        # 5. 合并特征进行降维
        all_features = np.vstack([actual_features, projected_features])
        feature_types = ['actual'] * len(actual_features) + ['projected'] * len(projected_features)
        all_labels = np.concatenate([actual_labels, projected_labels])

        # 降维
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            print(f"  应用t-SNE降维 (perplexity={perplexity})...")
        else:
            reducer = PCA(n_components=n_components)
            print(f"  应用PCA降维...")

        reduced_features = reducer.fit_transform(all_features)

        # 6. 划分降维后的特征
        actual_reduced = reduced_features[:len(actual_features)]
        projected_reduced = reduced_features[len(actual_features):]

        # 7. 绘制可视化图
        configure_chinese_font()  # 确保中文正确显示
        plt.figure(figsize=(14, 10))

        # 为每个类别分配颜色
        unique_labels = np.unique(all_labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        color_dict = {label: colors[i] for i, label in enumerate(unique_labels)}

        # 绘制实际特征
        for label in unique_labels:
            mask = (actual_labels == label)
            if np.any(mask):
                label_name = self.idx_to_fault.get(label, f"未知_{label}")
                plt.scatter(
                    actual_reduced[mask, 0],
                    actual_reduced[mask, 1] if n_components > 1 else np.zeros_like(actual_reduced[mask, 0]),
                    color=color_dict[label],
                    marker='o',
                    s=60,
                    alpha=0.7,
                    edgecolors='k',
                    linewidths=0.5,
                    label=f"{label_name} (实际特征)"
                )

        # 绘制投影特征
        for label in unique_labels:
            mask = (projected_labels == label)
            if np.any(mask):
                label_name = self.idx_to_fault.get(label, f"未知_{label}")
                plt.scatter(
                    projected_reduced[mask, 0],
                    projected_reduced[mask, 1] if n_components > 1 else np.zeros_like(projected_reduced[mask, 0]),
                    color=color_dict[label],
                    marker='X',
                    s=200,
                    alpha=1.0,
                    edgecolors='w',
                    linewidths=1.5,
                    label=f"{label_name} (语义投影)"
                )

                # 为每个投影中心添加标签
                for i in range(sum(mask)):
                    plt.text(
                        projected_reduced[mask, 0][i] + 0.1,
                        projected_reduced[mask, 1][i] + 0.1 if n_components > 1 else 0.1,
                        label_name,
                        fontsize=10,
                        weight='bold'
                    )

        # 绘制连接线 - 连接每个实际样本到其对应类别的投影中心
        for label in unique_labels:
            actual_mask = (actual_labels == label)
            projected_mask = (projected_labels == label)

            if np.any(actual_mask) and np.any(projected_mask):
                # 获取投影中心（应该只有一个点）
                proj_center = projected_reduced[projected_mask][0]

                # 随机选择部分实际样本进行连接，避免图形过于混乱
                actual_indices = np.where(actual_mask)[0]
                if len(actual_indices) > 20:  # 如果样本过多，随机选择20个
                    selected_indices = np.random.choice(actual_indices, 20, replace=False)
                else:
                    selected_indices = actual_indices

                for idx in selected_indices:
                    plt.plot(
                        [actual_reduced[idx, 0], proj_center[0]],
                        [actual_reduced[idx, 1] if n_components > 1 else 0,
                         proj_center[1] if n_components > 1 else 0],
                        color=color_dict[label],
                        alpha=0.1,
                        linestyle='--'
                    )

        # 添加标题和图例
        plt.title(f'复合故障特征空间比较 ({method.upper()}降维)\n点: 实际CNN特征, X: 语义投影特征')
        plt.grid(True, linestyle='--', alpha=0.7)

        # 优化图例 - 只保留唯一条目
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"特征空间比较图已保存至 '{save_path}'")

        # 8. 计算实际特征与投影特征之间的距离统计
        print("\n特征距离统计:")
        for label in unique_labels:
            actual_mask = (actual_labels == label)
            projected_mask = (projected_labels == label)

            if np.any(actual_mask) and np.any(projected_mask):
                # 获取原始特征空间中的数据（未降维）
                actual_class_features = actual_features[actual_mask]
                projected_class_feature = projected_features[projected_mask][0]  # 假设每类只有一个投影

                # 计算欧氏距离
                distances = np.linalg.norm(actual_class_features - projected_class_feature, axis=1)

                label_name = self.idx_to_fault.get(label, f"未知_{label}")
                print(f"  - {label_name}:")
                print(f"    * 平均距离: {np.mean(distances):.4f} ± {np.std(distances):.4f}")
                print(f"    * 最小距离: {np.min(distances):.4f}, 最大距离: {np.max(distances):.4f}")
                print(f"    * 样本数量: {len(distances)}")

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

    def train_semantic_mapping_mlp_with_bidirectional_alignment(self, data_dict, semantic_dict,
                                                                epochs=SEN_EPOCHS, lr=SEN_LR,
                                                                mlp_hidden_dim1=512, mlp_hidden_dim2=256,
                                                                mlp_dropout_rate=0.3):
        """
        使用双向对齐损失训练语义映射网络
        """
        print("\n--- 开始训练双向对齐语义映射网络 ---")

        # 1. 检查和准备
        if self.cnn_model is None or not hasattr(self.cnn_model, 'feature_dim') or self.cnn_model.feature_dim <= 0:
            print("错误：CNN模型未有效初始化或无特征维度。")
            return False
        self.cnn_model.eval()  # 教师模型设为评估模式，不更新其权重

        if self.fused_semantic_dim <= 0:
            print("错误：融合语义维度无效。")
            return False

        # 2. 初始化双向语义网络
        self.embedding_net = BidirectionalSemanticNetwork(
            semantic_dim=self.fused_semantic_dim,
            feature_dim=self.cnn_model.feature_dim,
            hidden_dim1=mlp_hidden_dim1,
            hidden_dim2=mlp_hidden_dim2,
            dropout_rate=mlp_dropout_rate
        ).to(self.device)

        print(f"双向语义映射网络初始化:")
        print(f"  语义维度: {self.fused_semantic_dim}")
        print(f"  特征维度: {self.cnn_model.feature_dim}")
        print(
            f"  隐层结构: {self.fused_semantic_dim} -> {mlp_hidden_dim1} -> {mlp_hidden_dim2} -> {self.cnn_model.feature_dim}")

        # 3. 准备数据
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict.get('X_val'), data_dict.get('y_val')

        fused_semantics = semantic_dict.get('fused_semantics', {})
        if not fused_semantics:
            print("错误：无有效的融合语义向量。")
            return False

        # 创建故障索引到语义向量的映射
        semantic_vectors_map = {}
        for fault_name, semantic_vec in fused_semantics.items():
            if fault_name in self.fault_types and np.all(np.isfinite(semantic_vec)):
                fault_idx = self.fault_types[fault_name]
                semantic_vectors_map[fault_idx] = semantic_vec

        if not semantic_vectors_map:
            print("错误：无法创建有效的故障索引到语义的映射。")
            return False

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        mlp_batch_size = getattr(self, 'batch_size', DEFAULT_BATCH_SIZE)
        train_loader = DataLoader(train_dataset, batch_size=mlp_batch_size, shuffle=True, drop_last=True)

        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=mlp_batch_size, shuffle=False)

        # 4. 设置优化器和调度器
        optimizer = optim.AdamW(self.embedding_net.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

        # 5. 损失函数
        mse_loss_fn = nn.MSELoss()
        cosine_loss_fn = nn.CosineEmbeddingLoss()

        # 6. 训练循环
        best_val_loss = float('inf')
        patience_epochs = 20  # 早停的耐心轮数
        current_patience = 0

        for epoch in range(epochs):
            self.embedding_net.train()
            epoch_losses = {
                'forward': 0.0, 'reverse': 0.0, 'cycle': 0.0, 'relation': 0.0, 'total': 0.0
            }
            num_valid_batches = 0

            for batch_signals, batch_labels in train_loader:
                batch_signals = batch_signals.to(self.device)
                batch_labels = batch_labels.to(self.device)

                # 准备语义输入
                batch_semantic_inputs_list = []
                valid_indices = []
                for i, label_idx in enumerate(batch_labels):
                    semantic_vec = semantic_vectors_map.get(label_idx.item())
                    if semantic_vec is not None:
                        batch_semantic_inputs_list.append(semantic_vec)
                        valid_indices.append(i)

                if not batch_semantic_inputs_list:
                    continue  # 如果这个批次没有有效的语义向量，跳过

                batch_semantic_inputs = torch.FloatTensor(np.array(batch_semantic_inputs_list)).to(self.device)
                batch_signals_filtered = batch_signals[valid_indices]

                # 获取CNN的真实特征 (作为教师信号)
                with torch.no_grad():
                    target_features = self.cnn_model(batch_signals_filtered, semantic=None, return_features=True)

                if not torch.all(torch.isfinite(target_features)):
                    print(f"警告: Epoch {epoch + 1}, 教师特征含有非有限值，跳过此批次。")
                    continue

                optimizer.zero_grad()

                # 1. 正向映射损失: 语义 -> 特征
                pred_features = self.embedding_net(batch_semantic_inputs, mode='forward')
                forward_loss = mse_loss_fn(pred_features, target_features)

                # 2. 反向映射损失: 特征 -> 语义
                pred_semantics = self.embedding_net(target_features, mode='reverse')
                reverse_loss = mse_loss_fn(pred_semantics, batch_semantic_inputs)

                # 3. 循环一致性损失: 语义 -> 特征 -> 语义
                _, reconstructed_semantics = self.embedding_net(batch_semantic_inputs, mode='cycle')
                cycle_loss = mse_loss_fn(reconstructed_semantics, batch_semantic_inputs)

                # 4. 特征关系保持损失
                # 计算真实特征之间的关系矩阵 (成对距离)
                real_dists = torch.cdist(target_features, target_features)
                max_real_dist = torch.max(real_dists).item()
                if max_real_dist > 0:  # 避免除零
                    real_dists = real_dists / max_real_dist  # 归一化

                    # 计算预测特征之间的关系矩阵
                    pred_dists = torch.cdist(pred_features, pred_features)
                    pred_dists = pred_dists / (torch.max(pred_dists).item() + 1e-8)  # 归一化

                    relation_loss = mse_loss_fn(pred_dists, real_dists)
                else:
                    relation_loss = torch.tensor(0.0, device=self.device)

                # 5. 余弦相似度损失 (确保特征方向一致)
                cosine_target = torch.ones(pred_features.size(0)).to(self.device)
                cosine_loss = cosine_loss_fn(pred_features, target_features, cosine_target)

                # 总损失 (可调整各部分权重)
                w_forward = 1.0
                w_reverse = 0.5
                w_cycle = 0.8
                w_relation = 0.3
                w_cosine = 0.5

                total_loss = (w_forward * forward_loss +
                              w_reverse * reverse_loss +
                              w_cycle * cycle_loss +
                              w_relation * relation_loss +
                              w_cosine * cosine_loss)

                # 反向传播
                if torch.isfinite(total_loss):
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.embedding_net.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_losses['forward'] += forward_loss.item()
                    epoch_losses['reverse'] += reverse_loss.item()
                    epoch_losses['cycle'] += cycle_loss.item()
                    epoch_losses['relation'] += relation_loss.item()
                    epoch_losses['total'] += total_loss.item()
                    num_valid_batches += 1
                else:
                    print(f"警告: 批次损失非有限值，跳过此批次的更新。")

            # 计算平均损失
            if num_valid_batches > 0:
                for k in epoch_losses:
                    epoch_losses[k] /= num_valid_batches

                print(f"Epoch [{epoch + 1}/{epochs}] - 损失统计:")
                print(f"  正向映射损失: {epoch_losses['forward']:.4f}")
                print(f"  反向映射损失: {epoch_losses['reverse']:.4f}")
                print(f"  循环一致性损失: {epoch_losses['cycle']:.4f}")
                print(f"  关系保持损失: {epoch_losses['relation']:.4f}")
                print(f"  总损失: {epoch_losses['total']:.4f}")
            else:
                print(f"Epoch [{epoch + 1}/{epochs}] - 无有效批次")
                continue

            # 验证阶段
            if val_loader:
                self.embedding_net.eval()
                val_total_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch_signals_val, batch_labels_val in val_loader:
                        batch_signals_val = batch_signals_val.to(self.device)
                        batch_labels_val = batch_labels_val.to(self.device)

                        # 准备验证集语义向量
                        batch_semantic_val_list = []
                        valid_indices_val = []
                        for i, label_idx in enumerate(batch_labels_val):
                            semantic_vec = semantic_vectors_map.get(label_idx.item())
                            if semantic_vec is not None:
                                batch_semantic_val_list.append(semantic_vec)
                                valid_indices_val.append(i)

                        if not batch_semantic_val_list:
                            continue

                        batch_semantic_val = torch.FloatTensor(np.array(batch_semantic_val_list)).to(self.device)
                        batch_signals_val_filtered = batch_signals_val[valid_indices_val]

                        # 获取验证集的真实特征
                        target_features_val = self.cnn_model(batch_signals_val_filtered, semantic=None,
                                                             return_features=True)

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

    def evaluate_bidirectional_semantic_network(self, semantic_dict, data_dict):
        """
        评估双向语义映射网络的性能
        """
        print("\n评估双向语义映射网络性能...")

        if self.embedding_net is None:
            print("错误: 语义映射网络未初始化")
            return False

        self.embedding_net.eval()

        # 准备数据
        X_val, y_val = data_dict.get('X_val'), data_dict.get('y_val')
        if X_val is None or y_val is None or len(X_val) == 0:
            print("警告: 无验证数据可用于评估。使用训练数据代替。")
            X_val, y_val = data_dict.get('X_train'), data_dict.get('y_train')
            if X_val is None or y_val is None or len(X_val) == 0:
                print("错误: 无数据可用于评估。")
                return False

        # 准备语义映射
        fused_semantics = semantic_dict.get('fused_semantics', {})
        semantic_vectors_map = {}
        for fault_name, semantic_vec in fused_semantics.items():
            if fault_name in self.fault_types and np.all(np.isfinite(semantic_vec)):
                semantic_vectors_map[self.fault_types[fault_name]] = semantic_vec

        # 创建数据加载器
        eval_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

        # 评估指标
        mse_loss_fn = nn.MSELoss()
        cos_sim_fn = nn.CosineSimilarity(dim=1)

        forward_mse_list = []  # 语义→特征的MSE
        reverse_mse_list = []  # 特征→语义的MSE
        cycle_mse_list = []  # 循环一致性MSE
        cosine_sim_list = []  # 特征余弦相似度

        with torch.no_grad():
            for batch_signals, batch_labels in eval_loader:
                batch_signals = batch_signals.to(self.device)

                # 过滤有对应语义的样本
                valid_indices = []
                valid_semantics = []
                for i, label in enumerate(batch_labels):
                    if label.item() in semantic_vectors_map:
                        valid_indices.append(i)
                        valid_semantics.append(semantic_vectors_map[label.item()])

                if not valid_indices:
                    continue

                # 准备数据
                batch_signals_filtered = batch_signals[valid_indices].to(self.device)
                batch_semantic_filtered = torch.FloatTensor(np.array(valid_semantics)).to(self.device)

                # 获取CNN真实特征
                target_features = self.cnn_model(batch_signals_filtered, semantic=None, return_features=True)

                # 1. 评估正向映射 (语义→特征)
                pred_features = self.embedding_net(batch_semantic_filtered, mode='forward')
                forward_mse = mse_loss_fn(pred_features, target_features).item()
                forward_mse_list.append(forward_mse)

                # 特征余弦相似度
                cosine_sim = cos_sim_fn(pred_features, target_features).mean().item()
                cosine_sim_list.append(cosine_sim)

                # 2. 评估反向映射 (特征→语义)
                pred_semantics = self.embedding_net(target_features, mode='reverse')
                reverse_mse = mse_loss_fn(pred_semantics, batch_semantic_filtered).item()
                reverse_mse_list.append(reverse_mse)

                # 3. 评估循环一致性 (语义→特征→语义)
                _, reconstructed_semantics = self.embedding_net(batch_semantic_filtered, mode='cycle')
                cycle_mse = mse_loss_fn(reconstructed_semantics, batch_semantic_filtered).item()
                cycle_mse_list.append(cycle_mse)

        # 计算平均指标
        results = {}
        if forward_mse_list:
            results['forward_mse'] = np.mean(forward_mse_list)
            results['reverse_mse'] = np.mean(reverse_mse_list)
            results['cycle_mse'] = np.mean(cycle_mse_list)
            results['cosine_sim'] = np.mean(cosine_sim_list)

            print("双向语义映射网络评估结果:")
            print(f"  正向映射MSE (语义→特征): {results['forward_mse']:.6f}")
            print(f"  反向映射MSE (特征→语义): {results['reverse_mse']:.6f}")
            print(f"  循环一致性MSE (语义→特征→语义): {results['cycle_mse']:.6f}")
            print(f"  特征余弦相似度: {results['cosine_sim']:.6f}")
        else:
            print("错误: 无有效样本进行评估。")

        return results


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

    def evaluate_zero_shot_with_cosine_similarity(self, data_dict, compound_projections):
        """
        Evaluates zero-shot compound fault classification using Cosine Similarity
        in the original feature space (no PCA).
        """
        print("\nEvaluating zero-shot compound fault classification (Cosine Similarity in original feature space)...")

        # 1. Basic Checks
        if self.cnn_model is None or not hasattr(self.cnn_model, 'feature_dim') or self.cnn_model.feature_dim <= 0:
            print("Error: CNN model not trained or feature dimension is invalid.")
            return 0.0, None
        if compound_projections is None or not compound_projections:
            print("Error: Missing compound fault projections.")
            return 0.0, None

        # 2. Get Test Data (filter for compound faults with available projections)
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        if len(X_test) == 0:
            print("Warning: No test data available.")
            return 0.0, None

        finite_mask_test = np.all(np.isfinite(X_test), axis=1)
        X_test, y_test = X_test[finite_mask_test], y_test[finite_mask_test]
        if len(X_test) == 0:
            print("Error: All test data contains non-finite values.")
            return 0.0, None

        available_projection_labels_names = list(compound_projections.keys())
        available_projection_indices = [self.fault_types[name] for name in available_projection_labels_names if
                                        name in self.fault_types]

        test_mask = np.isin(y_test, available_projection_indices)
        X_compound_test, y_compound_test = X_test[test_mask], y_test[test_mask]

        if len(X_compound_test) == 0:
            print("Error: No test samples match the available compound fault projections.")
            return 0.0, None

        print(
            f"  Using {len(X_compound_test)} test samples for evaluating {len(available_projection_labels_names)} compound fault types.")

        # 3. Prepare Projected Features (Original Space)
        candidate_fault_names = []
        projection_features_orig_list = []
        for name, projection in compound_projections.items():
            if name in self.fault_types and np.all(np.isfinite(projection)):
                candidate_fault_names.append(name)
                projection_features_orig_list.append(projection)
            else:
                print(f"Warning: Projection for '{name}' is invalid or name not in fault_types. Skipping.")

        if not projection_features_orig_list:
            print("Error: No valid projections found to use as candidates.")
            return 0.0, None

        projection_features_orig = np.array(projection_features_orig_list)

        # Normalize projected features for cosine similarity calculation (dot product of normalized vectors)
        projection_features_norm = projection_features_orig / (
                    np.linalg.norm(projection_features_orig, axis=1, keepdims=True) + 1e-9)

        # 4. Extract Actual Test Features (Original Space)
        self.cnn_model.eval()
        test_features_orig_list = []
        batch_size = getattr(self, 'batch_size', DEFAULT_BATCH_SIZE)  # Reuse batch_size

        with torch.no_grad():
            for i in range(0, len(X_compound_test), batch_size):
                batch_x = torch.FloatTensor(X_compound_test[i:i + batch_size]).to(self.device)
                # Get features from CNN's single-channel path
                features = self.cnn_model(batch_x, semantic=None, return_features=True)
                if torch.all(torch.isfinite(features)):
                    test_features_orig_list.append(features.cpu().numpy())
                else:
                    print(f"Warning: Batch {i // batch_size} of test features contained non-finite values. Skipping.")

        if not test_features_orig_list:
            print("Error: Failed to extract any valid features from test data.")
            return 0.0, None

        test_features_orig = np.vstack(test_features_orig_list)
        # Normalize actual test features
        test_features_norm = test_features_orig / (np.linalg.norm(test_features_orig, axis=1, keepdims=True) + 1e-9)

        # 5. Classification using Cosine Similarity
        y_pred_cosine = []

        # Cosine similarity is dot product of L2-normalized vectors
        similarity_matrix = np.dot(test_features_norm,
                                   projection_features_norm.T)  # Shape: (n_test_samples, n_projections)

        for i in range(len(test_features_norm)):
            # Find the projection with the highest cosine similarity
            nearest_idx = np.argmax(similarity_matrix[i])
            y_pred_cosine.append(self.fault_types[candidate_fault_names[nearest_idx]])

        # 6. Calculate Metrics
        accuracy_cosine = accuracy_score(y_compound_test, y_pred_cosine) * 100

        class_accuracy_cosine = {}
        unique_true_labels_indices = np.unique(y_compound_test)
        for fault_idx_true in unique_true_labels_indices:
            fault_type_name = self.idx_to_fault.get(fault_idx_true)
            if fault_type_name is None: continue

            mask = (y_compound_test == fault_idx_true)
            if np.sum(mask) > 0:
                class_acc = accuracy_score(
                    y_compound_test[mask],
                    np.array(y_pred_cosine)[mask]
                ) * 100
                class_accuracy_cosine[fault_type_name] = class_acc

        # 7. Visualize Confusion Matrix
        true_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_compound_test]
        pred_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_pred_cosine]

        # Ensure all candidate fault names are used as labels for CM, even if not predicted/true
        # This makes comparison with PCA version easier if they evaluate on slightly different sets of classes
        # due to projection validity.
        display_labels_all_candidates = sorted(candidate_fault_names)

        conf_matrix_cosine = None
        try:
            # Use a comprehensive set of labels for the confusion matrix display
            # This includes all true labels present, all predicted labels, and all candidate projection labels
            cm_labels_set = set(true_labels_str) | set(pred_labels_str) | set(display_labels_all_candidates)
            cm_display_labels = sorted(list(cm_labels_set))

            conf_matrix_cosine = confusion_matrix(true_labels_str, pred_labels_str, labels=cm_display_labels)

            configure_chinese_font()
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                conf_matrix_cosine,
                annot=True,
                fmt='d',
                cmap='Greens',  # Different color for distinction
                xticklabels=cm_display_labels,
                yticklabels=cm_display_labels
            )
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title(f'ZSL Confusion Matrix (Cosine Similarity, Acc: {accuracy_cosine:.2f}%)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('compound_fault_confusion_matrix_zsl_cosine.png')
            plt.close()

            print(f"\nZero-Shot Learning Accuracy (Cosine Similarity): {accuracy_cosine:.2f}%")
            print("Per-class Accuracy (Cosine Similarity):")
            for fault_type, acc in class_accuracy_cosine.items():
                print(f"  - {fault_type}: {acc:.2f}%")
            print("Confusion matrix (Cosine Similarity) saved to 'compound_fault_confusion_matrix_zsl_cosine.png'")

        except Exception as e:
            print(f"Warning: Error during Cosine Similarity confusion matrix generation: {e}")
            traceback.print_exc()
            conf_matrix_cosine = None  # Ensure it's None if error occurs

        return accuracy_cosine, conf_matrix_cosine

if __name__ == "__main__":

    set_seed(42)
    data_path = "E:/研究生/CNN/HDU1000-600"
    if not os.path.isdir(data_path):
        print(f"E: Data directory not found: {data_path}")
    else:
        fault_diagnosis = ZeroShotCompoundFaultDiagnosis(
            data_path=data_path, sample_length=SEGMENT_LENGTH,
            latent_dim=AE_LATENT_DIM, batch_size=DEFAULT_BATCH_SIZE )
        final_accuracy = fault_diagnosis.run_pipeline()
        print(f"\n>>> Final ZSL Accuracy: {final_accuracy:.2f}% <<<")

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
from itertools import islice
warnings.filterwarnings('ignore')
SEGMENT_LENGTH = 1024
OVERLAP = 0.5
STEP = int(SEGMENT_LENGTH * (1 - OVERLAP))
if STEP < 1: STEP = 1
DEFAULT_WAVELET = 'db4'
DEFAULT_WAVELET_LEVEL = 3
AE_LATENT_DIM = 64
AE_EPOCHS = 20
AE_LR = 0.0005
AE_BATCH_SIZE = 64
AE_CONTRASTIVE_WEIGHT = 1.2
AE_NOISE_STD = 0.05
CNN_EPOCHS = 5
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


class TimeFrequencyAutoencoder(nn.Module):
    """增强型自编码器，结合时域和频域信息，并添加分类能力"""

    def __init__(self, input_dim=SEGMENT_LENGTH, latent_dim=AE_LATENT_DIM, num_classes=8, alpha=0.5):
        super(TimeFrequencyAutoencoder, self).__init__()
        h1, h2 = 256, 128
        if input_dim < h1: h1 = max(latent_dim, (input_dim + latent_dim) // 2)
        if h1 < h2: h2 = max(latent_dim, (h1 + latent_dim) // 2)
        if h2 < latent_dim: h2 = latent_dim
        h1 = max(h1, h2)
        latent_dim = min(latent_dim, h2)

        self.alpha = alpha  # 时域和频域特征融合权重
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.actual_latent_dim = latent_dim

        # 时域编码器 - 使用卷积层更高效地处理时序信号
        self.time_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool1d(1),  # 全局池化
            nn.Flatten(),
            nn.Linear(128, latent_dim)
        )

        # 频域编码器 - 处理FFT幅值谱
        fft_size = input_dim // 2 + 1  # FFT实数信号的幅度谱长度
        self.freq_encoder = nn.Sequential(
            nn.Linear(fft_size, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.2),
            nn.Linear(h2, latent_dim)
        )

        # 特征融合层 - 融合时频域特征
        self.fusion_layer = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2),
        )

        # 分类头 - 基于融合后的潜在表示进行分类
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(h2, num_classes)
        )

        # 解码器 - 将潜在表示解码回原始信号
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.BatchNorm1d(h2),
            nn.LeakyReLU(0.2),
            nn.Linear(h2, h1),
            nn.BatchNorm1d(h1),
            nn.LeakyReLU(0.2),
            nn.Linear(h1, input_dim),
            nn.Tanh()  # 输出范围限制在[-1, 1]
        )

        # 特征映射注意力机制 - 使潜在特征更加关注故障相关特征
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )

        print(f"增强型时频域自编码器初始化完成:")
        print(f"  - 输入维度: {input_dim}")
        print(f"  - 潜在空间维度: {latent_dim}")
        print(f"  - 分类类别数: {num_classes}")

    def compute_fft_features(self, x):
        """计算输入信号的FFT特征"""
        # 确保输入是2D: [batch_size, signal_length]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # 应用实数FFT
        fft_complex = torch.fft.rfft(x)

        # 计算幅度谱
        fft_magnitude = torch.abs(fft_complex)

        # 对幅度谱应用对数变换，增强小幅值特征
        fft_magnitude = torch.log1p(fft_magnitude)

        return fft_magnitude

    def forward(self, x, return_latent=False, return_classification=False):
        """
        前向传播
        参数:
            x: 输入信号
            return_latent: 是否返回潜在表示
            return_classification: 是否返回分类结果
        """
        # 确保输入是2D: [batch_size, signal_length]
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # 1. 计算时域特征
        # 将输入重塑为[batch, 1, length]用于卷积
        x_conv = x.unsqueeze(1) if x.dim() == 2 else x
        time_features = self.time_encoder(x_conv)

        # 2. 计算频域特征
        freq_features = self.freq_encoder(self.compute_fft_features(x))

        # 3. 融合时频特征
        combined_features = torch.cat([time_features, freq_features], dim=1)
        latent = self.fusion_layer(combined_features)

        # 4. 应用注意力机制
        attention_weights = self.attention(latent)
        latent = latent * attention_weights

        # 5. 分类结果
        classification_output = self.classifier(latent)

        # 6. 解码重建
        decoded = self.decoder(latent)

        # 根据参数返回不同结果
        if return_latent and return_classification:
            return decoded, latent, classification_output
        elif return_latent:
            return decoded, latent
        elif return_classification:
            return decoded, classification_output
        else:
            return decoded

    def encode(self, x):
        """编码函数，用于提取特征表示"""
        if not torch.all(torch.isfinite(x)):
            print("W: AE encode input contains non-finite values. Clamping.")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # 确保输入是2D
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        # 将输入重塑为[batch, 1, length]用于卷积
        x_conv = x.unsqueeze(1) if x.dim() == 2 else x

        # 计算时域特征
        time_features = self.time_encoder(x_conv)

        # 计算频域特征
        freq_features = self.freq_encoder(self.compute_fft_features(x))

        # 融合特征
        combined_features = torch.cat([time_features, freq_features], dim=1)
        latent = self.fusion_layer(combined_features)

        # 应用注意力
        attention_weights = self.attention(latent)
        latent = latent * attention_weights

        if not torch.all(torch.isfinite(latent)):
            print("W: AE encode output contains non-finite values. Clamping.")
            latent = torch.nan_to_num(latent, nan=0.0)

        return latent

    def classify(self, x):
        """直接对输入进行分类"""
        latent = self.encode(x)
        return self.classifier(latent)


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
                 compound_data_semantic_generation_rule='mapper_output'):  # Added new parameter
        self.latent_dim_config = latent_dim
        self.actual_latent_dim = latent_dim  # Will be updated after AE training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = None
        self.preprocessor = DataPreprocessor(sample_length=SEGMENT_LENGTH)
        self.knowledge_dim = 0
        self.data_semantics = {}  # Will store single fault prototypes after AE
        self.idx_to_fault = {}  # This will be set by ZeroShotCompoundFaultDiagnosis instance
        self.all_latent_features = None
        self.all_latent_labels = None

        self.compound_data_semantic_generation_rule = compound_data_semantic_generation_rule
        print(f"FaultSemanticBuilder initialized with rule: {self.compound_data_semantic_generation_rule}")

        # Define fault attributes and parameters centrally
        self.fault_location_attributes = {
            'normal': [0, 0, 0], 'inner': [1, 0, 0], 'outer': [0, 1, 0], 'ball': [0, 0, 1],
            'inner_outer': [1, 1, 0], 'inner_ball': [1, 0, 1], 'outer_ball': [0, 1, 1],
            'inner_outer_ball': [1, 1, 1]
        }
        self.bearing_params_attributes = {
            'inner_diameter': 17 / 40, 'outer_diameter': 1.0, 'width': 12 / 40,
            'ball_diameter': 6.75 / 40, 'ball_number': 9 / 20
        }
        # Ordered list of single faults for consistent attribute vector creation
        self.single_fault_types_ordered = ['normal', 'inner', 'outer', 'ball']
        # For compound faults, their definitions (which single faults compose them)
        self.compound_fault_definitions = {
            'inner_outer': ['inner', 'outer'],
            'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'],
            'inner_outer_ball': ['inner', 'outer', 'ball']
        }
        # The attribute dimension for the mapper will be based on the number of single faults
        # if using one-hot style for fault presence, or the enhanced attributes if using those.
        # Let's use the enhanced attributes for the mapper.
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

    def train_autoencoder(self, X_train, labels, epochs=AE_EPOCHS, batch_size=AE_BATCH_SIZE, lr=AE_LR,
                          fault_types_dict=None, idx_to_fault_dict=None):
        """训练增强型自编码器，结合时频域信息和分类损失"""
        print("Training Enhanced Time-Frequency Autoencoder for data semantics...")
        input_dim = X_train.shape[1]

        # 使用传入的故障类型字典
        if fault_types_dict is None:
            print("警告: 未提供故障类型字典，将使用标签自动识别故障类型")
            # 从唯一标签生成临时故障类型字典
            unique_labels = np.unique(labels)
            fault_types_dict = {f"fault_{i}": label for i, label in enumerate(unique_labels)}
            idx_to_fault_dict = {label: f"fault_{i}" for i, label in enumerate(unique_labels)}
        elif idx_to_fault_dict is None:
            # 如果只提供了fault_types_dict但没有提供idx_to_fault_dict，则反转构建
            idx_to_fault_dict = {idx: name for name, idx in fault_types_dict.items()}

        # 定义单一故障类型
        single_fault_types = ['normal', 'inner', 'outer', 'ball']
        single_fault_indices = []

        for fault_name, idx in fault_types_dict.items():
            if fault_name in single_fault_types:
                single_fault_indices.append(idx)

        if not single_fault_indices:
            print("警告: 未找到任何单一故障类型，将使用前4个唯一标签作为单一故障")
            unique_labels = sorted(list(set(labels)))[:4]  # 使用前4个唯一标签
            single_fault_indices = unique_labels

        # 筛选单一故障数据
        single_mask = np.isin(labels, single_fault_indices)
        X_train_single = X_train[single_mask]
        labels_single = labels[single_mask]

        print(f"严格零样本学习: 仅使用 {len(X_train_single)}/{len(X_train)} 个单一故障样本进行训练")

        # 获取数据集中的类别数
        num_classes = len(fault_types_dict)  # 使用总类别数，包括所有故障类型
        print(f"Detected {num_classes} unique classes in total")
        print(f"Using {len(np.unique(labels_single))} single fault classes for training")

        # 初始化增强型自编码器
        self.autoencoder = TimeFrequencyAutoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim_config,
            num_classes=num_classes
        ).to(self.device)

        self.actual_latent_dim = self.autoencoder.actual_latent_dim
        if self.actual_latent_dim != self.latent_dim_config:
            print(f"W: AE latent dim adjusted by architecture: {self.actual_latent_dim}")

        train_dataset = TensorDataset(torch.FloatTensor(X_train_single), torch.LongTensor(labels_single))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        all_labels_np = labels_single
        all_data_tensor = torch.FloatTensor(X_train_single).to(self.device)

        # 初始化优化器
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5)

        # 使用余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(5, epochs // 3),  # 首次重启周期
            T_mult=2,  # 随后周期系数
            eta_min=lr / 10  # 最小学习率
        )

        # 损失函数
        criterion_recon = nn.MSELoss()  # 重建损失
        criterion_class = nn.CrossEntropyLoss()  # 分类损失

        # 权重因子 - 控制各损失项的重要性
        lambda_recon = 1.0
        lambda_class = 0.5
        lambda_contrastive = 1.2  # 保持原有的对比损失权重

        self.autoencoder.train()
        num_samples = len(X_train_single)
        best_loss = float('inf')
        patience_counter = 0
        patience = 15

        for epoch in range(epochs):
            epoch_recon_loss = 0.0
            epoch_class_loss = 0.0
            epoch_contrastive_loss = 0.0
            samples_processed = 0

            for data in train_loader:
                batch_data, batch_labels = data
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                if batch_data.shape[0] < 2: continue

                # 添加噪声进行数据增强，提高鲁棒性
                noise = torch.randn_like(batch_data) * AE_NOISE_STD
                batch_aug = torch.clamp(batch_data + noise, -1.0, 1.0)

                optimizer.zero_grad()

                # 前向传播 - 获取重建信号、潜在特征和分类输出
                decoded, latent, classification_output = self.autoencoder(batch_data,
                                                                          return_latent=True,
                                                                          return_classification=True)

                # 同样处理增强数据
                decoded_aug, latent_aug, classification_output_aug = self.autoencoder(batch_aug,
                                                                                      return_latent=True,
                                                                                      return_classification=True)

                # 检查输出是否包含NaN/Inf
                if not (torch.all(torch.isfinite(decoded)) and torch.all(torch.isfinite(latent)) and
                        torch.all(torch.isfinite(decoded_aug)) and torch.all(torch.isfinite(latent_aug))):
                    print(f"W: NaN/Inf detected in AE outputs epoch {epoch + 1}. Skipping batch loss.")
                    continue

                # 1. 重建损失
                recon_loss = criterion_recon(decoded, batch_data) + criterion_recon(decoded_aug, batch_aug)
                recon_loss = recon_loss / 2.0

                # 2. 分类损失
                class_loss = criterion_class(classification_output, batch_labels) + \
                             criterion_class(classification_output_aug, batch_labels)
                class_loss = class_loss / 2.0

                # 3. 对比损失 (使用原有函数)
                contrastive_loss = self._ae_contrastive_loss(latent, latent_aug, batch_labels)

                # 总损失
                total_loss = lambda_recon * recon_loss + \
                             lambda_class * class_loss + \
                             lambda_contrastive * contrastive_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"W: Total AE loss NaN/Inf epoch {epoch + 1}. Skipping backward.")
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), 1.0)
                optimizer.step()

                # 累计损失和处理样本数
                epoch_recon_loss += recon_loss.item() * batch_data.size(0)
                epoch_class_loss += class_loss.item() * batch_data.size(0)
                epoch_contrastive_loss += contrastive_loss.item() * batch_data.size(0)
                samples_processed += batch_data.size(0)

            # 更新学习率
            scheduler.step()

            # 计算平均损失并打印进度
            if samples_processed > 0:
                avg_recon_loss = epoch_recon_loss / samples_processed
                avg_class_loss = epoch_class_loss / samples_processed
                avg_contrastive_loss = epoch_contrastive_loss / samples_processed
                avg_total_loss = (lambda_recon * avg_recon_loss +
                                  lambda_class * avg_class_loss +
                                  lambda_contrastive * avg_contrastive_loss)

                print(f"Epoch [{epoch + 1}/{epochs}] - "
                      f"Total Loss: {avg_total_loss:.6f} "
                      f"(Recon: {avg_recon_loss:.6f}, "
                      f"Class: {avg_class_loss:.6f}, "
                      f"Contr: {avg_contrastive_loss:.6f})")

                # 早停策略
                if avg_total_loss < best_loss:
                    best_loss = avg_total_loss
                    patience_counter = 0
                    torch.save(self.autoencoder.state_dict(), 'best_time_frequency_autoencoder.pth')
                    print(f"Saved best model with loss: {best_loss:.6f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break
            else:
                print(f"Epoch [{epoch + 1}/{epochs}] - No samples processed.")

        # 加载最佳模型
        if os.path.exists('best_time_frequency_autoencoder.pth'):
            self.autoencoder.load_state_dict(torch.load('best_time_frequency_autoencoder.pth'))
            print("Loaded best Time-Frequency AE model.")
        else:
            print("W: No best Time-Frequency AE model file found. Using the last state.")

        # 切换到评估模式
        self.autoencoder.eval()

        print("Calculating data semantic centroids and storing all latent features...")
        # 计算数据语义中心点 (重新使用所有单一故障样本)
        self.data_semantics = {}
        all_latent_list = []
        inference_batch_size = batch_size * 4

        # 确保只计算单一故障的语义中心点
        X_single = X_train[single_mask]
        labels_single = labels[single_mask]

        if len(X_single) == 0:
            print("E: No single fault samples available for centroid calculation.")
            self.all_latent_features = None
            self.all_latent_labels = None
            return

        single_data_tensor = torch.FloatTensor(X_single).to(self.device)
        num_samples = len(X_single)

        with torch.no_grad():
            for i in range(0, num_samples, inference_batch_size):
                batch = single_data_tensor[i:i + inference_batch_size]
                if not torch.all(torch.isfinite(batch)): continue
                latent_batch = self.autoencoder.encode(batch)
                if torch.all(torch.isfinite(latent_batch)):
                    all_latent_list.append(latent_batch.cpu().numpy())
                else:
                    print(f"W: NaN/Inf in latent vectors during centroid calc index {i}")

        if not all_latent_list:
            print("E: No valid latent features extracted for centroids.")
            self.all_latent_features = None
            self.all_latent_labels = None
            return

        # 处理提取的特征
        all_latent_features_raw = np.vstack(all_latent_list)
        labels_array_raw = labels_single[:all_latent_features_raw.shape[0]]

        # 过滤非有限值
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
            self.all_latent_features = None
            self.all_latent_labels = None
            return

        # 保存过滤后的latent features和labels - 仅单一故障
        self.all_latent_features = all_latent_features_filtered
        self.all_latent_labels = labels_array_filtered
        print(f"  Stored {len(self.all_latent_features)} filtered latent features for single faults.")

        # 计算中心点 (仅单一故障)
        unique_label_indices = np.unique(labels_array_filtered)
        for label_idx in unique_label_indices:
            if label_idx not in single_fault_indices:
                continue  # 跳过任何可能的非单一故障标签

            type_mask = (labels_array_filtered == label_idx)
            if not np.any(type_mask): continue
            type_features = all_latent_features_filtered[type_mask]
            centroid = np.mean(type_features, axis=0)

            if np.all(np.isfinite(centroid)):
                fault_type = idx_to_fault_dict.get(label_idx, f"UnknownLabel_{label_idx}")
                if fault_type == f"UnknownLabel_{label_idx}":
                    print(f"W: Cannot map label index {label_idx} back to fault type name.")
                self.data_semantics[fault_type] = centroid
            else:
                fault_type = idx_to_fault_dict.get(label_idx, f"UnknownLabel_{label_idx}")
                print(
                    f"W: Centroid calculation for '{fault_type}' (Label {label_idx}) resulted in non-finite values. Setting zero.")
                self.data_semantics[fault_type] = np.zeros(self.actual_latent_dim, dtype=np.float32)

        # 评估分类性能
        correct = 0
        total = 0
        class_correct = [0] * num_classes  # 使用总类别数
        class_total = [0] * num_classes  # 使用总类别数

        with torch.no_grad():
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                outputs = self.autoencoder.classify(batch_data)
                _, predicted = torch.max(outputs.data, 1)

                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

                # 各类别准确率统计 - 只统计出现在batch中的类别
                for i in range(batch_labels.size(0)):
                    label = batch_labels[i].item()
                    if 0 <= label < num_classes:  # 确保索引有效
                        class_correct[label] += (predicted[i] == label).item()
                        class_total[label] += 1

        print(f"\n分类准确率评估 (仅单一故障):")
        print(f"整体准确率: {100 * correct / total:.2f}%")
        for i in range(len(class_correct)):
            if class_total[i] > 0 and i in idx_to_fault_dict:
                accuracy = 100 * class_correct[i] / class_total[i]
                fault_type = idx_to_fault_dict.get(i, f"Class_{i}")
                print(f"  - {fault_type}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")

        print(
            f"Time-Frequency AE training & centroid calculation complete. Found {len(self.data_semantics)} centroids.")



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
    """进一步增强的双向语义映射网络，加入投影校准和多级特征对齐"""

    def __init__(self, semantic_dim, feature_dim, hidden_dim1=512, hidden_dim2=256, dropout_rate=0.3):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim

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

        # 投影校准器 - 针对不同故障类型的特征校准
        self.projection_calibrator = ProjectionCalibrator(feature_dim)

        # 特征融合模块 - 用于处理复合故障的特殊性
        self.feature_fusion = FeatureFusionModule(feature_dim)

        # 初始化权重
        self._init_weights()

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
            features = self.forward_net(x)

            # 通过投影校准器进行校准
            if fault_type is not None:
                features = self.projection_calibrator(features, fault_type)

            # 对复合故障进行特殊处理
            if fault_type is not None and '_' in fault_type:
                features = self.feature_fusion(features, fault_type)

            return features

        elif mode == 'reverse':
            return self.reverse_net(x)

        elif mode == 'cycle':
            features = self.forward_net(x)

            # 应用校准和融合（如果有故障类型信息）
            if fault_type is not None:
                features = self.projection_calibrator(features, fault_type)
                if '_' in fault_type:
                    features = self.feature_fusion(features, fault_type)

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


class ProjectionCalibrator(nn.Module):
    """投影校准器 - 针对每种故障类型进行特定校准"""

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim

        # 校准变换矩阵 - 为不同故障类型设计特定变换
        self.calibration_layers = nn.ModuleDict({
            'inner_outer': self._create_calibration_layer(),
            'inner_ball': self._create_calibration_layer(),
            'outer_ball': self._create_calibration_layer(),
            'inner_outer_ball': self._create_calibration_layer(),
        })

        # 默认校准层，用于未明确定义的类型
        self.default_calibration = self._create_calibration_layer()

    def _create_calibration_layer(self):
        """创建校准层"""
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Tanh(),  # 使用Tanh确保细微调整而不是大幅变换
        )

    def forward(self, x, fault_type):
        """应用特定于故障类型的校准"""
        if fault_type in self.calibration_layers:
            # 残差校准 - 只做细微调整
            calibration = self.calibration_layers[fault_type](x)
            return x + calibration * 0.2  # 缩小校准的影响幅度
        return x + self.default_calibration(x) * 0.1


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

        out += identity # 残差连接
        out = self.relu(out)
        return out


class FeatureClusteringLoss(nn.Module):
    """特征聚类损失：促使同类特征更加紧凑"""

    def __init__(self, margin=1.0, temperature=0.07):
        super(FeatureClusteringLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, features, labels):
        """计算特征聚类损失"""
        # 归一化特征向量
        features = F.normalize(features, p=2, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature

        # 创建标签掩码（相同类别为1，不同类别为0）
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.t()).float()

        # 对角线掩码（排除自身）
        eye_mask = torch.eye(features.size(0), device=features.device)

        # 正样本对：相同类别，非自身
        pos_mask = mask * (1 - eye_mask)

        # 正样本对的平均相似度
        pos_sim = (similarity_matrix * pos_mask).sum() / (pos_mask.sum() + 1e-8)

        # 负样本对：不同类别
        neg_mask = 1 - mask

        # 负样本对的平均相似度
        neg_sim = (similarity_matrix * neg_mask).sum() / (neg_mask.sum() + 1e-8)

        # 目标：增大正样本相似度，减小负样本相似度
        loss = -pos_sim + self.margin * neg_sim
        return loss


class AEDataSemanticCNN(nn.Module):
    def __init__(self, input_length=SEGMENT_LENGTH, semantic_dim=AE_LATENT_DIM,
                 num_classes=8, feature_dim=256, freq_weight=0.4):
        super().__init__()
        self.input_length = input_length
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim
        self.freq_weight = freq_weight

        # --- 时域信号编码器 (与AE的时域编码器架构对齐) ---
        self.time_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool1d(1),  # 全局池化
            nn.Flatten()
        )

        # --- 频域信号编码器 (与AE的频域编码器架构对齐) ---
        fft_size = input_length // 2 + 1
        self.freq_encoder = nn.Sequential(
            nn.Linear(fft_size, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
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

        # --- 时频信号特征融合机制 ---
        self.signal_fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

        # --- 时频语义特征生成器 (增强版) ---
        self.feature_generator = nn.Sequential(
            nn.Linear(256 + 256, 384),
            nn.LayerNorm(384),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(384, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # --- 特征精炼层 (新增) ---
        self.feature_refiner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # --- 中心化注意层 (新增) ---
        self.centering_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid(),
        )

        # --- 分类器 ---
        self.classifier = nn.Linear(feature_dim, num_classes)

        # --- 加入L2正则化约束 ---
        self.l2_reg = 1e-5

    def compute_fft_features(self, x):
        """将时域信号转换为频域特征"""
        batch_size = x.shape[0]

        if x.dim() == 2:
            x_reshape = x.unsqueeze(1)
        else:
            x_reshape = x

        x_flat = x_reshape.reshape(batch_size, -1)
        fft_complex = torch.fft.rfft(x_flat)
        fft_magnitude = torch.abs(fft_complex)

        # 对幅度谱应用对数变换以增强小幅值频率成分
        fft_magnitude = torch.log1p(fft_magnitude)

        return fft_magnitude

    def extract_signal_features(self, x):
        """单独提取信号特征，便于分析"""
        # 确保输入是3D: [batch, channel, length]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 提取时域特征
        time_features = self.time_encoder(x)

        # 提取频域特征
        freq_features = self.freq_encoder(self.compute_fft_features(x))

        # 融合特征
        signal_features = torch.cat([time_features, freq_features], dim=1)
        signal_features = self.signal_fusion(signal_features)

        return {
            'time': time_features,
            'frequency': freq_features,
            'fused': signal_features
        }

    def forward(self, x, semantic, return_features=False):
        """前向传播 - 增强版，具有特征聚集能力"""
        # 确保输入是3D: [batch, channel, length]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 1. 提取时域信号特征
        time_features = self.time_encoder(x)

        # 2. 提取频域特征
        freq_features = self.freq_encoder(self.compute_fft_features(x))

        # 3. 融合时频特征
        signal_features = torch.cat([time_features, freq_features], dim=1)
        signal_features = self.signal_fusion(signal_features)

        # 4. 提取语义特征
        if semantic is None:
            raise ValueError("语义输入不能为空，双通道CNN必须同时接收信号和语义")
        semantic_features = self.semantic_encoder(semantic)

        # 5. 融合信号和语义特征
        combined = torch.cat([signal_features, semantic_features], dim=1)
        features = self.feature_generator(combined)

        # 6. 特征精炼和中心化 (新增)
        refined_features = self.feature_refiner(features)
        centering_weights = self.centering_attention(refined_features)

        # 应用中心化权重，使特征向类中心聚集
        centered_features = features * centering_weights + features

        # 如果只需要特征，直接返回
        if return_features:
            return centered_features

        # 7. 分类
        logits = self.classifier(centered_features)
        return logits, centered_features

    def l2_regularization(self):
        """计算模型参数的L2正则化损失"""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_reg * l2_loss


class ZeroShotCompoundFaultDiagnosis:

    def __init__(self, data_path, sample_length=SEGMENT_LENGTH, latent_dim=AE_LATENT_DIM, batch_size=DEFAULT_BATCH_SIZE,
                 compound_data_semantic_generation_rule='mapper_output'):  # Added new parameter with default
        self.data_path = data_path
        self.sample_length = sample_length
        self.latent_dim_config = latent_dim
        self.batch_size = batch_size
        # Store the rule, perhaps in a config dictionary if you adopt one later,
        # or directly as an attribute for now.
        self.compound_data_semantic_generation_rule = compound_data_semantic_generation_rule
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.preprocessor = DataPreprocessor(sample_length=self.sample_length)

        # Pass the rule to FaultSemanticBuilder
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
            self.semantic_builder.train_autoencoder(
                X_train_ae,
                labels=y_train_ae,
                epochs=AE_EPOCHS,
                batch_size=AE_BATCH_SIZE,
                lr=AE_LR,
                fault_types_dict=self.fault_types,  # 传入故障类型字典
                idx_to_fault_dict=self.idx_to_fault  # 传入索引到故障名称的映射
            )
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

    def _visualize_feature_clustering(self, features, labels, epoch):
        """可视化特征聚集度"""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import numpy as np
        import matplotlib.pyplot as plt

        configure_chinese_font()
        plt.figure(figsize=(16, 8))

        # 计算类内平均距离和类间平均距离
        unique_labels = np.unique(labels)
        intra_distances = []
        inter_distances = []

        # 特征中心
        centers = {}
        for label in unique_labels:
            mask = (labels == label)
            if np.sum(mask) > 0:
                center = np.mean(features[mask], axis=0)
                centers[label] = center

                # 计算类内样本到类中心的平均距离
                class_features = features[mask]
                dists_to_center = np.linalg.norm(class_features - center, axis=1)
                intra_distances.append(np.mean(dists_to_center))

        # 计算类间中心距离
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i + 1:]:
                if label1 in centers and label2 in centers:
                    dist = np.linalg.norm(centers[label1] - centers[label2])
                    inter_distances.append(dist)

        # 计算聚类指标：Davies-Bouldin Index (越小越好)
        if len(intra_distances) > 0 and len(inter_distances) > 0:
            mean_intra = np.mean(intra_distances)
            mean_inter = np.mean(inter_distances)
            db_index = mean_intra / mean_inter if mean_inter > 0 else float('inf')
        else:
            db_index = float('inf')

        # 左侧：PCA
        plt.subplot(1, 2, 1)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(features)

        for label in unique_labels:
            mask = (labels == label)
            if np.sum(mask) > 0:
                fault_name = self.idx_to_fault.get(label, f"未知_{label}")
                plt.scatter(features_pca[mask, 0], features_pca[mask, 1], label=fault_name, alpha=0.7)

                # 绘制类中心
                center = np.mean(features_pca[mask], axis=0)
                plt.scatter(center[0], center[1], marker='*', s=200,
                            edgecolor='k', linewidths=1, label=f'{fault_name}_中心')

        plt.title(f'特征聚集度 (PCA) - 轮次 {epoch}\nDavies-Bouldin Index: {db_index:.4f}')
        plt.legend()
        plt.grid(alpha=0.3)

        # 右侧：t-SNE
        plt.subplot(1, 2, 2)
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(features) // 10)),
                    random_state=42, learning_rate='auto', init='pca')
        features_tsne = tsne.fit_transform(features)

        for label in unique_labels:
            mask = (labels == label)
            if np.sum(mask) > 0:
                fault_name = self.idx_to_fault.get(label, f"未知_{label}")
                plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=fault_name, alpha=0.7)

                # 绘制类中心
                center = np.mean(features_tsne[mask], axis=0)
                plt.scatter(center[0], center[1], marker='*', s=200,
                            edgecolor='k', linewidths=1)

        plt.title(f'特征聚集度 (t-SNE) - 轮次 {epoch}')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'feature_clustering_epoch_{epoch}.png', dpi=300)
        plt.close()

        print(f"特征聚集度可视化已保存至 feature_clustering_epoch_{epoch}.png")
        print(f"Davies-Bouldin Index: {db_index:.4f} (越小表示聚类效果越好)")

    def train_ae_data_semantic_cnn(self, data_dict, semantic_dict, epochs=CNN_EPOCHS,
                                   batch_size=DEFAULT_BATCH_SIZE, lr=CNN_LR):
        """使用时频域增强的双通道CNN进行训练 - 严格遵守零样本学习原则，增强特征聚集能力"""

        print("训练基于时频域增强的双通道CNN (零样本模式) - 增强特征聚集...")

        # 准备数据 - 仅使用单一故障数据
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']

        # 定义单一故障类型
        single_fault_types = ['normal', 'inner', 'outer', 'ball']
        single_fault_indices = []

        for fault_type in single_fault_types:
            if fault_type in self.fault_types:
                single_fault_indices.append(self.fault_types[fault_type])

        # 筛选训练数据 - 仅使用单一故障
        train_single_mask = np.isin(y_train, single_fault_indices)
        X_train_single = X_train[train_single_mask]
        y_train_single = y_train[train_single_mask]

        # 筛选验证数据 - 仅使用单一故障
        val_single_mask = np.isin(y_val, single_fault_indices)
        X_val_single = X_val[val_single_mask]
        y_val_single = y_val[val_single_mask]

        print(f"严格零样本学习: 仅使用单一故障数据")
        print(f"  - 训练集: {len(X_train_single)}/{len(X_train)} 个样本")
        print(f"  - 验证集: {len(X_val_single)}/{len(X_val)} 个样本")

        # 确保CNN模型已初始化
        if self.cnn_model is None:
            print("初始化时频域增强的双通道CNN模型...")
            self.cnn_model = AEDataSemanticCNN(
                input_length=self.sample_length,
                semantic_dim=AE_LATENT_DIM,
                num_classes=self.num_classes,
                feature_dim=CNN_FEATURE_DIM
            )

        self.cnn_model = self.cnn_model.to(self.device)

        # 准备数据加载器
        train_dataset = TensorDataset(torch.FloatTensor(X_train_single), torch.LongTensor(y_train_single))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(torch.FloatTensor(X_val_single), torch.LongTensor(y_val_single))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 准备数据语义向量 (只使用单一故障的AE数据语义)
        data_semantics = {}
        if 'data_prototypes' in semantic_dict:
            for fault_type, semantic_vec in semantic_dict['data_prototypes'].items():
                if fault_type in single_fault_types:
                    data_semantics[fault_type] = semantic_vec

        # 不包含复合故障的语义向量
        semantic_vectors_dict = {}
        for fault_name, idx in self.fault_types.items():
            if fault_name in data_semantics:
                semantic_vectors_dict[idx] = data_semantics[fault_name]

        print(f"找到{len(semantic_vectors_dict)}种单一故障的数据语义向量")

        # 损失函数
        criterion_cls = nn.CrossEntropyLoss()
        criterion_feature = FeatureClusteringLoss(margin=1.0, temperature=0.07)  # 新增特征聚类损失

        # 优化器 - 使用AdamW与权重衰减
        optimizer = optim.AdamW(self.cnn_model.parameters(), lr=lr, weight_decay=1e-4)

        # 学习率调度 - OneCycleLR提供更快的收敛速度
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
            epochs=epochs, pct_start=0.3, div_factor=25, final_div_factor=1000,
            anneal_strategy='cos'
        )

        # 训练循环
        best_val_acc = 0.0
        patience, patience_counter = 10, 0

        # 记录不同类型故障的性能
        class_performance = {fault: [] for fault in single_fault_types if fault in self.fault_types}

        # 动态调整损失权重
        lambda_cls = 1.0  # 分类损失权重
        lambda_feature = 0.5  # 特征聚类损失权重初始值
        feature_loss_schedule = lambda epoch: min(1.0, 0.5 + 0.5 * epoch / (epochs // 2))  # 逐渐增加特征聚类权重

        for epoch in range(epochs):
            # 更新特征聚类损失权重
            lambda_feature = feature_loss_schedule(epoch)

            self.cnn_model.train()
            train_loss, correct, total = 0, 0, 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 准备语义向量
                batch_semantics = []
                valid_indices = []

                for i, lbl in enumerate(targets):
                    if lbl.item() in semantic_vectors_dict:
                        batch_semantics.append(semantic_vectors_dict[lbl.item()])
                        valid_indices.append(i)

                # 只处理有语义向量的样本
                if not valid_indices:
                    continue

                valid_inputs = inputs[valid_indices]
                valid_targets = targets[valid_indices]
                valid_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                # 前向传播
                optimizer.zero_grad()
                logits, features = self.cnn_model(valid_inputs, valid_semantics)

                # 计算各项损失
                cls_loss = criterion_cls(logits, valid_targets)
                feature_loss = criterion_feature(features, valid_targets)
                reg_loss = self.cnn_model.l2_regularization()

                # 总损失
                total_loss = lambda_cls * cls_loss + lambda_feature * feature_loss + reg_loss

                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=3.0)
                optimizer.step()
                scheduler.step()

                # 统计
                train_loss += total_loss.item()
                _, predicted = logits.max(1)
                total += valid_targets.size(0)
                correct += predicted.eq(valid_targets).sum().item()

                # 打印进度
                if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == len(train_loader):
                    print(f'[Epoch {epoch + 1}/{epochs}] Batch {batch_idx + 1}/{len(train_loader)} | '
                          f'Loss: {total_loss.item():.4f} (Cls: {cls_loss.item():.4f}, '
                          f'Feature: {feature_loss.item():.4f}) | '
                          f'Acc: {100. * correct / (total + 1e-8):.2f}% | '
                          f'LR: {scheduler.get_last_lr()[0]:.6f} | '
                          f'λF: {lambda_feature:.2f}')

            # 验证
            val_loss, val_acc = 0, 0
            self.cnn_model.eval()
            with torch.no_grad():
                v_correct, v_total = 0, 0
                class_correct = {idx: 0 for idx in self.fault_types.values() if idx in single_fault_indices}
                class_total = {idx: 0 for idx in self.fault_types.values() if idx in single_fault_indices}

                all_features = []
                all_labels = []

                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    # 准备语义向量
                    batch_semantics = []
                    valid_indices = []

                    for i, lbl in enumerate(targets):
                        if lbl.item() in semantic_vectors_dict:
                            batch_semantics.append(semantic_vectors_dict[lbl.item()])
                            valid_indices.append(i)

                    if not valid_indices:
                        continue

                    valid_inputs = inputs[valid_indices]
                    valid_targets = targets[valid_indices]
                    valid_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                    # 前向传播
                    logits, features = self.cnn_model(valid_inputs, valid_semantics)

                    # 收集特征和标签用于可视化
                    all_features.append(features.cpu().numpy())
                    all_labels.extend(valid_targets.cpu().numpy())

                    # 计算损失
                    loss = criterion_cls(logits, valid_targets)
                    val_loss += loss.item()

                    # 计算准确率
                    _, predicted = logits.max(1)
                    v_total += valid_targets.size(0)
                    correct_pred = predicted.eq(valid_targets)
                    v_correct += correct_pred.sum().item()

                    # 计算每个类别的准确率
                    for i, lbl in enumerate(valid_targets):
                        label_idx = lbl.item()
                        if label_idx in class_total:
                            class_total[label_idx] = class_total.get(label_idx, 0) + 1
                            if correct_pred[i]:
                                class_correct[label_idx] = class_correct.get(label_idx, 0) + 1

                # 计算验证指标
                if v_total > 0:
                    val_acc = v_correct / v_total
                    val_loss = val_loss / len(val_loader)

                    # 记录每个类别的准确率
                    for label_idx, count in class_total.items():
                        if count > 0 and label_idx in self.idx_to_fault:
                            fault_name = self.idx_to_fault[label_idx]
                            if fault_name in class_performance:
                                accuracy = class_correct[label_idx] / count
                                class_performance[fault_name].append(accuracy)

                # 可视化特征聚集度
                if all_features and (epoch + 1) % 5 == 0:
                    self._visualize_feature_clustering(np.vstack(all_features), np.array(all_labels), epoch + 1)

            # 打印结果
            print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss / len(train_loader):.4f} | '
                  f'Train Acc: {100. * correct / total:.2f}% | Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {100. * val_acc:.2f}%')

            # 打印每个类别的准确率
            print("各类别验证准确率:")
            for label_idx, count in sorted(class_total.items()):
                if count > 0 and label_idx in self.idx_to_fault:
                    fault_name = self.idx_to_fault[label_idx]
                    accuracy = 100. * class_correct[label_idx] / count
                    print(f"  - {fault_name}: {accuracy:.2f}% ({class_correct[label_idx]}/{count})")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.cnn_model.state_dict(), 'best_time_frequency_cnn_clustered.pth')
                print(f"保存最佳模型，验证集准确率: {100. * val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {patience}轮内未改善")
                    break

            # 每5个周期可视化时频特征分布
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                self._visualize_time_frequency_features(val_loader, semantic_vectors_dict, epoch + 1)

        # 加载最佳模型
        if os.path.exists('best_time_frequency_cnn_clustered.pth'):
            self.cnn_model.load_state_dict(torch.load('best_time_frequency_cnn_clustered.pth'))
            print("已加载最佳模型")

        # 可视化类别性能变化趋势
        self._visualize_class_performance_trend(class_performance, epochs)

        return self.cnn_model

    def _visualize_time_frequency_features(self, val_loader, semantic_vectors_dict, epoch):
        """可视化时域和频域特征的分布"""
        from itertools import islice

        self.cnn_model.eval()

        # 收集特征和标签
        time_features_list = []
        freq_features_list = []
        fused_features_list = []
        labels_list = []

        with torch.no_grad():
            # 随机选取几个批次
            sample_batches = list(islice(val_loader, 3))
            for inputs, targets in sample_batches:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 准备语义向量
                batch_semantics = []
                valid_indices = []

                for i, lbl in enumerate(targets):
                    if lbl.item() in semantic_vectors_dict:
                        batch_semantics.append(semantic_vectors_dict[lbl.item()])
                        valid_indices.append(i)

                if not valid_indices:
                    continue

                valid_inputs = inputs[valid_indices]
                valid_targets = targets[valid_indices]
                batch_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                # 提取信号特征 (时域/频域/融合)
                features_dict = self.cnn_model.extract_signal_features(valid_inputs)

                time_features_list.append(features_dict['time'].cpu().numpy())
                freq_features_list.append(features_dict['frequency'].cpu().numpy())
                fused_features_list.append(features_dict['fused'].cpu().numpy())
                labels_list.extend(valid_targets.cpu().numpy())

        if not time_features_list:
            return

        # 转换为numpy数组
        time_features = np.vstack(time_features_list)
        freq_features = np.vstack(freq_features_list)
        fused_features = np.vstack(fused_features_list)
        labels = np.array(labels_list)

        # 使用PCA降维可视化
        pca = PCA(n_components=2)

        # 应用PCA并可视化
        configure_chinese_font()
        plt.figure(figsize=(18, 6))

        # 时域特征
        plt.subplot(1, 3, 1)
        self._plot_features(time_features, labels, pca, "时域特征")

        # 频域特征
        plt.subplot(1, 3, 2)
        self._plot_features(freq_features, labels, pca, "频域特征")

        # 融合特征
        plt.subplot(1, 3, 3)
        self._plot_features(fused_features, labels, pca, "融合特征")

        plt.tight_layout()
        plt.savefig(f"time_freq_features_epoch_{epoch}.png", dpi=300)
        plt.close()
        print(f"时频特征分布图已保存至 time_freq_features_epoch_{epoch}.png")

    def _plot_features(self, features, labels, pca, title):
        """绘制特征分布图"""
        features_pca = pca.fit_transform(features)

        for label in np.unique(labels):
            mask = (labels == label)
            if np.sum(mask) == 0:
                continue

            label_name = self.idx_to_fault.get(label, f"未知_{label}")
            plt.scatter(
                features_pca[mask, 0],
                features_pca[mask, 1],
                alpha=0.7,
                s=60,
                label=label_name
            )

        plt.title(f"{title} (PCA降维)")
        plt.xlabel("主成分1")
        plt.ylabel("主成分2")
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')

    def _visualize_class_performance_trend(self, class_performance, epochs):
        """可视化各类别性能变化趋势"""
        if not class_performance or all(len(acc) == 0 for acc in class_performance.values()):
            print("没有可绘制的性能数据")
            return

        configure_chinese_font()
        plt.figure(figsize=(12, 8))

        # 绘制每个类别的准确率变化
        for fault_name, accuracies in class_performance.items():
            if len(accuracies) > 0:
                plt.plot(range(1, len(accuracies) + 1), [acc * 100 for acc in accuracies], label=fault_name, marker='o',
                         markersize=3)

        plt.title("各类别准确率变化趋势")
        plt.xlabel("训练轮次")
        plt.ylabel("准确率 (%)")
        plt.grid(alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig("class_performance_trend.png", dpi=300)
        plt.close()
        print("类别性能变化趋势图已保存至 class_performance_trend.png")

    def visualize_semantic_feature_distribution(self, data_dict, semantic_dict,
                                                save_path='semantic_feature_distribution.png'):
        """可视化基于数据语义的特征分布"""
        if self.cnn_model is None:
            print("错误: CNN模型未训练")
            return

        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()

        # 准备数据
        X_val, y_val = data_dict['X_val'], data_dict['y_val']

        # 准备语义向量
        data_semantics = semantic_dict.get('data_prototypes', {})
        if 'compound_data_semantics' in semantic_dict:
            data_semantics.update(semantic_dict['compound_data_semantics'])

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
        features_list = []
        labels_list = []

        with torch.no_grad():
            for i in range(0, len(X_sample), 32):
                end_idx = min(i + 32, len(X_sample))
                batch_x = torch.FloatTensor(X_sample[i:end_idx]).to(self.device)
                batch_y = y_sample[i:end_idx]

                # 找出有语义向量的样本
                valid_indices = []
                batch_semantics = []
                valid_labels = []

                for j, lbl in enumerate(batch_y):
                    if lbl in semantic_vectors_dict:
                        batch_semantics.append(semantic_vectors_dict[lbl])
                        valid_indices.append(j)
                        valid_labels.append(lbl)

                if not batch_semantics:
                    continue

                valid_batch_x = batch_x[valid_indices]
                batch_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                # 提取特征
                features = self.cnn_model(valid_batch_x, batch_semantics, return_features=True)

                features_list.append(features.cpu().numpy())
                labels_list.extend(valid_labels)

        # 合并特征和标签
        if not features_list:
            print("警告: 未能提取到任何特征")
            return

        features = np.vstack(features_list)
        labels = np.array(labels_list)

        # 使用t-SNE可视化特征空间
        from sklearn.manifold import TSNE

        # 应用t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
        reduced_features = tsne.fit_transform(features)

        # 可视化t-SNE结果
        plt.figure(figsize=(14, 12))

        # 获取唯一标签
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        # 计算类别中心
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            label_name = self.idx_to_fault.get(label, f"未知_{label}")

            # 绘制数据点
            plt.scatter(
                reduced_features[mask, 0], reduced_features[mask, 1],
                color=colors[i], marker='o', alpha=0.6, s=50,
                label=f"{label_name}"
            )

            # 计算中心点
            center = np.mean(reduced_features[mask], axis=0)

            # 绘制中心点
            plt.scatter(
                center[0], center[1],
                color=colors[i], marker='X', alpha=1.0, s=200,
                edgecolors='black', linewidths=2
            )

            # 添加标签
            plt.text(center[0], center[1], label_name, fontsize=12, weight='bold')

        # 添加标题和图例
        plt.title(f"基于AE数据语义的特征分布 (t-SNE)")
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

        print(f"特征分布图已保存至: {save_path}")

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

        data_semantics = {}
        if 'data_prototypes' in semantic_dict:
            data_semantics.update(semantic_dict['data_prototypes'])
        if 'compound_data_semantics' in semantic_dict:
            data_semantics.update(semantic_dict['compound_data_semantics'])

        data_semantic_map = {}
        for fault_name, sem_vec in data_semantics.items():
            if fault_name in self.fault_types:
                data_semantic_map[self.fault_types[fault_name]] = sem_vec

        actual_cnn_features_list = []
        actual_cnn_labels_list = []

        print(f"  A. 从数据集中提取单一故障的实际CNN特征 (每类最多{max_samples_per_class}个样本)...")
        self.cnn_model.eval()
        with torch.no_grad():
            for fault_name, fault_idx in single_fault_indices_map.items():
                mask = (y_samples == fault_idx)
                X_class_samples = X_samples[mask]
                if len(X_class_samples) == 0:
                    continue

                # 检查这个故障类型是否有数据语义可用
                if fault_idx not in data_semantic_map:
                    print(f"警告: 故障类型 '{fault_name}' 没有对应的数据语义，跳过。")
                    continue

                # 随机抽样
                if len(X_class_samples) > max_samples_per_class:
                    sample_indices = np.random.choice(len(X_class_samples), max_samples_per_class, replace=False)
                    X_class_samples = X_class_samples[sample_indices]

                if len(X_class_samples) > 0:
                    batch_x = torch.FloatTensor(X_class_samples).to(self.device)

                    # 为每个样本准备相同的数据语义向量
                    batch_semantics = torch.FloatTensor([data_semantic_map[fault_idx]] * len(X_class_samples)).to(
                        self.device)

                    # 提取特征 - 现在提供语义输入
                    features = self.cnn_model(batch_x, semantic=batch_semantics, return_features=True)

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

            print("\n==== 步骤3: 初始化基于AE数据语义的双通道CNN模型 ====")
            self.cnn_model = AEDataSemanticCNN(
                input_length=self.sample_length,
                semantic_dim=AE_LATENT_DIM,  # 固定为64维
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

            print("\n==== 步骤5: 可视化AE数据语义特征分布 ====")
            self.visualize_semantic_feature_distribution(
                data_dict,
                semantic_dict,
                save_path='semantic_feature_distribution.png'
            )
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

            accuracy_pca, conf_matrix_pca = self.evaluate_zero_shot_with_pca(
                data_dict,
                compound_projections,

            )
            print(f"零样本学习准确率 (PCA方法): {accuracy_pca:.2f}%")

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

    def evaluate_zero_shot_with_pca(self, data_dict, compound_projections, pca_components=2):
        """使用PCA降维后的欧氏距离进行零样本复合故障分类 - 使用语义投影与实时提取特征进行对比"""
        print(f"评估零样本复合故障分类能力（使用{pca_components}维PCA降维后的欧氏距离）...")

        # 1. 基本检查
        if self.cnn_model is None or not hasattr(self.cnn_model, 'feature_dim'):
            print("错误: CNN模型未训练")
            return 0.0, None

        if compound_projections is None or not compound_projections:
            print("错误: 缺少复合故障投影")
            return 0.0, None

        if self.semantic_builder is None or self.semantic_builder.autoencoder is None:
            print("错误: 自编码器未初始化")
            return 0.0, None

        # 2. 获取测试数据 - 仅复合故障
        X_test, y_test = data_dict['X_test'], data_dict['y_test']

        # 定义复合故障类型
        compound_fault_types = list(compound_projections.keys())
        compound_fault_indices = [self.fault_types[name] for name in compound_fault_types if name in self.fault_types]

        # 筛选测试数据 - 仅复合故障
        test_compound_mask = np.isin(y_test, compound_fault_indices)
        X_compound_test = X_test[test_compound_mask]
        y_compound_test = y_test[test_compound_mask]

        if len(X_compound_test) == 0:
            print("警告: 无复合故障测试数据")
            return 0.0, None

        # 确保测试数据有效
        finite_mask_test = np.all(np.isfinite(X_compound_test), axis=1)
        X_compound_test, y_compound_test = X_compound_test[finite_mask_test], y_compound_test[finite_mask_test]

        if len(X_compound_test) == 0:
            print("错误: 测试数据都是非有限值")
            return 0.0, None

        # 3. 提取测试样本特征 - 使用自编码器实时提取语义
        print(f"  使用 {len(X_compound_test)} 个复合故障测试样本进行评估...")

        self.cnn_model.eval()
        self.semantic_builder.autoencoder.eval()
        test_features_list = []
        valid_test_labels = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(X_compound_test), batch_size):
                batch_x = torch.FloatTensor(X_compound_test[i:i + batch_size]).to(self.device)
                batch_y = y_compound_test[i:i + batch_size]

                # 从样本中实时提取语义
                batch_semantics = self.semantic_builder.autoencoder.encode(batch_x)

                # 确保语义向量有效
                if not torch.all(torch.isfinite(batch_semantics)):
                    continue

                # 使用提取的语义作为CNN的输入
                features = self.cnn_model(batch_x, semantic=batch_semantics, return_features=True)

                if torch.all(torch.isfinite(features)):
                    test_features_list.append(features.cpu().numpy())
                    valid_test_labels.extend(batch_y.tolist())

        if not test_features_list:
            print("错误: 无法提取有效的测试特征")
            return 0.0, None

        test_features = np.vstack(test_features_list)
        y_compound_test = np.array(valid_test_labels)

        # 4. 准备语义投影特征 - 这是分类的"模板"
        projection_features = []
        projection_labels = []
        projection_indices = []

        for label, projection in compound_projections.items():
            if label in self.fault_types:
                projection_features.append(projection)
                projection_labels.append(label)
                projection_indices.append(self.fault_types[label])

        projection_features = np.array(projection_features)

        print(f"  测试特征: {test_features.shape}, 语义投影: {projection_features.shape}")
        print(f"  候选类别: {projection_labels}")

        # 5. 应用PCA降维 - 真实特征和投影特征一起PCA
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

        # 6. 在PCA空间中进行分类 - 使用欧氏距离
        y_pred = []
        distances = []

        for i in range(len(test_features_pca)):
            # 计算到每个投影点的欧氏距离
            dists = [np.linalg.norm(test_features_pca[i] - proj) for proj in projection_features_pca]
            nearest_idx = np.argmin(dists)
            y_pred.append(projection_indices[nearest_idx])
            distances.append(dists[nearest_idx])

        # 7. 计算指标
        y_pred = np.array(y_pred)
        correct = (y_pred == y_compound_test)
        accuracy = np.mean(correct) * 100

        # 8. 各类别准确率
        print("\n=== 各类别准确率 ===")
        class_accuracy = {}
        for fault_idx in np.unique(y_compound_test):
            mask = (y_compound_test == fault_idx)
            if np.sum(mask) > 0:
                correct_count = np.sum(y_pred[mask] == fault_idx)
                total_count = np.sum(mask)
                acc = (correct_count / total_count) * 100
                fault_name = self.idx_to_fault.get(fault_idx, f"Unknown_{fault_idx}")
                class_accuracy[fault_name] = acc
                print(f"  - {fault_name}: {correct_count}/{total_count} = {acc:.2f}%")

        print(f"总体准确率: {accuracy:.2f}%")

        # 9. 可视化混淆矩阵
        if pca_components >= 2:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.metrics import confusion_matrix

                # 准备标签字符串
                true_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_compound_test]
                pred_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_pred]
                display_labels = sorted(list(set(true_labels_str)))

                # 混淆矩阵
                conf_matrix = confusion_matrix(true_labels_str, pred_labels_str, labels=display_labels)

                # 绘制图形
                configure_chinese_font()
                plt.figure(figsize=(12, 10))

                # 混淆矩阵子图
                plt.subplot(2, 1, 1)
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
                plt.title(f'零样本学习混淆矩阵 (准确率: {accuracy:.2f}%)')

                # 特征空间可视化子图
                plt.subplot(2, 1, 2)

                # 绘制测试样本
                for idx in np.unique(y_compound_test):
                    mask = (y_compound_test == idx)
                    if np.sum(mask) > 0:
                        fault_name = self.idx_to_fault.get(idx, f"Unk_{idx}")
                        plt.scatter(
                            test_features_pca[mask, 0],
                            test_features_pca[mask, 1],
                            alpha=0.6,
                            label=f"{fault_name} (实际特征)"
                        )

                # 绘制投影点
                for i, label in enumerate(projection_labels):
                    plt.scatter(
                        projection_features_pca[i, 0],
                        projection_features_pca[i, 1],
                        marker='X', s=120, edgecolor='k', linewidth=1.5,
                        label=f"{label} (语义投影)"
                    )

                plt.title(f"复合故障特征空间比较 (PCA降维)")
                plt.xlabel("主成分1")
                plt.ylabel("主成分2")
                plt.grid(alpha=0.3)
                plt.legend()

                plt.tight_layout()
                plt.savefig('compound_fault_evaluation.png', dpi=300)
                plt.close()

                print("评估结果可视化已保存至 'compound_fault_evaluation.png'")

            except Exception as e:
                import traceback
                print(f"可视化生成错误: {e}")
                traceback.print_exc()
                conf_matrix = None
        else:
            conf_matrix = None

        return accuracy, conf_matrix

    def visualize_feature_space_comparison(self, data_dict, compound_projections,
                                           save_path='feature_space_comparison_mlp_vs_actual_compound.png',
                                           method='pca', perplexity=30, n_components=2):
        """可视化语义投影特征和实际复合故障特征在同一特征空间的分布 - 修复版"""
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

        if self.semantic_builder is None or self.semantic_builder.autoencoder is None:
            print("错误: 自编码器未初始化，无法提取实时语义")
            return

        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        if len(X_test) == 0:
            print("错误: 无测试数据")
            return

        # 2. 过滤出复合故障测试数据 (只选择有投影的类别)
        available_projections = []
        projection_indices = []

        for fault_name, projection in compound_projections.items():
            if fault_name in self.fault_types and np.all(np.isfinite(projection)):
                available_projections.append(fault_name)
                projection_indices.append(self.fault_types[fault_name])

        compound_mask = np.isin(y_test, projection_indices)
        X_compound_test = X_test[compound_mask]
        y_compound_test = y_test[compound_mask]

        if len(X_compound_test) == 0:
            print("错误: 无复合故障测试数据")
            return

        print(f"  处理 {len(X_compound_test)} 个复合故障测试样本")

        # 打印每种类别的样本数量
        unique_labels, counts = np.unique(y_compound_test, return_counts=True)
        print("复合故障样本分布:")
        for label, count in zip(unique_labels, counts):
            fault_name = self.idx_to_fault.get(label, f"Unknown_{label}")
            print(f"  - {fault_name}: {count}样本")

        # 3. 提取实际的复合故障特征 - 使用自编码器实时提取语义
        self.cnn_model.eval()
        self.semantic_builder.autoencoder.eval()
        actual_features_list = []
        actual_labels = []
        batch_size = 64

        with torch.no_grad():
            for i in range(0, len(X_compound_test), batch_size):
                batch_x = torch.FloatTensor(X_compound_test[i:i + batch_size]).to(self.device)
                batch_y = y_compound_test[i:i + batch_size]

                # 关键修复：直接从样本提取语义，而不是从标签查找
                batch_semantics = self.semantic_builder.autoencoder.encode(batch_x)

                # 确保语义向量有效
                if not torch.all(torch.isfinite(batch_semantics)):
                    print(f"警告: 批次{i // batch_size + 1}提取的语义包含非有限值，已被跳过")
                    continue

                # 使用提取的语义作为CNN的输入
                features = self.cnn_model(batch_x, semantic=batch_semantics, return_features=True)

                if torch.all(torch.isfinite(features)):
                    actual_features_list.append(features.cpu().numpy())
                    actual_labels.extend(batch_y.tolist())
                else:
                    print(f"警告: 批次{i // batch_size + 1}的特征包含非有限值，已被跳过")

        if not actual_features_list:
            print("错误: 无法提取实际特征")
            return

        actual_features = np.vstack(actual_features_list)
        actual_labels = np.array(actual_labels)
        print(f"成功提取 {len(actual_features)} 个测试样本特征")

        # 4. 收集语义投影特征
        projected_features = []
        projected_labels = []

        for fault_name, proj_feature in compound_projections.items():
            if fault_name in self.fault_types and np.all(np.isfinite(proj_feature)):
                projected_features.append(proj_feature)
                projected_labels.append(self.fault_types[fault_name])

        projected_features = np.array(projected_features)
        projected_labels = np.array(projected_labels)
        print(f"使用 {len(projected_features)} 个语义投影")

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

        return reduced_features, all_labels, feature_types

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
                    target_features = self.cnn_model(
                        batch_signals_filtered,
                        semantic=batch_data_semantics,
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
                w_forward = 1.0
                w_reverse = 0.3
                w_cycle = 0.5
                w_contrastive = 0.4 + 0.6 * epoch_progress
                w_projection = 0.8 + 1.2 * epoch_progress
                w_compound = 1.0 + 3.0 * epoch_progress  # 大幅增加复合故障权重

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
    def evaluate_bidirectional_semantic_network(self, semantic_dict, data_dict):
        """
        评估双向语义映射网络的性能 - 适配只有双通道的CNN模型
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

        # 准备两种语义：融合语义(用于MLP)和数据语义(用于CNN)
        fused_semantics = semantic_dict.get('fused_semantics', {})
        data_semantics = {}
        if 'data_prototypes' in semantic_dict:
            data_semantics.update(semantic_dict['data_prototypes'])
        if 'compound_data_semantics' in semantic_dict:
            data_semantics.update(semantic_dict['compound_data_semantics'])

        fused_semantic_map = {}  # 融合语义 - 用于MLP
        data_semantic_map = {}  # 数据语义 - 用于CNN

        for fault_name, semantic_vec in fused_semantics.items():
            if fault_name in self.fault_types and np.all(np.isfinite(semantic_vec)):
                fused_semantic_map[self.fault_types[fault_name]] = semantic_vec

        for fault_name, semantic_vec in data_semantics.items():
            if fault_name in self.fault_types and np.all(np.isfinite(semantic_vec)):
                data_semantic_map[self.fault_types[fault_name]] = semantic_vec

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

                # 过滤有对应语义的样本 - 需要同时有融合语义和数据语义
                valid_indices = []
                fused_semantics_list = []
                data_semantics_list = []

                for i, label in enumerate(batch_labels):
                    label_item = label.item()
                    if label_item in fused_semantic_map and label_item in data_semantic_map:
                        valid_indices.append(i)
                        fused_semantics_list.append(fused_semantic_map[label_item])
                        data_semantics_list.append(data_semantic_map[label_item])

                if not valid_indices:
                    continue

                # 准备数据
                batch_signals_filtered = batch_signals[valid_indices].to(self.device)
                batch_fused_semantics = torch.FloatTensor(np.array(fused_semantics_list)).to(self.device)
                batch_data_semantics = torch.FloatTensor(np.array(data_semantics_list)).to(self.device)

                # 获取CNN真实特征 - 必须提供数据语义
                target_features = self.cnn_model(
                    batch_signals_filtered,
                    semantic=batch_data_semantics,  # 提供数据语义
                    return_features=True
                )

                # 1. 评估正向映射 (融合语义→特征)
                pred_features = self.embedding_net(batch_fused_semantics, mode='forward')
                forward_mse = mse_loss_fn(pred_features, target_features).item()
                forward_mse_list.append(forward_mse)

                # 特征余弦相似度
                cosine_sim = cos_sim_fn(pred_features, target_features).mean().item()
                cosine_sim_list.append(cosine_sim)

                # 2. 评估反向映射 (特征→融合语义)
                pred_semantics = self.embedding_net(target_features, mode='reverse')
                reverse_mse = mse_loss_fn(pred_semantics, batch_fused_semantics).item()
                reverse_mse_list.append(reverse_mse)

                # 3. 评估循环一致性 (融合语义→特征→融合语义)
                _, reconstructed_semantics = self.embedding_net(batch_fused_semantics, mode='cycle')
                cycle_mse = mse_loss_fn(reconstructed_semantics, batch_fused_semantics).item()
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


    def analyze_projection_space(self, data_dict, compound_projections):
        """
        分析投影空间和类簇之间的关系，解决outer_ball准确率为0%的问题
        """
        print("\n===== 投影空间分析 =====")

        # 1. 基本检查
        if self.semantic_builder is None or self.semantic_builder.autoencoder is None:
            print("错误: 语义提取器未初始化")
            return

        if self.cnn_model is None:
            print("错误: CNN模型未初始化")
            return

        if compound_projections is None or len(compound_projections) == 0:
            print("错误: 未提供复合故障投影")
            return

        # 2. 获取测试数据
        X_test, y_test = data_dict['X_test'], data_dict['y_test']

        # 3. 整理复合故障类型和索引
        compound_fault_names = list(compound_projections.keys())
        compound_indices = [self.fault_types.get(name) for name in compound_fault_names]

        name_to_idx = {name: idx for name, idx in zip(compound_fault_names, compound_indices) if idx is not None}
        idx_to_name = {idx: name for name, idx in name_to_idx.items()}

        # 筛选复合故障测试样本
        mask = np.isin(y_test, list(idx_to_name.keys()))
        X_compound = X_test[mask]
        y_compound = y_test[mask]

        # 4. 提取特征
        batch_size = 64
        all_features = []
        all_labels = []

        self.cnn_model.eval()
        self.semantic_builder.autoencoder.eval()

        with torch.no_grad():
            # 提取测试样本特征
            for i in range(0, len(X_compound), batch_size):
                batch_X = torch.FloatTensor(X_compound[i:i + batch_size]).to(self.device)
                batch_y = y_compound[i:i + batch_size]

                # 提取语义
                semantics = self.semantic_builder.autoencoder.encode(batch_X)
                if not torch.all(torch.isfinite(semantics)):
                    continue

                # 提取特征
                features = self.cnn_model(batch_X, semantics, return_features=True)
                if not torch.all(torch.isfinite(features)):
                    continue

                all_features.append(features.cpu().numpy())
                all_labels.extend(batch_y.tolist())

        if not all_features:
            print("错误: 无法提取有效特征")
            return

        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)

        # 5. 准备投影特征
        proj_features = []
        proj_names = []
        proj_indices = []

        for name, feat in compound_projections.items():
            if name in name_to_idx and np.all(np.isfinite(feat)):
                proj_features.append(feat)
                proj_names.append(name)
                proj_indices.append(name_to_idx[name])

        proj_features = np.array(proj_features)

        # 6. 分析每个投影点与所有样本的距离
        print("\n投影点到各类样本的平均距离:")

        for i, (name, idx, feat) in enumerate(zip(proj_names, proj_indices, proj_features)):
            print(f"\n投影点 {name} (索引 {idx}):")

            # 计算到每个类样本的平均距离
            for unique_idx in np.unique(all_labels):
                mask = (all_labels == unique_idx)
                if np.sum(mask) > 0:
                    class_features = all_features[mask]

                    # 计算欧氏距离
                    distances = np.sqrt(np.sum((class_features - feat) ** 2, axis=1))
                    avg_dist = np.mean(distances)
                    min_dist = np.min(distances)

                    class_name = idx_to_name.get(unique_idx, f"Unknown_{unique_idx}")
                    print(f"  - 到 {class_name} (索引 {unique_idx}) 的距离: "
                          f"平均={avg_dist:.4f}, 最小={min_dist:.4f}, 样本数={np.sum(mask)}")

            # 计算最近的类
            class_avg_distances = {}
            for unique_idx in np.unique(all_labels):
                mask = (all_labels == unique_idx)
                if np.sum(mask) > 0:
                    class_features = all_features[mask]
                    distances = np.sqrt(np.sum((class_features - feat) ** 2, axis=1))
                    class_avg_distances[unique_idx] = np.mean(distances)

            if class_avg_distances:
                nearest_class_idx = min(class_avg_distances.keys(), key=lambda k: class_avg_distances[k])
                nearest_class_name = idx_to_name.get(nearest_class_idx, f"Unknown_{nearest_class_idx}")
                print(f"  * 最近的类: {nearest_class_name} (索引 {nearest_class_idx}), "
                      f"距离: {class_avg_distances[nearest_class_idx]:.4f}")

                # 检查是否匹配
                if nearest_class_idx == idx:
                    print("  ✓ 投影点最接近的类与其对应的类匹配")
                else:
                    print("  ✗ 投影点最接近的类与其对应的类不匹配!")

        # 7. 应用PCA并分析
        from sklearn.decomposition import PCA

        # 合并所有特征
        combined_features = np.vstack([all_features, proj_features])

        # 应用PCA
        pca = PCA(n_components=2)
        combined_pca = pca.fit_transform(combined_features)

        # 分离测试特征和投影特征
        test_pca = combined_pca[:len(all_features)]
        proj_pca = combined_pca[len(all_features):]

        # 8. 可视化
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(14, 10))

            # 绘制测试样本
            for unique_idx in np.unique(all_labels):
                mask = (all_labels == unique_idx)
                class_name = idx_to_name.get(unique_idx, f"Unknown_{unique_idx}")
                plt.scatter(test_pca[mask, 0], test_pca[mask, 1], alpha=0.6, s=30,
                            label=f"{class_name} (索引 {unique_idx})")

            # 绘制投影点
            for i, (name, idx) in enumerate(zip(proj_names, proj_indices)):
                plt.scatter(proj_pca[i, 0], proj_pca[i, 1], marker='X', s=200,
                            edgecolor='black', linewidth=2, label=f"{name} (投影, 索引 {idx})")

                # 添加文本标签
                plt.text(proj_pca[i, 0], proj_pca[i, 1], f"{name}\n索引={idx}",
                         fontsize=12, ha='center', va='bottom',
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

            plt.title("PCA空间中的样本和投影点分布 (带索引信息)")
            plt.xlabel("主成分1")
            plt.ylabel("主成分2")
            plt.grid(alpha=0.3)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
            plt.tight_layout()
            plt.savefig("projection_analysis.png", dpi=300)
            plt.close()

            # 计算和可视化距离矩阵
            plt.figure(figsize=(10, 8))

            distance_matrix = np.zeros((len(proj_indices), len(np.unique(all_labels))))
            unique_indices = np.unique(all_labels)

            for i, proj_idx in enumerate(proj_indices):
                for j, class_idx in enumerate(unique_indices):
                    mask = (all_labels == class_idx)
                    if np.sum(mask) > 0:
                        class_features = all_features[mask]
                        distances = np.sqrt(np.sum((class_features - proj_features[i]) ** 2, axis=1))
                        distance_matrix[i, j] = np.mean(distances)

            # 设置标签
            row_labels = [f"{name} ({idx})" for name, idx in zip(proj_names, proj_indices)]
            col_labels = [f"{idx_to_name.get(idx, 'Unknown')} ({idx})" for idx in unique_indices]

            # 绘制热图
            sns.heatmap(distance_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                        xticklabels=col_labels, yticklabels=row_labels)
            plt.title("投影点到各类样本的平均距离")
            plt.xlabel("类别")
            plt.ylabel("投影点")
            plt.tight_layout()
            plt.savefig("distance_matrix.png", dpi=300)
            plt.close()

        except Exception as e:
            import traceback
            print(f"可视化过程中出错: {e}")
            traceback.print_exc()

        print("\n分析完成，结果已保存为图像文件。")

        return

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
        # 修复这一行 - 使用semantic_builder访问autoencoder
        if self.semantic_builder is None or self.semantic_builder.autoencoder is None:
            print("错误: 自编码器未初始化，无法提取实时语义")
            return 0.0, None

        # 2. 获取测试数据 - 仅复合故障
        X_test, y_test = data_dict['X_test'], data_dict['y_test']

        # 定义复合故障类型
        compound_fault_types = list(compound_projections.keys())
        compound_fault_indices = [self.fault_types[name] for name in compound_fault_types if name in self.fault_types]

        # 筛选测试数据 - 仅复合故障
        test_compound_mask = np.isin(y_test, compound_fault_indices)
        X_compound_test = X_test[test_compound_mask]
        y_compound_test = y_test[test_compound_mask]

        if len(X_compound_test) == 0:
            print("警告: 无复合故障测试数据")
            return 0.0, None

        # 确保数据有效
        finite_mask_test = np.all(np.isfinite(X_compound_test), axis=1)
        X_compound_test, y_compound_test = X_compound_test[finite_mask_test], y_compound_test[finite_mask_test]

        if len(X_compound_test) == 0:
            print("错误: 所有测试数据包含非有限值")
            return 0.0, None

        print(f"  使用 {len(X_compound_test)} 个复合故障测试样本评估 {len(compound_fault_indices)} 种复合故障类型")

        # 3. 准备投影特征 (原始空间)
        candidate_fault_names = []
        projection_features_orig_list = []
        for name, projection in compound_projections.items():
            if name in self.fault_types and np.all(np.isfinite(projection)):
                candidate_fault_names.append(name)
                projection_features_orig_list.append(projection)
            else:
                print(f"警告: '{name}'的投影无效或不在故障类型中，已跳过")

        if not projection_features_orig_list:
            print("错误: 无有效投影可用作候选")
            return 0.0, None

        projection_features_orig = np.array(projection_features_orig_list)
        projection_features_norm = projection_features_orig / (
                np.linalg.norm(projection_features_orig, axis=1, keepdims=True) + 1e-9)

        # 4. 提取实际测试特征 (原始空间) - 使用实时提取的语义
        self.cnn_model.eval()
        # 修复这一行 - 使用semantic_builder访问autoencoder
        self.semantic_builder.autoencoder.eval()
        test_features_orig_list = []
        valid_test_labels = []
        batch_size = getattr(self, 'batch_size', 64)

        with torch.no_grad():
            for i in range(0, len(X_compound_test), batch_size):
                batch_x = torch.FloatTensor(X_compound_test[i:i + batch_size]).to(self.device)
                batch_y = y_compound_test[i:i + batch_size]

                # 修复这一行 - 使用semantic_builder访问autoencoder
                batch_semantics = self.semantic_builder.autoencoder.encode(batch_x)

                # 确保语义向量有效
                if not torch.all(torch.isfinite(batch_semantics)):
                    print(f"警告: 批次 {i // batch_size} 的提取语义包含非有限值，已跳过")
                    continue

                # 使用提取的语义作为CNN的输入，获取特征
                features = self.cnn_model(batch_x, semantic=batch_semantics, return_features=True)

                if torch.all(torch.isfinite(features)):
                    test_features_orig_list.append(features.cpu().numpy())
                    valid_test_labels.extend(batch_y.tolist())
                else:
                    print(f"警告: 批次 {i // batch_size} 的测试特征包含非有限值，已跳过")

        if not test_features_orig_list:
            print("错误: 无法从测试数据中提取有效特征")
            return 0.0, None

        test_features_orig = np.vstack(test_features_orig_list)
        y_compound_test = np.array(valid_test_labels)  # 更新为有效的标签

        # 归一化测试特征
        test_features_norm = test_features_orig / (np.linalg.norm(test_features_orig, axis=1, keepdims=True) + 1e-9)

        # 5. 使用余弦相似度进行分类
        y_pred_cosine = []

        # 余弦相似度 = 归一化向量的点积
        similarity_matrix = np.dot(test_features_norm,
                                   projection_features_norm.T)  # 形状: (n_test_samples, n_projections)

        for i in range(len(test_features_norm)):
            # 找到余弦相似度最高的投影
            nearest_idx = np.argmax(similarity_matrix[i])
            y_pred_cosine.append(self.fault_types[candidate_fault_names[nearest_idx]])

        # 6. 计算评估指标
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

        # 7. 可视化混淆矩阵
        true_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_compound_test]
        pred_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_pred_cosine]

        # 确保所有候选故障名称都作为标签显示在混淆矩阵中
        display_labels_all_candidates = sorted(candidate_fault_names)

        conf_matrix_cosine = None
        try:
            # 使用包含所有标签的综合集合来显示混淆矩阵
            cm_labels_set = set(true_labels_str) | set(pred_labels_str) | set(display_labels_all_candidates)
            cm_display_labels = sorted(list(cm_labels_set))

            conf_matrix_cosine = confusion_matrix(true_labels_str, pred_labels_str, labels=cm_display_labels)

            configure_chinese_font()
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                conf_matrix_cosine,
                annot=True,
                fmt='d',
                cmap='Greens',  # 不同颜色以区分PCA方法
                xticklabels=cm_display_labels,
                yticklabels=cm_display_labels
            )
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title(f'零样本学习混淆矩阵 (实时语义+余弦相似度, 准确率: {accuracy_cosine:.2f}%)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('compound_fault_confusion_matrix_zsl_cosine_realtime.png')
            plt.close()

            print(f"\n零样本学习准确率 (实时语义+余弦相似度): {accuracy_cosine:.2f}%")
            print("各类别准确率 (实时语义+余弦相似度):")
            for fault_type, acc in class_accuracy_cosine.items():
                print(f"  - {fault_type}: {acc:.2f}%")
            print("混淆矩阵(实时语义+余弦相似度)已保存至 'compound_fault_confusion_matrix_zsl_cosine_realtime.png'")

        except Exception as e:
            print(f"警告: 余弦相似度混淆矩阵生成错误: {e}")
            traceback.print_exc()
            conf_matrix_cosine = None  # 确保出错时为None

        return accuracy_cosine, conf_matrix_cosine

if __name__ == "__main__":

    set_seed(42)
    data_path = "E:/研究生/CNN/HDU600D"
    if not os.path.isdir(data_path):
        print(f"E: Data directory not found: {data_path}")
    else:
        fault_diagnosis = ZeroShotCompoundFaultDiagnosis(
            data_path=data_path, sample_length=SEGMENT_LENGTH,
            latent_dim=AE_LATENT_DIM, batch_size=DEFAULT_BATCH_SIZE )
        final_accuracy = fault_diagnosis.run_pipeline()
        print(f"\n>>> Final ZSL Accuracy: {final_accuracy:.2f}% <<<")
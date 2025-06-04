import numpy as np
import scipy.io as sio
from scipy.interpolate import interp1d
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix, accuracy_score

import os
import warnings
import random
import time
from collections import Counter
import traceback
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

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
SAE_EPOCHS = 50
SAE_LR = 0.0005
SAE_MU = 1.0
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


class FrequencyAttributeExtractor:
    """频域属性提取器"""

    def __init__(self, fs=12000, lf_cutoff=1000, hf_cutoff=3000):
        self.fs = fs
        self.lf_cutoff = lf_cutoff
        self.hf_cutoff = hf_cutoff
        self.epsilon = 1e-8

    def extract_frequency_ratios(self, signal):
        """提取HF_ratio和LF_ratio"""
        # 确保信号是一维的
        if signal.ndim > 1:
            signal = signal.flatten()

        # FFT计算
        N_fft = len(signal)
        if N_fft < 2:
            return 0.0, 0.0

        fft_result = np.fft.fft(signal, N_fft)
        power_spectrum = np.abs(fft_result[:N_fft // 2]) ** 2 / N_fft

        # 频率轴
        freqs = np.fft.fftfreq(N_fft, 1 / self.fs)[:N_fft // 2]

        # 频带能量计算
        lf_mask = (freqs >= 0) & (freqs <= self.lf_cutoff)
        hf_mask = (freqs >= self.hf_cutoff) & (freqs <= self.fs / 2)

        E_lf = np.sum(power_spectrum[lf_mask])
        E_hf = np.sum(power_spectrum[hf_mask])
        E_total = np.sum(power_spectrum) + self.epsilon

        # 计算比值
        lf_ratio = (E_lf + self.epsilon) / E_total
        hf_ratio = (E_hf + self.epsilon) / E_total

        return hf_ratio, lf_ratio


class PseudoCompoundGenerator:
    """伪复合故障生成器"""

    def __init__(self, alpha_range=(0.3, 0.7)):
        self.alpha_range = alpha_range
        self.random_seed = 42
        np.random.seed(self.random_seed)

    def generate_compound_signals(self, signals_dict, num_samples_per_combination=50):
        """生成伪复合故障信号"""
        compound_signals = {}
        compound_definitions = {
            'inner_outer': ['inner', 'outer'],
            'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'],
            'inner_outer_ball': ['inner', 'outer', 'ball']
        }

        for compound_name, components in compound_definitions.items():
            print(f"生成 {compound_name} 伪复合信号...")
            compound_list = []

            for _ in range(num_samples_per_combination):
                if len(components) == 2:
                    # 双组件混合
                    comp1, comp2 = components
                    if comp1 in signals_dict and comp2 in signals_dict:
                        sig1 = signals_dict[comp1][np.random.randint(len(signals_dict[comp1]))]
                        sig2 = signals_dict[comp2][np.random.randint(len(signals_dict[comp2]))]
                        alpha = np.random.uniform(*self.alpha_range)
                        compound_sig = alpha * sig1 + (1 - alpha) * sig2
                        compound_list.append(compound_sig)

                elif len(components) == 3:
                    # 三组件混合
                    comp1, comp2, comp3 = components
                    if all(comp in signals_dict for comp in components):
                        sig1 = signals_dict[comp1][np.random.randint(len(signals_dict[comp1]))]
                        sig2 = signals_dict[comp2][np.random.randint(len(signals_dict[comp2]))]
                        sig3 = signals_dict[comp3][np.random.randint(len(signals_dict[comp3]))]

                        # 三元权重分配
                        weights = np.random.dirichlet([1, 1, 1])
                        compound_sig = weights[0] * sig1 + weights[1] * sig2 + weights[2] * sig3
                        compound_list.append(compound_sig)

            if compound_list:
                compound_signals[compound_name] = np.array(compound_list)
                print(f"  生成了 {len(compound_list)} 个 {compound_name} 样本")

        return compound_signals
class ContrastiveAutoencoder(nn.Module):
    """
    1D 卷积编码器 + 全连接解码器的自编码器，
    重构输入长度，并在 latent 空间应用有监督对比损失。
    """

    def __init__(self, input_length: int = 1024, latent_dim: int = 64):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim

        # --- Encoder: 3x Conv1d ↓ length by /2 each time ---
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),  # L -> L/2
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),

            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),  # L/2 -> L/4
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # L/4 -> L/8
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.AdaptiveAvgPool1d(1),  # -> [B,128,1]
            nn.Flatten(),  # -> [B,128]
            nn.Linear(128, latent_dim)  # -> [B, latent_dim]
        )

        # --- Decoder: 全连接直接重构到原始长度 ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, input_length),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L] or [B,1,L] -> recon [B, L]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # -> [B,1,L]
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """返回 [B, latent_dim]"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.encoder(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从 [B, latent_dim] 重构到 [B, input_length]
        """
        recon = self.decoder(z)
        return recon  # [B, L]


class SupervisedContrastiveLoss(nn.Module):
    """
    有监督对比损失，将同类拉近、异类推远。
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.size(0)

        # L2 归一化
        features = F.normalize(features, p=2, dim=1)
        sim = torch.mm(features, features.T) / self.temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask - torch.eye(batch_size, device=device)

        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * mask).sum(dim=1)
        total_sum = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim))

        loss = -torch.log((pos_sum + 1e-9) / (total_sum + 1e-9))
        return loss.mean()


class AETrainer:
    """
    训练 ContrastiveAutoencoder，用 recon_loss + contrastive_loss。
    """

    def __init__(self, model: ContrastiveAutoencoder, device: torch.device = None):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.recon_loss_fn = nn.MSELoss()
        self.contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.1)

    def train(self,
              X: torch.Tensor,
              y: torch.Tensor,
              epochs: int = 30,
              batch_size: int = 64,
              lr: float = 1e-3,
              contrastive_weight: float = 1.0):
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             drop_last=True)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        best_metric = float('inf')
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_recon, total_contrast, n = 0.0, 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                recon = self.model(xb)  # [B, L]
                loss_recon = self.recon_loss_fn(recon, xb)
                z = self.model.encode(xb)  # [B, latent_dim]
                loss_contrast = self.contrastive_loss_fn(z, yb)
                loss = loss_recon + contrastive_weight * loss_contrast
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_recon += loss_recon.item()
                total_contrast += loss_contrast.item()
                n += 1

            avg_recon = total_recon / n if n > 0 else 0
            avg_contrast = total_contrast / n if n > 0 else 0
            metric = avg_recon + contrastive_weight * avg_contrast
            print(f"[AE Epoch {epoch}/{epochs}] Recon: {avg_recon:.6f}, Contrast: {avg_contrast:.6f}")

            # 保存最优
            if metric < best_metric:
                best_metric = metric
                torch.save(self.model.state_dict(), "ae_best.pth")

        # 加载最优模型
        if os.path.exists("ae_best.pth"):
            self.model.load_state_dict(torch.load("ae_best.pth", map_location=self.device))
        self.model.eval()
        print("AE training complete, best model loaded.")
        return self.model


class EnhancedAttributeSemanticMapper(nn.Module):
    def __init__(self, attr_dim=5, semantic_dim=64, hidden_dims=[128, 256, 128]):
        super().__init__()

        # 位置编码分支 (3维)
        self.position_branch = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 频域特征分支 (2维)
        self.frequency_branch = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 融合网络
        fusion_input_dim = 128  # 64 + 64
        layers = []
        layers.append(nn.Linear(fusion_input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(0.2))

        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.2))

        layers.append(nn.Linear(hidden_dims[-1], semantic_dim))

        self.fusion_mapper = nn.Sequential(*layers)

    def forward(self, x):
        # 分离输入
        position_features = x[:, :3]  # [is_inner, is_outer, is_ball]
        frequency_features = x[:, 3:5]  # [HF_ratio, LF_ratio]

        # 分支处理
        pos_encoded = self.position_branch(position_features)
        freq_encoded = self.frequency_branch(frequency_features)

        # 特征融合
        fused_features = torch.cat([pos_encoded, freq_encoded], dim=1)

        # 最终映射
        semantic_output = self.fusion_mapper(fused_features)

        return semantic_output


class FaultSemanticBuilder:
    def __init__(self, latent_dim=AE_LATENT_DIM, hidden_dim=128,  # hidden_dim seems unused here
                 compound_data_semantic_generation_rule='mapper_output'):
        self.latent_dim_config = latent_dim
        self.compound_fusion_mlp = None
        self.actual_latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = None
        self.preprocessor = DataPreprocessor(sample_length=SEGMENT_LENGTH)
        self.knowledge_dim = 3
        self.freq_extractor = FrequencyAttributeExtractor()
        self.pseudo_generator = PseudoCompoundGenerator()
        self.enhanced_attribute_dim = 5
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
        self.fault_types = {
            'normal': 0, 'inner': 1, 'outer': 2, 'ball': 3,
            'inner_outer': 4, 'inner_ball': 5, 'outer_ball': 6, 'inner_outer_ball': 7
        }
        # 用于存储频域特征统计
        self.frequency_stats = {}


        self.single_fault_types_ordered = ['normal', 'inner', 'outer', 'ball']

        self.compound_fault_definitions = {
            'inner_outer': ['inner', 'outer'],
            'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'],
            'inner_outer_ball': ['inner', 'outer', 'ball']
        }

        self.enhanced_attribute_dim = 5
        self.freq_extractor = FrequencyAttributeExtractor()
        self.pseudo_generator = PseudoCompoundGenerator()

        # 用于存储原始训练信号
        self.original_training_signals = {}
        self.original_training_labels = None

        # 用于存储伪复合数据
        self.pseudo_compound_data = {}
    def _get_enhanced_attributes(self):

        return {ft: np.array(attrs, dtype=np.float32)
                for ft, attrs in self.fault_location_attributes.items()}

    def compute_enhanced_attributes(self, signal, fault_type):
        """计算5维增强属性向量"""
        # 获取3维位置编码
        location_attrs = np.array(self.fault_location_attributes[fault_type], dtype=np.float32)

        # 计算频域特征
        hf_ratio, lf_ratio = self.freq_extractor.extract_frequency_ratios(signal)

        # 组合成5维向量
        enhanced_attrs = np.concatenate([location_attrs, [hf_ratio, lf_ratio]])

        return enhanced_attrs.astype(np.float32)

    def train_autoencoder_with_pseudo_compound(self,
                                               X_train, labels,
                                               epochs=AE_EPOCHS,
                                               batch_size=AE_BATCH_SIZE,
                                               lr=AE_LR,
                                               contrastive_weight=AE_CONTRASTIVE_WEIGHT):
        """训练AE并生成伪复合数据"""
        print("训练自编码器并生成伪复合故障数据...")

        # 1. 先按原来的方式训练AE
        self.train_autoencoder(X_train, labels, epochs, batch_size, lr, contrastive_weight)

        # 2. 准备单一故障信号字典
        single_fault_signals = {}
        single_fault_indices = [self.fault_types[name] for name in self.single_fault_types_ordered]

        for i, fault_name in enumerate(self.single_fault_types_ordered):
            fault_idx = self.fault_types[fault_name]
            mask = (labels == fault_idx)
            if np.any(mask):
                single_fault_signals[fault_name] = X_train[mask]
                print(f"  {fault_name}: {np.sum(mask)} 个样本")

        # 3. 生成伪复合故障
        pseudo_compound_signals = self.pseudo_generator.generate_compound_signals(
            single_fault_signals, num_samples_per_combination=100
        )

        # 4. 对伪复合信号进行AE编码并计算增强属性
        self.pseudo_compound_data = {}

        with torch.no_grad():
            self.autoencoder.eval()
            for compound_name, signals in pseudo_compound_signals.items():
                encoded_list = []
                enhanced_attrs_list = []

                for signal in signals:
                    # AE编码
                    signal_tensor = torch.FloatTensor(signal).unsqueeze(0).to(self.device)
                    encoded = self.autoencoder.encode(signal_tensor).cpu().numpy().squeeze()
                    encoded_list.append(encoded)

                    # 计算增强属性
                    enhanced_attr = self.compute_enhanced_attributes(signal, compound_name)
                    enhanced_attrs_list.append(enhanced_attr)

                self.pseudo_compound_data[compound_name] = {
                    'encoded': np.array(encoded_list),
                    'attributes': np.array(enhanced_attrs_list),
                    'signals': signals
                }

                print(f"  生成 {compound_name} 的编码和属性: {len(encoded_list)} 个")

        return self.autoencoder
    def build_knowledge_semantics(self):
        """构建基于轴承故障位置和尺寸的知识语义 (Unchanged from original)"""
        knowledge_semantics = {
            fault_type: np.array(location_encoding, dtype=np.float32)
            for fault_type, location_encoding in self.fault_location_attributes.items()
        }
        self.knowledge_dim = 3

        return knowledge_semantics

    def _ae_contrastive_loss(self, latent, latent_aug, labels, temperature=0.2):
        """计算对比损失 (logic matching gen.py _contrastive_loss)."""
        batch_size = latent.size(0)
        if batch_size <= 1: return torch.tensor(0.0, device=latent.device)

        latent_norm = nn.functional.normalize(latent, p=2, dim=1)
        latent_aug_norm = nn.functional.normalize(latent_aug, p=2, dim=1)

        if torch.isnan(latent_norm).any() or torch.isnan(latent_aug_norm).any():
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

            finite_loss_terms = contrastive_loss_terms[torch.isfinite(contrastive_loss_terms)]
            if len(finite_loss_terms) > 0:
                mean_loss = finite_loss_terms.mean()
            else:

                mean_loss = torch.tensor(0.0, device=latent.device)
        else:
            mean_loss = contrastive_loss_terms.mean()

        if torch.isnan(mean_loss) or torch.isinf(mean_loss):
            return torch.tensor(0.0, device=latent.device)

        return mean_loss

    def train_autoencoder(self,
                          X_train,  # numpy array (N, L)
                          labels,  # numpy array (N,)
                          epochs=AE_EPOCHS,
                          batch_size=AE_BATCH_SIZE,
                          lr=AE_LR,
                          contrastive_weight=AE_CONTRASTIVE_WEIGHT):

        # 1) 数据转张量
        device = self.device
        X_tensor = torch.FloatTensor(X_train).to(device)  # [N, L]
        y_tensor = torch.LongTensor(labels).to(device)  # [N]

        # 2) 构建 AE 模型与 Trainer
        ae_model = ContrastiveAutoencoder(
            input_length=X_train.shape[1],
            latent_dim=self.latent_dim_config
        ).to(device)
        trainer = AETrainer(ae_model, device=device)

        # 3) 训练
        trainer.train(
            X=X_tensor,
            y=y_tensor,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            contrastive_weight=contrastive_weight
        )

        # 4) 保存训练结果
        self.autoencoder = trainer.model
        self.actual_latent_dim = self.autoencoder.latent_dim
        print(f"  AE latent dim set to {self.actual_latent_dim}")

        # 5) 提取所有训练样本的潜特征
        self.autoencoder.eval()
        all_latent = []
        with torch.no_grad():
            step = batch_size * 2  # Can be larger for inference
            for i in range(0, X_tensor.size(0), step):
                batch = X_tensor[i:i + step]
                if batch.size(0) == 0:
                    continue
                # Ensure input to AE encode is [B, L] or [B, 1, L]
                if batch.dim() == 3 and batch.shape[1] == 1:  # if [B,1,L]
                    pass  # already correct
                elif batch.dim() == 2:  # if [B,L]
                    pass  # AE encode handles unsqueezing
                else:  # Reshape if needed, e.g. from other formats
                    batch = batch.view(batch.size(0), -1)  # Flatten to [B,L]
                    if batch.shape[1] != self.autoencoder.input_length:
                        print(
                            f"Warning: AE batch input length mismatch after flatten. Expected {self.autoencoder.input_length}, got {batch.shape[1]}. Skipping batch.")
                        continue

                z = self.autoencoder.encode(batch)  # [b, latent_dim]
                all_latent.append(z.cpu().numpy())

        if not all_latent:
            print("Warning: No latent features extracted from AE.")
            self.all_latent_features = np.empty((0, self.actual_latent_dim))
            self.all_latent_labels = np.empty((0,))
            self.data_semantics = {}
            return

        all_latent = np.vstack(all_latent)
        self.all_latent_features = all_latent
        # Ensure labels correspond to the successfully processed latent features
        self.all_latent_labels = labels[:all_latent.shape[0]]

        # 6) 计算每个故障类型的语义中心 (centroids)
        self.data_semantics = {}
        unique_labels = np.unique(self.all_latent_labels)
        for lbl in unique_labels:
            mask = (self.all_latent_labels == lbl)
            feats = self.all_latent_features[mask]  # Use self.all_latent_features
            if feats.shape[0] == 0:
                continue
            # 最多两个簇保留主要分布
            n_clusters = min(2, feats.shape[0])
            if feats.shape[0] > 1:  # KMeans needs at least 2 samples for n_clusters=2
                try:
                    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(feats)
                    counts = np.bincount(km.labels_)
                    main_cluster_label = np.argmax(counts)
                    centroid = km.cluster_centers_[main_cluster_label]
                except Exception as e:
                    print(f"Warning: KMeans clustering failed for label {lbl}: {e}. Using mean as centroid.")
                    centroid = np.mean(feats, axis=0)
            else:  # Only one sample for this label
                centroid = feats[0]

            fault_name = self.idx_to_fault.get(int(lbl), f"label_{lbl}")
            self.data_semantics[fault_name] = centroid.astype(np.float32)

        print(f"AE training complete. Computed {len(self.data_semantics)} class centroids.")

    def extract_data_semantics(self, X, fault_labels=None):
        """Extract data semantics using the trained AE (spectral clustering part retained)."""

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
                    batch_x = batch_x[batch_valid_mask]
                    batch_indices = batch_indices[batch_valid_mask.cpu().numpy()]

                if batch_x.shape[0] == 0: continue

                # Ensure input to AE encode is [B, L] or [B, 1, L]
                if batch_x.dim() == 3 and batch_x.shape[1] == 1:  # if [B,1,L]
                    pass
                elif batch_x.dim() == 2:  # if [B,L]
                    pass
                else:
                    batch_x = batch_x.view(batch_x.size(0), -1)
                    if batch_x.shape[1] != self.autoencoder.input_length:
                        print(
                            f"Warning: AE batch input length mismatch in extract_data_semantics. Expected {self.autoencoder.input_length}, got {batch_x.shape[1]}. Skipping batch.")
                        continue

                z_batch = self.autoencoder.encode(batch_x)

                batch_z_valid_mask = torch.all(torch.isfinite(z_batch), dim=1)
                if not batch_z_valid_mask.all():
                    z_batch = z_batch[batch_z_valid_mask]
                    batch_indices = batch_indices[batch_z_valid_mask.cpu().numpy()]

                if z_batch.shape[0] > 0:
                    all_data_semantics_list.append(z_batch.cpu().numpy())
                    filtered_indices.extend(batch_indices)

        if not all_data_semantics_list:
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
                    if len(fault_type_semantics) > 10:  # spectral clustering needs enough samples
                        n_clusters = max(2, min(5, len(fault_type_semantics) // 3))
                        # Ensure n_neighbors is less than n_samples
                        n_neighbors_spectral = max(5, min(10, len(fault_type_semantics) - 1))
                        if n_neighbors_spectral < 2:  # SpectralClustering needs at least 2 neighbors
                            prototype = np.mean(fault_type_semantics, axis=0)
                        else:
                            try:
                                spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors',
                                                              assign_labels='kmeans', random_state=42,
                                                              n_neighbors=n_neighbors_spectral)
                                cluster_labels = spectral.fit_predict(fault_type_semantics)

                                if cluster_labels is None or len(cluster_labels) != len(fault_type_semantics):
                                    prototype = np.mean(fault_type_semantics, axis=0)
                                else:
                                    valid_cluster_labels = cluster_labels[cluster_labels >= 0]
                                    if len(valid_cluster_labels) == 0:  # All samples might be outliers
                                        prototype = np.mean(fault_type_semantics, axis=0)
                                    else:
                                        counts = np.bincount(valid_cluster_labels)
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
                                print(f"W: Spectral clustering failed for fault {fault}: {e}. Using mean.")
                                prototype = np.mean(fault_type_semantics, axis=0)
                    else:  # Not enough samples for spectral clustering
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
        """使用增强属性合成复合故障语义"""

        # 使用增强的AttributeSemanticMapper
        mapper = EnhancedAttributeSemanticMapper(
            attr_dim=self.enhanced_attribute_dim,
            semantic_dim=self.actual_latent_dim,
        ).to(self.device)

        optimizer = torch.optim.AdamW(mapper.parameters(), lr=0.001, weight_decay=1e-5)
        mse_loss_fn = nn.MSELoss()

        X_mapper_list = []
        Y_mapper_list = []

        print("准备Mapper训练数据...")

        # 1. 单一故障的训练数据
        for sf_name in self.single_fault_types_ordered:
            if sf_name in single_fault_prototypes:
                # 计算该类型信号的平均频域特征
                if hasattr(self, 'all_latent_features') and hasattr(self, 'all_latent_labels'):
                    fault_idx = self.fault_types[sf_name]
                    mask = (self.all_latent_labels == fault_idx)

                    if np.any(mask):
                        # 从原始训练信号中采样计算频域特征
                        sample_signals = self._get_sample_signals_for_fault(sf_name)

                        for signal in sample_signals[:10]:  # 取10个样本
                            enhanced_attr = self.compute_enhanced_attributes(signal, sf_name)
                            proto_vec = single_fault_prototypes[sf_name]

                            if np.all(np.isfinite(enhanced_attr)) and np.all(np.isfinite(proto_vec)):
                                X_mapper_list.append(enhanced_attr)
                                Y_mapper_list.append(proto_vec)

        # 2. 伪复合故障的训练数据
        if hasattr(self, 'pseudo_compound_data'):
            for compound_name, data in self.pseudo_compound_data.items():
                encoded_list = data['encoded']
                attrs_list = data['attributes']

                for encoded, attr in zip(encoded_list, attrs_list):
                    if np.all(np.isfinite(attr)) and np.all(np.isfinite(encoded)):
                        X_mapper_list.append(attr)
                        Y_mapper_list.append(encoded)

        if not X_mapper_list:
            print("警告: 没有足够的训练数据，使用回退方法")
            return self._synthesize_by_rule(single_fault_prototypes, 'average_prototypes')

        # 转换为张量
        X_mapper_train = torch.FloatTensor(np.array(X_mapper_list)).to(self.device)
        Y_mapper_train = torch.FloatTensor(np.array(Y_mapper_list)).to(self.device)

        print(f"Mapper训练数据: {len(X_mapper_list)} 个样本")

        # 训练Mapper
        mapper_epochs = 200
        mapper_batch_size = min(32, len(X_mapper_train))

        train_dataset = TensorDataset(X_mapper_train, Y_mapper_train)
        train_loader = DataLoader(train_dataset, batch_size=mapper_batch_size, shuffle=True)

        best_loss = float('inf')
        patience = 30
        patience_counter = 0

        for epoch in range(mapper_epochs):
            mapper.train()
            epoch_loss = 0
            num_batches = 0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred_semantics = mapper(x_batch)
                loss = mse_loss_fn(pred_semantics, y_batch)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(mapper.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches

                if (epoch + 1) % 20 == 0:
                    print(f"  Mapper Epoch {epoch + 1}/{mapper_epochs}, Loss: {avg_loss:.6f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Mapper早停于epoch {epoch + 1}")
                        break

        # 生成复合故障语义
        mapper.eval()
        synthesized_compound_semantics = {}

        with torch.no_grad():
            for cf_name in self.compound_fault_definitions.keys():
                # 计算该复合故障的平均频域特征（如果有伪数据）
                if hasattr(self, 'pseudo_compound_data') and cf_name in self.pseudo_compound_data:
                    avg_attrs = np.mean(self.pseudo_compound_data[cf_name]['attributes'], axis=0)
                else:
                    # 回退到基本属性
                    location_attrs = np.array(self.fault_location_attributes[cf_name])
                    avg_attrs = np.concatenate([location_attrs, [0.2, 0.3]])  # 默认频域值

                if np.all(np.isfinite(avg_attrs)):
                    attr_tensor = torch.FloatTensor(avg_attrs).unsqueeze(0).to(self.device)
                    generated_semantic = mapper(attr_tensor).cpu().numpy().squeeze(0)

                    if np.all(np.isfinite(generated_semantic)):
                        synthesized_compound_semantics[cf_name] = generated_semantic
                        print(f"  生成 {cf_name} 的语义表示")

        return synthesized_compound_semantics

    def _get_sample_signals_for_fault(self, fault_name):
        """获取指定故障类型的样本信号"""
        if fault_name in self.original_training_signals:
            signals = self.original_training_signals[fault_name]
            # 随机选择一些样本（比如最多10个）
            num_samples = min(10, len(signals))
            indices = np.random.choice(len(signals), num_samples, replace=False)
            return signals[indices]
        else:
            print(f"警告: 未找到故障类型 {fault_name} 的原始信号")
            return []

    def _synthesize_by_rule(self, single_fault_prototypes, rule, specific_types=None):
        """Helper function for direct rule-based synthesis (average or sum)."""
        temp_compound_semantics = {}
        target_compound_types = specific_types if specific_types is not None else self.compound_fault_definitions.keys()

        for cf_name in target_compound_types:
            constituents = self.compound_fault_definitions.get(cf_name)
            if constituents is None:
                print(f"Warning: No definition for compound fault '{cf_name}' in _synthesize_by_rule.")
                continue

            component_semantic_list = []
            valid_components = True
            for constituent_name in constituents:
                if constituent_name in single_fault_prototypes and \
                        np.all(np.isfinite(single_fault_prototypes[constituent_name])):
                    component_semantic_list.append(single_fault_prototypes[constituent_name])
                else:
                    valid_components = False
                    print(
                        f"Warning: Constituent '{constituent_name}' for '{cf_name}' is missing or invalid in single_fault_prototypes.")
                    break

            if not component_semantic_list:  # Handles cases where constituents might be empty or all invalid
                valid_components = False

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
        component_vecs = [single_prototypes[comp] for comp in components if
                          np.all(np.isfinite(single_prototypes[comp]))]
        if not component_vecs: return refined  # No valid component vectors

        avg_component_vec = np.mean(component_vecs, axis=0)
        if not np.all(np.isfinite(avg_component_vec)): return refined  # Avg component vec is not finite

        # 计算当前相似度
        norm_refined = np.linalg.norm(refined)
        norm_avg_comp = np.linalg.norm(avg_component_vec)

        if norm_refined < 1e-8 or norm_avg_comp < 1e-8:
            current_sim = 0.0
        else:
            current_sim = np.dot(refined, avg_component_vec) / (norm_refined * norm_avg_comp)

        # 如果相似度过低，向平均组件向量方向调整
        if current_sim < 0.3:
            adjustment_factor = 0.3
            refined = (1 - adjustment_factor) * refined + adjustment_factor * avg_component_vec

        if compound_type == 'inner_outer':

            mid = len(refined) // 2
            refined[mid:] = refined[mid:] * 1.05  # Small boost

        elif compound_type == 'inner_ball' or compound_type == 'outer_ball':

            if len(refined) > 0:  # Ensure refined is not empty
                mod_pattern = np.sin(np.linspace(0, 3 * np.pi, len(refined))) * 0.05  # Small amplitude
                refined = refined + mod_pattern * np.mean(np.abs(refined))  # Scale modulation by mean abs value

        elif compound_type == 'inner_outer_ball':

            refined = refined * 1.1

        final_norm = np.linalg.norm(refined)
        original_norm = np.linalg.norm(semantic_vec)

        if final_norm > 1e-8 and original_norm > 1e-8:
            refined = refined / final_norm * original_norm
        elif final_norm > 1e-8:
            refined = refined / final_norm

        return refined if np.all(np.isfinite(refined)) else semantic_vec

    def _generate_fallback_semantics(self, fault_types, single_prototypes):
        compound_semantics = {}

        for compound_type in fault_types:
            components = compound_type.split('_')
            component_semantics = []

            # 收集有效的组件语义
            for comp in components:
                if comp in single_prototypes and np.all(np.isfinite(single_prototypes[comp])):
                    component_semantics.append(single_prototypes[comp])

            if not component_semantics:  # If no valid components found
                print(f"Warning: No valid components to generate fallback for {compound_type}")
                continue

            if component_semantics:
                # 使用平均组合
                combined = np.mean(component_semantics, axis=0)

                # 应用后处理
                combined = self._post_process_compound_semantic(
                    combined, compound_type, single_prototypes
                )

                if np.all(np.isfinite(combined)):
                    compound_semantics[compound_type] = combined
                    print(f"  - 已补充生成 {compound_type} 的语义表示 (fallback)")
                else:
                    print(f"Warning: Fallback generation for {compound_type} resulted in non-finite semantic.")

        return compound_semantics

    def _fallback_compound_synthesis(self, single_fault_prototypes):
        print("Note: _fallback_compound_synthesis called, using average of components.")
        compound_semantics = {}
        compound_combinations = self.compound_fault_definitions  # Use the class definition

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

            if not component_semantics:
                all_valid = False

            if all_valid and component_semantics:
                # 简单平均作为回退策略
                synthesized = np.mean(component_semantics, axis=0)

                # 应用后处理
                synthesized = self._post_process_compound_semantic(
                    synthesized, compound_type, single_fault_prototypes
                )

                if np.all(np.isfinite(synthesized)):
                    compound_semantics[compound_type] = synthesized
                else:
                    print(
                        f"Warning: Fallback synthesis for {compound_type} (in _fallback_compound_synthesis) resulted in non-finite semantic.")

        return compound_semantics


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

        if identity.shape == out.shape:
            out += identity
        else:
            print(
                f"Warning: Skipping residual connection in ResidualBlock1D due to shape mismatch: identity {identity.shape}, out {out.shape}")

        out = self.relu(out)
        return out


class AEDataSemanticCNN(nn.Module):
    def __init__(self, input_length=SEGMENT_LENGTH, semantic_dim=AE_LATENT_DIM,
                 num_classes=8, feature_dim=CNN_FEATURE_DIM, dropout_rate=0.3):  # Added dropout_rate
        super().__init__()
        self.input_length = input_length
        self.semantic_dim = semantic_dim
        self.feature_dim = feature_dim
        self.signal_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7),  # L/2
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/4

            ResidualBlock1D(64, 64),

            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),  # L/8
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(kernel_size=2, stride=2),  # L/16

            ResidualBlock1D(128, 128),

            nn.Conv1d(128, self.feature_dim, kernel_size=5, stride=2, padding=2),  # L/32
            nn.BatchNorm1d(self.feature_dim),
            nn.LeakyReLU(0.1),

            ResidualBlock1D(self.feature_dim, self.feature_dim),
            nn.AdaptiveAvgPool1d(1)
        )
        self.semantic_processor = nn.Sequential(
            nn.Linear(self.semantic_dim, (self.feature_dim + self.semantic_dim) // 2),
            nn.LayerNorm((self.feature_dim + self.semantic_dim) // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear((self.feature_dim + self.semantic_dim) // 2, 2 * self.feature_dim)
        )
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(min(dropout_rate + 0.1, 0.7)),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x, semantic, return_features=False):
        """
        Forward pass with FiLM modulation.
        x: signal input [B, L] or [B, 1, L]
        semantic: semantic input from AE [B, semantic_dim (AE_LATENT_DIM)]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Ensure [B, 1, L] for Conv1d

        signal_features = self.signal_encoder(x).squeeze(-1)  # Output: [B, feature_dim]

        if semantic is None:

            gamma = torch.zeros_like(signal_features)
            beta = torch.zeros_like(signal_features)
        else:
            if semantic.shape[1] != self.semantic_dim:
                raise ValueError(
                    f"AEDataSemanticCNN: Semantic input dim mismatch. Expected {self.semantic_dim}, got {semantic.shape[1]}")

            film_params = self.semantic_processor(semantic)  # semantic is AE's output
            gamma = film_params[:, :self.feature_dim]
            beta = film_params[:, self.feature_dim:]

        modulated_features = (1 + gamma) * signal_features + beta

        if return_features:
            return modulated_features

        logits = self.classifier_head(modulated_features)
        return logits
class SemanticAutoencoder(nn.Module):
    def __init__(self, feature_dim, semantic_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        self.projection_H = nn.Linear(feature_dim, semantic_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection_H.weight)
        if self.projection_H.bias is not None:
            nn.init.zeros_(self.projection_H.bias)

    def forward(self, x_features):
        projected_semantics = self.projection_H(x_features)
        return projected_semantics

    def reconstruct_features(self, a_semantics):
        reconstructed_x = F.linear(a_semantics, self.projection_H.weight.t(), bias=None)
        return reconstructed_x


class ZeroShotCompoundFaultDiagnosis:

    def __init__(self, data_path, sample_length=SEGMENT_LENGTH, latent_dim=AE_LATENT_DIM,
                 batch_size=DEFAULT_BATCH_SIZE, cnn_dropout_rate=0.3,
                 compound_data_semantic_generation_rule='mapper_output',
                 sae_mu=SAE_MU):
        self.data_path = data_path
        self.sample_length = sample_length
        self.ae_latent_dim_config = latent_dim
        self.batch_size = batch_size
        self.cnn_dropout_rate = cnn_dropout_rate
        self.compound_data_semantic_generation_rule = compound_data_semantic_generation_rule
        self.sae_mu = sae_mu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.preprocessor = DataPreprocessor(sample_length=self.sample_length)
        self.semantic_builder = FaultSemanticBuilder(
            latent_dim=self.ae_latent_dim_config,  # Pass AE's latent dim
            compound_data_semantic_generation_rule=self.compound_data_semantic_generation_rule
        )
        self.fault_types = {
            'normal': 0, 'inner': 1, 'outer': 2, 'ball': 3,
            'inner_outer': 4, 'inner_ball': 5, 'outer_ball': 6, 'inner_outer_ball': 7
        }
        self.idx_to_fault = {v: k for k, v in self.fault_types.items()}
        self.semantic_builder.idx_to_fault = self.idx_to_fault
        self.single_fault_types_ordered = ['normal', 'inner', 'outer', 'ball']
        self.compound_fault_definitions = {
            'inner_outer': ['inner', 'outer'],
            'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'],
            'inner_outer_ball': ['inner', 'outer', 'ball']
        }
        self.compound_fault_types = list(self.compound_fault_definitions.keys())
        self.num_classes = len(self.fault_types)

        self.cnn_model = None
        self.sae_model = None

        self.actual_ae_latent_dim = -1
        self.cnn_feature_dim = CNN_FEATURE_DIM
        self.fused_semantic_dim = -1

    def load_data(self):
        """改进的数据加载函数，确保每类故障样本数量一致且训练集验证集同样平衡"""
        print("加载并预处理数据，确保训练和测试集严格分离，且类别平衡...")
        single_fault_keys = ['normal', 'inner', 'outer', 'ball']

        fault_files = {
            'normal': 'normal.mat', 'inner': 'inner.mat', 'outer': 'outer.mat', 'ball': 'ball.mat',
            'inner_outer': 'inner_outer.mat', 'inner_ball': 'inner_ball.mat',
            'outer_ball': 'outer_ball.mat', 'inner_outer_ball': 'inner_outer_ball.mat'
        }

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
                                  isinstance(mat_data[k], np.ndarray) and mat_data[k].ndim > 0 and mat_data[
                                      k].size > 500]

                if not potential_keys:
                    common_keys_order = ['DE_time', 'FE_time', 'BA_time', 'Data']
                    for ck_part in common_keys_order:
                        found_key = next((k for k in mat_data if ck_part in k and not k.startswith('__')), None)
                        if found_key:
                            potential_keys.append(found_key)
                            break

                if potential_keys:
                    de_keys = [k for k in potential_keys if 'DE_time' in k]
                    if de_keys:
                        signal_data_raw = mat_data[de_keys[0]]  # Pick first DE key
                    else:
                        signal_data_raw = mat_data[max(potential_keys, key=lambda k: mat_data[k].size)]


                signal_data_flat = signal_data_raw.ravel().astype(np.float64)
                max_len = 200000  # Limit signal length for performance
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


        print("\n预处理所有单一故障数据...")
        train_preprocessor = DataPreprocessor(
            sample_length=self.sample_length,
            overlap=0.25,
            augment=True,
            random_seed=42
        )

        all_single_fault_segments_by_type = {}
        min_samples_after_preprocessing = float('inf')

        for fault_type, signal_data in single_fault_raw_signals.items():
            print(f"预处理 {fault_type} 数据...")
            processed_segments = train_preprocessor.preprocess(signal_data, augmentation=True)

            if processed_segments is None or processed_segments.shape[0] == 0:
                print(f"警告: {fault_type} 数据预处理后无有效分段")
                continue

            all_single_fault_segments_by_type[fault_type] = processed_segments
            min_samples_after_preprocessing = min(min_samples_after_preprocessing, len(processed_segments))
            print(f"  - {fault_type}: {len(processed_segments)} 个分段")


        print(
            f"\n平衡各单一故障类型样本数量至每类 {min_samples_after_preprocessing} 个样本 (基于预处理和增强后的最少样本数)")

        train_ratio = 0.7
        samples_per_class_train = int(min_samples_after_preprocessing * train_ratio)
        samples_per_class_val = min_samples_after_preprocessing - samples_per_class_train

        print(f"每类故障分配: {samples_per_class_train}个训练样本, {samples_per_class_val}个验证样本")

        X_train_list = []
        y_train_list = []
        X_val_list = []
        y_val_list = []

        for fault_type, segments in all_single_fault_segments_by_type.items():
            label_idx = self.fault_types[fault_type]
            num_available_segments = len(segments)
            current_samples_train = min(samples_per_class_train, num_available_segments)
            current_samples_val = min(samples_per_class_val, num_available_segments - current_samples_train)

            indices = np.random.permutation(num_available_segments)
            train_indices = indices[:current_samples_train]
            val_indices = indices[current_samples_train: current_samples_train + current_samples_val]

            if len(train_indices) > 0:
                X_train_list.append(segments[train_indices])
                y_train_list.extend([label_idx] * len(train_indices))

            if len(val_indices) > 0:
                X_val_list.append(segments[val_indices])
                y_val_list.extend([label_idx] * len(val_indices))

            print(f"  - {fault_type}: 训练 {len(train_indices)} 个, 验证 {len(val_indices)} 个")

        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)

        X_val = np.vstack(X_val_list) if X_val_list else np.empty((0, self.sample_length), dtype=np.float32)
        y_val = np.array(y_val_list) if y_val_list else np.array([], dtype=int)

        print(f"\n数据集划分完成:")
        print(f"* 训练集: {len(X_train)} 个样本")
        if len(X_train) > 0: print(f"  - 形状: {X_train.shape}")
        print(f"* 验证集: {len(X_val)} 个样本")
        if len(X_val) > 0: print(f"  - 形状: {X_val.shape}")

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
        if not compound_fault_raw_signals:
            print("警告: 没有加载复合故障原始信号数据，测试集将为空。")

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
        y_test = np.array(y_test_list) if y_test_list else np.array([], dtype=int)

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
        """构建增强的语义表示"""
        print("构建增强故障语义...")
        knowledge_semantics = self.semantic_builder.build_knowledge_semantics()

        print("  训练自编码器并生成伪复合数据...")
        X_train_ae = data_dict.get('X_train')
        y_train_ae = data_dict.get('y_train')

        # 使用新的训练方法
        self.semantic_builder.train_autoencoder_with_pseudo_compound(
            X_train_ae, labels=y_train_ae, epochs=AE_EPOCHS,
            batch_size=AE_BATCH_SIZE, lr=AE_LR
        )

        self.actual_ae_latent_dim = self.semantic_builder.actual_latent_dim
        single_fault_prototypes = self.semantic_builder.data_semantics

        # 使用增强的语义合成
        compound_data_semantics = self.semantic_builder.synthesize_compound_semantics(single_fault_prototypes)
        print(f"  Compound data semantics synthesized for {len(compound_data_semantics)} types.")

        data_only_semantics = {**single_fault_prototypes, **compound_data_semantics}

        fused_semantics = {}
        self.fused_semantic_dim = self.semantic_builder.knowledge_dim + self.actual_ae_latent_dim
        print(
            f"  Fused semantic dimension will be: {self.fused_semantic_dim} (Knowledge: {self.semantic_builder.knowledge_dim} + AE_Data: {self.actual_ae_latent_dim})")

        for ft, k_vec in knowledge_semantics.items():
            d_vec = data_only_semantics.get(ft)

            if d_vec is not None and k_vec is not None and \
                    np.all(np.isfinite(k_vec)) and np.all(np.isfinite(d_vec)) and \
                    len(k_vec) == self.semantic_builder.knowledge_dim and \
                    len(d_vec) == self.actual_ae_latent_dim:
                fused_vec = np.concatenate([k_vec, d_vec]).astype(np.float32)

                if np.all(np.isfinite(fused_vec)) and len(fused_vec) == self.fused_semantic_dim:
                    fused_semantics[ft] = fused_vec
                else:
                    print(
                        f"W: Fused vector for '{ft}' is invalid or has wrong dimension after concatenation. k_len={len(k_vec)}, d_len={len(d_vec)}, expected_fused={self.fused_semantic_dim}")

        single_fault_latent_features = self.semantic_builder.all_latent_features
        single_fault_latent_labels = self.semantic_builder.all_latent_labels
        for ft_name in self.fault_types.keys():
            if ft_name not in fused_semantics:
                print(f"Warning: Fused semantic for '{ft_name}' not generated. Attempting fallback.")
                k_fallback = knowledge_semantics.get(ft_name)
                d_fallback = None
                if ft_name in single_fault_prototypes and np.all(np.isfinite(single_fault_prototypes[ft_name])):
                    d_fallback = single_fault_prototypes[ft_name]
                elif ft_name in compound_data_semantics and np.all(np.isfinite(compound_data_semantics[ft_name])):
                    d_fallback = compound_data_semantics[ft_name]
                else:  # Last resort for d_vec if not found
                    if self.actual_ae_latent_dim > 0:
                        d_fallback = np.zeros(self.actual_ae_latent_dim, dtype=np.float32)
                        print(f"  Using zero vector for AE semantic part of '{ft_name}' in fallback.")

                if k_fallback is not None and d_fallback is not None and \
                        len(k_fallback) == self.semantic_builder.knowledge_dim and \
                        len(d_fallback) == self.actual_ae_latent_dim:
                    fused_vec_fallback = np.concatenate([k_fallback, d_fallback]).astype(np.float32)
                    if np.all(np.isfinite(fused_vec_fallback)) and len(fused_vec_fallback) == self.fused_semantic_dim:
                        fused_semantics[ft_name] = fused_vec_fallback
                        print(f"  Successfully created fallback fused semantic for '{ft_name}'.")
                    else:
                        print(f"Error: Fallback fused semantic for '{ft_name}' is invalid.")
                else:
                    print(f"Error: Could not create fallback fused semantic for '{ft_name}' due to missing components.")

        return {
            'knowledge_semantics': knowledge_semantics,
            'data_prototypes': single_fault_prototypes,
            'compound_data_semantics': compound_data_semantics,
            'data_only_semantics': data_only_semantics,  # AE latent space prototypes
            'fused_semantics': fused_semantics,  # knowledge + AE latent space prototypes
            'single_fault_latent_features': single_fault_latent_features,  # All AE latent features from training
            'single_fault_latent_labels': single_fault_latent_labels
        }

    def train_ae_data_semantic_cnn(self, data_dict, semantic_dict, epochs=CNN_EPOCHS,
                                   batch_size=DEFAULT_BATCH_SIZE, lr=CNN_LR):
        """使用AE实时提取的语义训练CNN (FiLM modulation)"""
        print("训练基于AE实时数据语义的CNN (with FiLM)...")

        if self.cnn_model is None:  # Initialize if not done externally
            self.cnn_model = AEDataSemanticCNN(
                input_length=self.sample_length,
                semantic_dim=self.actual_ae_latent_dim,  # FiLM uses AE's latent dim
                num_classes=self.num_classes,
                feature_dim=self.cnn_feature_dim,
                dropout_rate=self.cnn_dropout_rate
            ).to(self.device)
        else:
            self.cnn_model = self.cnn_model.to(self.device)

        self.semantic_builder.autoencoder.eval()

        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

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
        best_val_loss_at_best_acc = float('inf')
        patience, patience_counter = 10, 0

        for epoch in range(epochs):
            self.cnn_model.train()
            train_loss_epoch, correct_epoch, total_epoch = 0, 0, 0
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                with torch.no_grad():
                    ae_inputs = inputs.squeeze(1) if inputs.dim() == 3 and inputs.shape[1] == 1 else inputs
                    if ae_inputs.dim() != 2 or ae_inputs.shape[1] != self.semantic_builder.autoencoder.input_length:
                        print(
                            f"W: CNN train - AE input shape mismatch. Expected [B, {self.semantic_builder.autoencoder.input_length}], got {ae_inputs.shape}. Skipping batch.")
                        continue
                    real_time_ae_semantics = self.semantic_builder.autoencoder.encode(ae_inputs)  # [B, ae_latent_dim]

                logits = self.cnn_model(inputs, real_time_ae_semantics)
                loss = criterion(logits, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"W: CNN loss is NaN/Inf in train batch {batch_idx}. Skipping backward.")
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=5.0)
                optimizer.step()
                if scheduler: scheduler.step()

                train_loss_epoch += loss.item() * inputs.size(0)
                _, predicted = logits.max(1)
                total_epoch += targets.size(0)
                correct_epoch += predicted.eq(targets).sum().item()
            avg_train_loss = train_loss_epoch / total_epoch if total_epoch > 0 else 0
            avg_train_acc = 100. * correct_epoch / total_epoch if total_epoch > 0 else 0

            val_loss_epoch, val_correct_epoch, val_total_epoch = 0, 0, 0
            if val_loader:
                self.cnn_model.eval()
                with torch.no_grad():
                    for inputs_val, targets_val in val_loader:
                        inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)

                        ae_inputs_val = inputs_val.squeeze(1) if inputs_val.dim() == 3 and inputs_val.shape[
                            1] == 1 else inputs_val
                        if ae_inputs_val.dim() != 2 or ae_inputs_val.shape[
                            1] != self.semantic_builder.autoencoder.input_length:
                            print(
                                f"W: CNN val - AE input shape mismatch. Expected [B, {self.semantic_builder.autoencoder.input_length}], got {ae_inputs_val.shape}. Skipping batch.")
                            continue
                        real_time_ae_semantics_val = self.semantic_builder.autoencoder.encode(ae_inputs_val)

                        logits_val = self.cnn_model(inputs_val, real_time_ae_semantics_val)
                        loss_val = criterion(logits_val, targets_val)
                        val_loss_epoch += loss_val.item() * inputs_val.size(0)
                        _, predicted_val = logits_val.max(1)
                        val_total_epoch += targets_val.size(0)
                        val_correct_epoch += predicted_val.eq(targets_val).sum().item()

                avg_val_loss = val_loss_epoch / val_total_epoch if val_total_epoch > 0 else float('inf')
                avg_val_acc = 100. * val_correct_epoch / val_total_epoch if val_total_epoch > 0 else 0
                print(
                    f'CNN Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | '
                    f'Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%')

                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    best_val_loss_at_best_acc = avg_val_loss
                    patience_counter = 0
                    torch.save(self.cnn_model.state_dict(), 'best_ae_semantic_cnn_realtime.pth')
                    print(
                        f"保存最佳CNN模型 (实时语义)，验证集准确率: {avg_val_acc:.2f}%, 验证集损失: {avg_val_loss:.4f}")
                elif avg_val_acc == best_val_acc and avg_val_loss < best_val_loss_at_best_acc:
                    best_val_loss_at_best_acc = avg_val_loss
                    patience_counter = 0
                    torch.save(self.cnn_model.state_dict(), 'best_ae_semantic_cnn_realtime.pth')
                    print(
                        f"保存最佳CNN模型 (实时语义)，验证集准确率: {avg_val_acc:.2f}%, 验证集损失: {avg_val_loss:.4f} (损失更优)")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"CNN训练早停: {patience}轮内未改善 (基于准确率优先，其次损失)")
                    break
            else:  # No validation loader
                print(
                    f'CNN Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%')
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    torch.save(self.cnn_model.state_dict(), f'ae_semantic_cnn_realtime_epoch_{epoch + 1}.pth')
                    print(f"CNN模型已保存 (无验证集): epoch {epoch + 1}")

        if os.path.exists('best_ae_semantic_cnn_realtime.pth'):
            self.cnn_model.load_state_dict(torch.load('best_ae_semantic_cnn_realtime.pth', map_location=self.device))
            print(
                f"已加载最佳CNN模型 (实时语义) - 最终验证准确率: {best_val_acc:.2f}%, 对应损失: {best_val_loss_at_best_acc:.4f}")
        else:
            print("警告: 未找到最佳CNN模型文件 'best_ae_semantic_cnn_realtime.pth'。使用最后训练的模型。")

        self.cnn_model.eval()
        return self.cnn_model

    def train_semantic_autoencoder(self, data_dict, semantic_dict, epochs=SAE_EPOCHS, lr=SAE_LR, batch_size_sae=None):
        print("\n--- 开始训练语义自编码器 (SAE) ---")

        if batch_size_sae is None: batch_size_sae = self.batch_size

        self.cnn_model.eval()
        self.semantic_builder.autoencoder.eval()
        self.sae_model = SemanticAutoencoder(
            feature_dim=self.cnn_feature_dim,
            semantic_dim=self.fused_semantic_dim
        ).to(self.device)
        print(f"SAE 初始化: 输入维度 (CNN特征)={self.cnn_feature_dim}, 输出维度 (融合语义)={self.fused_semantic_dim}")

        X_train_signals, y_train_labels = data_dict['X_train'], data_dict['y_train']
        X_val_signals, y_val_labels = data_dict.get('X_val'), data_dict.get('y_val')

        fused_semantics_map = semantic_dict.get('fused_semantics')
        if not fused_semantics_map:
            print("错误: SAE训练缺少融合语义 (fused_semantics)。")
            return False
        sae_train_cnn_features_list = []
        sae_train_target_semantics_list = []
        single_fault_indices = [self.fault_types[name] for name in self.single_fault_types_ordered]

        temp_train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_signals), torch.LongTensor(y_train_labels)),
            batch_size=batch_size_sae, shuffle=False)

        with torch.no_grad():
            for signals_batch, labels_batch in temp_train_loader:
                signals_batch = signals_batch.to(self.device)
                ae_input_for_cnn = signals_batch.squeeze(1) if signals_batch.dim() == 3 and signals_batch.shape[
                    1] == 1 else signals_batch
                if ae_input_for_cnn.dim() != 2 or ae_input_for_cnn.shape[
                    1] != self.semantic_builder.autoencoder.input_length:
                    continue
                current_ae_semantics = self.semantic_builder.autoencoder.encode(ae_input_for_cnn)

                cnn_features_batch = self.cnn_model(signals_batch, current_ae_semantics, return_features=True)

                for i, label_idx in enumerate(labels_batch):
                    label_item = label_idx.item()
                    if label_item in single_fault_indices:
                        fault_name = self.idx_to_fault.get(label_item)
                        target_fused_semantic = fused_semantics_map.get(fault_name)
                        if target_fused_semantic is not None and np.all(np.isfinite(target_fused_semantic)):
                            sae_train_cnn_features_list.append(cnn_features_batch[i].cpu().numpy())
                            sae_train_target_semantics_list.append(target_fused_semantic)


        X_sae_train = torch.FloatTensor(np.array(sae_train_cnn_features_list)).to(self.device)
        A_sae_train = torch.FloatTensor(np.array(sae_train_target_semantics_list)).to(self.device)

        sae_train_dataset = TensorDataset(X_sae_train, A_sae_train)
        sae_train_loader = DataLoader(sae_train_dataset, batch_size=batch_size_sae, shuffle=True, drop_last=True)

        sae_val_loader = None
        if X_val_signals is not None and y_val_labels is not None and len(X_val_signals) > 0:
            sae_val_cnn_features_list = []
            sae_val_target_semantics_list = []
            temp_val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val_signals), torch.LongTensor(y_val_labels)),
                batch_size=batch_size_sae, shuffle=False)
            with torch.no_grad():
                for signals_batch_val, labels_batch_val in temp_val_loader:
                    signals_batch_val = signals_batch_val.to(self.device)
                    ae_input_val = signals_batch_val.squeeze(1) if signals_batch_val.dim() == 3 and \
                                                                   signals_batch_val.shape[
                                                                       1] == 1 else signals_batch_val
                    if ae_input_val.dim() != 2 or ae_input_val.shape[
                        1] != self.semantic_builder.autoencoder.input_length:
                        continue
                    current_ae_semantics_val = self.semantic_builder.autoencoder.encode(ae_input_val)
                    cnn_features_batch_val = self.cnn_model(signals_batch_val, current_ae_semantics_val,
                                                            return_features=True)

                    for i, label_idx_val in enumerate(labels_batch_val):
                        label_item_val = label_idx_val.item()
                        if label_item_val in single_fault_indices:
                            fault_name_val = self.idx_to_fault.get(label_item_val)
                            target_fused_semantic_val = fused_semantics_map.get(fault_name_val)
                            if target_fused_semantic_val is not None and np.all(np.isfinite(target_fused_semantic_val)):
                                sae_val_cnn_features_list.append(cnn_features_batch_val[i].cpu().numpy())
                                sae_val_target_semantics_list.append(target_fused_semantic_val)
            if sae_val_cnn_features_list:
                X_sae_val = torch.FloatTensor(np.array(sae_val_cnn_features_list)).to(self.device)
                A_sae_val = torch.FloatTensor(np.array(sae_val_target_semantics_list)).to(self.device)
                sae_val_dataset = TensorDataset(X_sae_val, A_sae_val)
                sae_val_loader = DataLoader(sae_val_dataset, batch_size=batch_size_sae, shuffle=False)

        optimizer = optim.AdamW(self.sae_model.parameters(), lr=lr, weight_decay=1e-5)
        mse_loss_fn = nn.MSELoss()
        sae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

        best_sae_val_loss = float('inf')
        sae_patience_epochs = 20
        sae_current_patience = 0

        for epoch in range(epochs):
            self.sae_model.train()
            epoch_total_loss, epoch_recon_loss, epoch_proj_loss = 0, 0, 0
            num_batches = 0
            for x_s_batch, a_s_batch in sae_train_loader:
                optimizer.zero_grad()
                a_pred_batch = self.sae_model(x_s_batch)
                x_recon_batch = self.sae_model.reconstruct_features(a_s_batch)

                loss1_feature_recon = mse_loss_fn(x_recon_batch, x_s_batch)
                loss2_semantic_proj = mse_loss_fn(a_pred_batch, a_s_batch)

                total_loss = loss1_feature_recon + self.sae_mu * loss2_semantic_proj

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"W: SAE loss NaN/Inf at epoch {epoch + 1}. Skipping batch.")
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sae_model.parameters(), 1.0)
                optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_recon_loss += loss1_feature_recon.item()
                epoch_proj_loss += loss2_semantic_proj.item()
                num_batches += 1

            avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else 0
            avg_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0
            avg_proj_loss = epoch_proj_loss / num_batches if num_batches > 0 else 0

            current_sae_lr = optimizer.param_groups[0]['lr']
            print(
                f"SAE Epoch [{epoch + 1}/{epochs}] LR: {current_sae_lr:.7f} | Total Loss: {avg_epoch_loss:.4f} | FeatRecon: {avg_recon_loss:.4f} | SemProj: {avg_proj_loss:.4f}")

            if sae_val_loader:
                self.sae_model.eval()
                val_total_loss, val_recon_loss, val_proj_loss = 0, 0, 0
                num_val_batches = 0
                with torch.no_grad():
                    for x_s_val_batch, a_s_val_batch in sae_val_loader:
                        a_pred_val_batch = self.sae_model(x_s_val_batch)
                        x_recon_val_batch = self.sae_model.reconstruct_features(a_s_val_batch)

                        val_loss1 = mse_loss_fn(x_recon_val_batch, x_s_val_batch)
                        val_loss2 = mse_loss_fn(a_pred_val_batch, a_s_val_batch)
                        batch_val_loss = val_loss1 + self.sae_mu * val_loss2

                        if not torch.isnan(batch_val_loss):
                            val_total_loss += batch_val_loss.item()
                            val_recon_loss += val_loss1.item()
                            val_proj_loss += val_loss2.item()
                            num_val_batches += 1

                avg_val_total_loss = val_total_loss / num_val_batches if num_val_batches > 0 else float('inf')
                avg_val_recon_loss = val_recon_loss / num_val_batches if num_val_batches > 0 else float('inf')
                avg_val_proj_loss = val_proj_loss / num_val_batches if num_val_batches > 0 else float('inf')

                print(
                    f"  SAE Val Loss: {avg_val_total_loss:.4f} | Val FeatRecon: {avg_val_recon_loss:.4f} | Val SemProj: {avg_val_proj_loss:.4f}")

                sae_scheduler.step(avg_val_total_loss)  # Step scheduler with validation loss

                if avg_val_total_loss < best_sae_val_loss:
                    best_sae_val_loss = avg_val_total_loss
                    torch.save(self.sae_model.state_dict(), 'best_semantic_autoencoder.pth')
                    print(f"  New best SAE model saved (Val Loss: {best_sae_val_loss:.4f})")
                    sae_current_patience = 0
                else:
                    sae_current_patience += 1
                    if sae_current_patience >= sae_patience_epochs:
                        print(f"SAE training early stopping at epoch {epoch + 1}.")
                        break
            else:  # No validation
                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    torch.save(self.sae_model.state_dict(), f'semantic_autoencoder_epoch_{epoch + 1}.pth')
                    print(f"  SAE model saved (Epoch {epoch + 1}, no validation)")

        if os.path.exists('best_semantic_autoencoder.pth'):
            self.sae_model.load_state_dict(torch.load('best_semantic_autoencoder.pth', map_location=self.device))
            print("Best SAE model loaded.")
        else:
            print("Warning: No best SAE model found. Using last trained model (if any).")

        self.sae_model.eval()
        return True

    def get_target_compound_semantics_for_eval(self, semantic_dict):
        """
        Retrieves the target fused semantic vectors for compound faults.
        This replaces the old 'generate_compound_fault_projections_fixed' as SAE projects features to semantics.
        The "projections" for ZSL evaluation are now these target semantic vectors themselves.
        """
        print("\n获取复合故障的目标融合语义用于评估...")
        fused_semantics = semantic_dict.get('fused_semantics')

        target_compound_semantics = {}
        for fault_type in self.compound_fault_types:
            if fault_type in fused_semantics:
                sem_vec = fused_semantics[fault_type]
                if sem_vec is not None and np.all(np.isfinite(sem_vec)) and len(sem_vec) == self.fused_semantic_dim:
                    target_compound_semantics[fault_type] = sem_vec
                else:
                    print(
                        f"警告: '{fault_type}' 的融合语义无效或维度错误 (expected {self.fused_semantic_dim}, got {len(sem_vec) if sem_vec is not None else 'None'}). 将尝试修复/重新生成。")

                    regenerated_sem = self.semantic_builder._generate_fallback_semantics([fault_type],
                                                                                         semantic_dict.get(
                                                                                             'data_prototypes', {}))
                    if fault_type in regenerated_sem and np.all(np.isfinite(regenerated_sem[fault_type])) and len(
                            regenerated_sem[fault_type]) == self.actual_ae_latent_dim:
                        k_sem = semantic_dict.get('knowledge_semantics', {}).get(fault_type)
                        if k_sem is not None and np.all(np.isfinite(k_sem)) and len(
                                k_sem) == self.semantic_builder.knowledge_dim:
                            fused_fallback = np.concatenate([k_sem, regenerated_sem[fault_type]]).astype(np.float32)
                            if np.all(np.isfinite(fused_fallback)) and len(fused_fallback) == self.fused_semantic_dim:
                                target_compound_semantics[fault_type] = fused_fallback
                                print(f"  - 成功为 '{fault_type}' 生成回退融合语义。")
                            else:
                                print(f"  - 错误: 为 '{fault_type}' 生成的回退融合语义在拼接后无效。")
                        else:
                            print(f"  - 错误: 为 '{fault_type}' 生成回退融合语义时缺少知识语义部分。")
                    else:
                        print(f"  - 错误: 无法为 '{fault_type}' 生成有效的融合语义作为目标。")
            else:
                print(f"警告: 复合故障类型 '{fault_type}' 在融合语义中未定义。")

        if not target_compound_semantics:
            print("错误: 未能获取任何有效的复合故障目标语义。")
            return None

        print(f"成功获取 {len(target_compound_semantics)} 种复合故障的目标融合语义。")
        return target_compound_semantics

    def evaluate_zero_shot_with_enhanced_attributes(self, data_dict, target_compound_semantics):
        """使用增强属性的零样本评估"""
        print("\n评估ZSL（增强属性方法）...")

        self.cnn_model.eval()
        self.sae_model.eval()
        self.semantic_builder.autoencoder.eval()

        X_test_signals, y_test_labels = data_dict['X_test'], data_dict['y_test']

        # 为每个测试信号预测最佳属性组合
        predicted_labels = []
        actual_labels = []

        attribute_combinations = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1],  # 单一故障
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]  # 复合故障
        ]

        batch_size = self.batch_size

        for i in range(0, len(X_test_signals), batch_size):
            batch_signals = X_test_signals[i:i + batch_size]
            batch_labels = y_test_labels[i:i + batch_size]

            for j, signal in enumerate(batch_signals):
                # 计算信号的频域特征
                hf_ratio, lf_ratio = self.semantic_builder.freq_extractor.extract_frequency_ratios(signal)

                # 获取AE编码
                signal_tensor = torch.FloatTensor(signal).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    ae_encoded = self.semantic_builder.autoencoder.encode(signal_tensor)

                best_similarity = -1
                best_combination_idx = 0

                # 枚举所有属性组合
                for k, combo in enumerate(attribute_combinations):
                    # 构造5维属性向量
                    attr_vector = np.array(combo + [hf_ratio, lf_ratio], dtype=np.float32)

                    # 找到对应的目标语义
                    fault_name = self._combination_to_fault_name(combo)
                    if fault_name in target_compound_semantics:
                        target_semantic = target_compound_semantics[fault_name]
                        target_tensor = torch.FloatTensor(target_semantic).unsqueeze(0).to(self.device)

                        # 计算相似度
                        similarity = F.cosine_similarity(ae_encoded, target_tensor, dim=1).item()

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_combination_idx = k

                # 转换为故障类型索引
                best_combo = attribute_combinations[best_combination_idx]
                pred_fault_name = self._combination_to_fault_name(best_combo)
                pred_label = self.fault_types.get(pred_fault_name, 0)

                predicted_labels.append(pred_label)
                actual_labels.append(batch_labels[j])

        # 计算准确率
        predicted_labels = np.array(predicted_labels)
        actual_labels = np.array(actual_labels)
        accuracy = accuracy_score(actual_labels, predicted_labels) * 100

        print(f"增强属性零样本学习准确率: {accuracy:.2f}%")
        return accuracy

    def _combination_to_fault_name(self, combination):
        """将属性组合转换为故障名称"""
        combo_to_name = {
            (1, 0, 0): 'inner',
            (0, 1, 0): 'outer',
            (0, 0, 1): 'ball',
            (0, 0, 0): 'normal',
            (1, 1, 0): 'inner_outer',
            (1, 0, 1): 'inner_ball',
            (0, 1, 1): 'outer_ball',
            (1, 1, 1): 'inner_outer_ball'
        }
        return combo_to_name.get(tuple(combination), 'normal')
    def evaluate_zero_shot_with_pca(self, data_dict, target_compound_semantics, pca_components=3):
        """使用SAE投影的语义和PCA降维后的欧氏距离进行零样本复合故障分类"""
        print(f"\n评估ZSL（PCA方法）：使用SAE投影语义，在{pca_components}维PCA空间比较...")

        self.cnn_model.eval()
        self.sae_model.eval()
        self.semantic_builder.autoencoder.eval()

        X_test_signals, y_test_labels = data_dict['X_test'], data_dict['y_test']
        valid_compound_fault_names = list(target_compound_semantics.keys())
        valid_compound_fault_indices = [self.fault_types[name] for name in valid_compound_fault_names if
                                        name in self.fault_types]

        mask_eval = np.isin(y_test_labels, valid_compound_fault_indices)
        X_eval_signals = X_test_signals[mask_eval]
        y_eval_labels = y_test_labels[mask_eval]
        reference_semantic_vectors_list = []
        reference_labels_list = []  # Store corresponding fault indices
        for name, vec in target_compound_semantics.items():
            reference_semantic_vectors_list.append(vec)
            reference_labels_list.append(self.fault_types[name])
        reference_semantic_vectors_np = np.array(reference_semantic_vectors_list)
        predicted_semantic_vectors_list = []
        actual_labels_for_pred_list = []
        batch_size_eval = self.batch_size

        with torch.no_grad():
            for i in range(0, len(X_eval_signals), batch_size_eval):
                batch_x_sig = torch.FloatTensor(X_eval_signals[i:i + batch_size_eval]).to(self.device)
                batch_y_lbl = y_eval_labels[i:i + batch_size_eval]
                ae_input = batch_x_sig.squeeze(1) if batch_x_sig.dim() == 3 and batch_x_sig.shape[
                    1] == 1 else batch_x_sig
                if ae_input.dim() != 2 or ae_input.shape[1] != self.semantic_builder.autoencoder.input_length:
                    continue
                current_ae_sem = self.semantic_builder.autoencoder.encode(ae_input)
                cnn_feats = self.cnn_model(batch_x_sig, current_ae_sem, return_features=True)
                sae_pred_sem = self.sae_model(cnn_feats)

                predicted_semantic_vectors_list.append(sae_pred_sem.cpu().numpy())
                actual_labels_for_pred_list.extend(batch_y_lbl)
        predicted_semantic_vectors_np = np.vstack(predicted_semantic_vectors_list)
        y_eval_labels_final = np.array(actual_labels_for_pred_list)
        all_semantics_for_pca = np.vstack([predicted_semantic_vectors_np, reference_semantic_vectors_np])

        actual_pca_comps = min(pca_components, all_semantics_for_pca.shape[0], all_semantics_for_pca.shape[1])


        pca = PCA(n_components=actual_pca_comps)
        pca.fit(all_semantics_for_pca)

        predicted_sem_pca = pca.transform(predicted_semantic_vectors_np)
        reference_sem_pca = pca.transform(reference_semantic_vectors_np)
        print(f"  语义空间维度: {predicted_semantic_vectors_np.shape[1]} -> PCA降维后: {predicted_sem_pca.shape[1]}")

        y_classified_indices = []
        for pred_pca_vec in predicted_sem_pca:
            dists = [np.linalg.norm(pred_pca_vec - ref_pca_vec) for ref_pca_vec in reference_sem_pca]
            nearest_idx = np.argmin(dists)
            y_classified_indices.append(reference_labels_list[nearest_idx])  # Predicted fault index

        y_classified_indices_np = np.array(y_classified_indices)

        accuracy = accuracy_score(y_eval_labels_final, y_classified_indices_np) * 100
        class_accuracy_results = {}
        print("\n=== 各类别分类详情 (PCA + 欧氏距离, 语义空间) ===")
        for fault_idx_true in np.unique(y_eval_labels_final):
            mask = (y_eval_labels_final == fault_idx_true)
            count = np.sum(mask)
            correct_count = np.sum(y_classified_indices_np[mask] == y_eval_labels_final[
                mask])  # Check if classified index matches true index
            acc = (correct_count / count) * 100 if count > 0 else 0
            fault_name = self.idx_to_fault.get(fault_idx_true, f"Unknown_{fault_idx_true}")
            class_accuracy_results[fault_name] = acc
            print(f"类别 {fault_name}: {correct_count}/{count} 正确, 准确率 {acc:.2f}%")
        print(f"\n总体准确率 (PCA, 语义空间): {accuracy:.2f}%")

        # Confusion Matrix
        true_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_eval_labels_final]
        pred_labels_str = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_classified_indices_np]
        # Use names from target_compound_semantics as the full set of labels for CM display
        cm_display_labels = sorted(list(target_compound_semantics.keys()))

        conf_matrix_val = confusion_matrix(true_labels_str, pred_labels_str, labels=cm_display_labels)

        configure_chinese_font()
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues',
                    xticklabels=cm_display_labels, yticklabels=cm_display_labels)
        plt.xlabel('预测')
        plt.ylabel('真实')
        plt.title(f'ZSL混淆矩阵 (SAE投影语义+PCA, 准确率: {accuracy:.2f}%)')
        plt.xticks(rotation=45, ha='right');
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('compound_fault_cm_zsl_sae_pca.png')
        plt.close()
        print("混淆矩阵 (SAE+PCA) 已保存至 'compound_fault_cm_zsl_sae_pca.png'")

        return accuracy, class_accuracy_results

    def evaluate_zero_shot_with_cosine_similarity(self, data_dict, target_compound_semantics):
        """使用SAE投影的语义和原始语义空间中的余弦相似度进行零样本复合故障分类"""
        print("\n评估ZSL（余弦相似度方法）：使用SAE投影语义，在原始融合语义空间比较...")

        self.cnn_model.eval()
        self.sae_model.eval()
        self.semantic_builder.autoencoder.eval()

        X_test_signals, y_test_labels = data_dict['X_test'], data_dict['y_test']

        valid_compound_fault_names = list(target_compound_semantics.keys())
        valid_compound_fault_indices = [self.fault_types[name] for name in valid_compound_fault_names if
                                        name in self.fault_types]

        mask_eval_cosine = np.isin(y_test_labels, valid_compound_fault_indices)
        X_eval_signals_cosine = X_test_signals[mask_eval_cosine]
        y_eval_labels_cosine = y_test_labels[mask_eval_cosine]
        reference_semantic_vectors_list_cos = []
        reference_labels_list_cos = []
        for name, vec in target_compound_semantics.items():
            reference_semantic_vectors_list_cos.append(vec)
            reference_labels_list_cos.append(self.fault_types[name])
        reference_semantic_vectors_np_cos = np.array(reference_semantic_vectors_list_cos)
        reference_semantic_norm_cos = reference_semantic_vectors_np_cos / (
                    np.linalg.norm(reference_semantic_vectors_np_cos, axis=1, keepdims=True) + 1e-9)
        predicted_semantic_vectors_list_cos = []
        actual_labels_for_pred_list_cos = []
        batch_size_eval = self.batch_size

        with torch.no_grad():
            for i in range(0, len(X_eval_signals_cosine), batch_size_eval):
                batch_x_sig_cos = torch.FloatTensor(X_eval_signals_cosine[i:i + batch_size_eval]).to(self.device)
                batch_y_lbl_cos = y_eval_labels_cosine[i:i + batch_size_eval]

                ae_input_cos = batch_x_sig_cos.squeeze(1) if batch_x_sig_cos.dim() == 3 and batch_x_sig_cos.shape[
                    1] == 1 else batch_x_sig_cos
                if ae_input_cos.dim() != 2 or ae_input_cos.shape[1] != self.semantic_builder.autoencoder.input_length:
                    continue
                current_ae_sem_cos = self.semantic_builder.autoencoder.encode(ae_input_cos)

                cnn_feats_cos = self.cnn_model(batch_x_sig_cos, current_ae_sem_cos, return_features=True)
                sae_pred_sem_cos = self.sae_model(cnn_feats_cos)

                predicted_semantic_vectors_list_cos.append(sae_pred_sem_cos.cpu().numpy())
                actual_labels_for_pred_list_cos.extend(batch_y_lbl_cos)

        predicted_semantic_vectors_np_cos = np.vstack(predicted_semantic_vectors_list_cos)
        y_eval_labels_final_cos = np.array(actual_labels_for_pred_list_cos)
        predicted_semantic_norm_cos = predicted_semantic_vectors_np_cos / (
                    np.linalg.norm(predicted_semantic_vectors_np_cos, axis=1, keepdims=True) + 1e-9)

        # Classification using Cosine Similarity
        similarity_matrix = np.dot(predicted_semantic_norm_cos, reference_semantic_norm_cos.T)
        y_classified_indices_cos = []
        for i in range(similarity_matrix.shape[0]):
            nearest_idx = np.argmax(similarity_matrix[i])
            y_classified_indices_cos.append(reference_labels_list_cos[nearest_idx])

        y_classified_indices_np_cos = np.array(y_classified_indices_cos)

        accuracy_cos = accuracy_score(y_eval_labels_final_cos, y_classified_indices_np_cos) * 100
        class_accuracy_cosine_results = {}
        print("\n=== 各类别分类详情 (余弦相似度, 语义空间) ===")
        for fault_idx_true_cos in np.unique(y_eval_labels_final_cos):
            mask_cos = (y_eval_labels_final_cos == fault_idx_true_cos)
            count_cos = np.sum(mask_cos)
            correct_count_cos = np.sum(y_classified_indices_np_cos[mask_cos] == y_eval_labels_final_cos[mask_cos])
            acc_cos_cls = (correct_count_cos / count_cos) * 100 if count_cos > 0 else 0
            fault_name_cos = self.idx_to_fault.get(fault_idx_true_cos, f"Unknown_{fault_idx_true_cos}")
            class_accuracy_cosine_results[fault_name_cos] = acc_cos_cls
            print(f"类别 {fault_name_cos}: {correct_count_cos}/{count_cos} 正确, 准确率 {acc_cos_cls:.2f}%")
        print(f"\n总体准确率 (余弦相似度, 语义空间): {accuracy_cos:.2f}%")

        # Confusion Matrix
        true_labels_str_cos = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_eval_labels_final_cos]
        pred_labels_str_cos = [self.idx_to_fault.get(l, f"Unk_{l}") for l in y_classified_indices_np_cos]
        cm_display_labels_cos = sorted(list(target_compound_semantics.keys()))

        conf_matrix_val_cos = confusion_matrix(true_labels_str_cos, pred_labels_str_cos, labels=cm_display_labels_cos)

        configure_chinese_font()
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_val_cos, annot=True, fmt='d', cmap='Greens',
                    xticklabels=cm_display_labels_cos, yticklabels=cm_display_labels_cos)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'ZSL混淆矩阵 (SAE投影语义+余弦相似度, 准确率: {accuracy_cos:.2f}%)')
        plt.xticks(rotation=45, ha='right');
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('compound_fault_cm_zsl_sae_cosine.png')
        plt.close()
        print("混淆矩阵 (SAE+余弦) 已保存至 'compound_fault_cm_zsl_sae_cosine.png'")

        return accuracy_cos, class_accuracy_cosine_results

    def visualize_semantic_space(self, data_dict, semantic_dict):
        """
        可视化单一故障和复合故障在自编码器语义空间的分布 (AE latent space from FaultSemanticBuilder)
        """
        print("\n==== 可视化AE语义空间分布 (FaultSemanticBuilder的潜空间) ====")

        all_features = self.semantic_builder.all_latent_features
        all_labels = self.semantic_builder.all_latent_labels
        valid_indices = np.all(np.isfinite(all_features), axis=1)
        all_features = all_features[valid_indices]
        all_labels = all_labels[valid_indices]
        max_samples_viz = 2000
        if len(all_features) > max_samples_viz:
            sample_indices = np.random.choice(len(all_features), max_samples_viz, replace=False)
            all_features = all_features[sample_indices]
            all_labels = all_labels[sample_indices]

        pca = PCA(n_components=3)  # Ensure 3 components for 3D plot
        features_reduced_pca = pca.fit_transform(all_features)

        fault_names_pca = [self.idx_to_fault.get(label, f"Unknown_{label}") for label in all_labels]
        fault_labels_unique_pca = sorted(list(set(fault_names_pca)))

        colors_pca = sns.color_palette('husl', n_colors=len(fault_labels_unique_pca))
        color_map_pca = {fault: colors_pca[i] for i, fault in enumerate(fault_labels_unique_pca)}
        marker_map_pca = {fault: 'o' if '_' not in fault else '^' for fault in fault_labels_unique_pca}

        configure_chinese_font()

        # 2D PCA
        plt.figure(figsize=(12, 10))
        for fault in fault_labels_unique_pca:
            mask = np.array(fault_names_pca) == fault
            plt.scatter(features_reduced_pca[mask, 0], features_reduced_pca[mask, 1],
                        c=[color_map_pca[fault]], marker=marker_map_pca[fault], label=fault, alpha=0.6, s=40)
        plt.title('AE潜语义空间 (PCA 2D)')
        plt.xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.legend(loc='best', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.grid(True, linestyle='--', alpha=0.7);
        plt.tight_layout()
        plt.savefig('ae_latent_semantic_space_pca2d.png');
        plt.close()

        # 3D PCA
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        for fault in fault_labels_unique_pca:
            mask = np.array(fault_names_pca) == fault
            ax.scatter(features_reduced_pca[mask, 0], features_reduced_pca[mask, 1], features_reduced_pca[mask, 2],
                       c=[color_map_pca[fault]], marker=marker_map_pca[fault], label=fault, alpha=0.6, s=40)
        ax.set_title('AE潜语义空间 (PCA 3D)')
        ax.set_xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PCA3 ({pca.explained_variance_ratio_[2]:.2%})')
        ax.legend(loc='best', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.tight_layout();
        plt.savefig('ae_latent_semantic_space_pca3d.png');
        plt.close()
        tsne = TSNE(n_components=2, perplexity=min(30, len(all_features) - 1 if len(all_features) > 1 else 1),
                    n_iter=300, random_state=42, learning_rate='auto', init='pca')  # Reduced n_iter for speed
        if len(all_features) > 1:  # tSNE needs more than 1 sample
            features_tsne = tsne.fit_transform(all_features)
            plt.figure(figsize=(12, 10))
            for fault in fault_labels_unique_pca:  # Use same labels and colors
                mask = np.array(fault_names_pca) == fault
                plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                            c=[color_map_pca[fault]], marker=marker_map_pca[fault], label=fault, alpha=0.6, s=40)
            plt.title('AE潜语义空间 (t-SNE 2D)')
            plt.xlabel('t-SNE Dim 1');
            plt.ylabel('t-SNE Dim 2')
            plt.legend(loc='best', bbox_to_anchor=(1.01, 1), borderaxespad=0)
            plt.grid(True, linestyle='--', alpha=0.7);
            plt.tight_layout()
            plt.savefig('ae_latent_semantic_space_tsne.png');
            plt.close()
            print("AE潜语义空间可视化图 (PCA 2D/3D, t-SNE 2D) 已保存。")
        else:
            print("样本过少，跳过AE潜语义空间t-SNE可视化。")

    def visualize_sae_semantic_space(self, data_dict, semantic_dict):
        """
        可视化SAE投影后的融合语义空间。
        - 输入: CNN特征 (X^S)
        - 通过SAE投影: A_pred = SAE(X^S)
        - 可选: 叠印目标融合语义 A_target
        """
        print("\n==== 可视化SAE投影语义空间 ====")

        self.cnn_model.eval()
        self.sae_model.eval()
        self.semantic_builder.autoencoder.eval()

        X_train_signals, y_train_labels = data_dict['X_train'], data_dict['y_train']
        X_test_signals, y_test_labels = data_dict['X_test'], data_dict['y_test']
        max_samples_per_class_viz = 100

        viz_signals_list = []
        viz_labels_list = []

        # Single faults from training set
        for ft_idx in [self.fault_types[name] for name in self.single_fault_types_ordered]:
            mask = y_train_labels == ft_idx
            signals = X_train_signals[mask][:max_samples_per_class_viz]
            labels = y_train_labels[mask][:max_samples_per_class_viz]
            if len(signals) > 0:
                viz_signals_list.append(signals)
                viz_labels_list.append(labels)

        # Compound faults from test set
        for ft_idx in [self.fault_types[name] for name in self.compound_fault_types]:
            mask = y_test_labels == ft_idx
            signals = X_test_signals[mask][:max_samples_per_class_viz]
            labels = y_test_labels[mask][:max_samples_per_class_viz]
            if len(signals) > 0:
                viz_signals_list.append(signals)
                viz_labels_list.append(labels)

        if not viz_signals_list:
            print("SAE可视化：无可用信号数据。")
            return

        X_viz_signals = np.vstack(viz_signals_list)
        y_viz_labels = np.concatenate(viz_labels_list)
        sae_projected_semantics_list = []
        with torch.no_grad():
            bs = self.batch_size
            for i in range(0, len(X_viz_signals), bs):
                batch_sig = torch.FloatTensor(X_viz_signals[i:i + bs]).to(self.device)
                ae_in = batch_sig.squeeze(1) if batch_sig.dim() == 3 and batch_sig.shape[1] == 1 else batch_sig
                if ae_in.dim() != 2 or ae_in.shape[1] != self.semantic_builder.autoencoder.input_length: continue

                ae_sem_for_cnn = self.semantic_builder.autoencoder.encode(ae_in)
                cnn_feats = self.cnn_model(batch_sig, ae_sem_for_cnn, return_features=True)
                sae_proj_sem = self.sae_model(cnn_feats)
                sae_projected_semantics_list.append(sae_proj_sem.cpu().numpy())


        sae_projected_semantics_np = np.vstack(sae_projected_semantics_list)
        target_fused_semantics_map = semantic_dict.get('fused_semantics', {})
        target_semantics_for_viz_list = []
        target_labels_for_viz_list = []
        for ft_name, ft_vec in target_fused_semantics_map.items():
            if ft_vec is not None and np.all(np.isfinite(ft_vec)):
                target_semantics_for_viz_list.append(ft_vec)
                target_labels_for_viz_list.append(self.fault_types[ft_name])
        target_semantics_for_viz_np = np.array(target_semantics_for_viz_list)
        combined_semantics_for_dim_reduction = np.vstack([sae_projected_semantics_np, target_semantics_for_viz_np])

        pca_sae = PCA(n_components=3)
        reduced_combined_pca = pca_sae.fit_transform(combined_semantics_for_dim_reduction)

        num_projected = len(sae_projected_semantics_np)
        reduced_projected_pca = reduced_combined_pca[:num_projected]
        reduced_target_pca = reduced_combined_pca[num_projected:]

        # Plotting
        configure_chinese_font()
        fault_names_viz = [self.idx_to_fault.get(lbl, f"Unk_{lbl}") for lbl in y_viz_labels]
        unique_fault_names_viz = sorted(list(set(fault_names_viz)))

        colors_viz = sns.color_palette('tab10', n_colors=len(unique_fault_names_viz))
        color_map_viz = {name: colors_viz[i] for i, name in enumerate(unique_fault_names_viz)}

        # 2D PCA plot for SAE projected semantics
        plt.figure(figsize=(13, 10))
        for name in unique_fault_names_viz:
            mask = np.array(fault_names_viz) == name
            plt.scatter(reduced_projected_pca[mask, 0], reduced_projected_pca[mask, 1],
                        c=[color_map_viz[name]], label=f"Proj: {name}", alpha=0.7, s=50,
                        marker='o' if '_' not in name else 's')

        # Overlay target semantics
        target_fault_names_viz = [self.idx_to_fault.get(lbl, f"Unk_{lbl}") for lbl in target_labels_for_viz_list]
        for i, name in enumerate(target_fault_names_viz):
            if name in color_map_viz:  # Only plot if it's one of the visualized fault types
                plt.scatter(reduced_target_pca[i, 0], reduced_target_pca[i, 1],
                            c=[color_map_viz[name]],
                            label=f"Target: {name}" if name not in plt.gca().get_legend_handles_labels()[1] else None,
                            edgecolors='k', marker='X', s=150, linewidths=1.5)

        plt.title('SAE投影语义空间 vs 目标语义 (PCA 2D)')
        plt.xlabel(f'PCA1 ({pca_sae.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PCA2 ({pca_sae.explained_variance_ratio_[1]:.2%})')
        plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1)
        plt.grid(True, linestyle='--', alpha=0.6);
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust for legend
        plt.savefig('sae_projected_vs_target_semantic_pca2d.png');
        plt.close()

        # t-SNE
        tsne_sae_perplexity = min(30, combined_semantics_for_dim_reduction.shape[0] - 1)
        if tsne_sae_perplexity > 0:
            tsne_sae = TSNE(n_components=2, perplexity=tsne_sae_perplexity, n_iter=300, random_state=42,
                            learning_rate='auto', init='pca')
            reduced_combined_tsne = tsne_sae.fit_transform(combined_semantics_for_dim_reduction)
            reduced_projected_tsne = reduced_combined_tsne[:num_projected]
            reduced_target_tsne = reduced_combined_tsne[num_projected:]

            plt.figure(figsize=(13, 10))
            for name in unique_fault_names_viz:
                mask = np.array(fault_names_viz) == name
                plt.scatter(reduced_projected_tsne[mask, 0], reduced_projected_tsne[mask, 1],
                            c=[color_map_viz[name]], label=f"Proj: {name}", alpha=0.7, s=50,
                            marker='o' if '_' not in name else 's')
            for i, name in enumerate(target_fault_names_viz):
                if name in color_map_viz:
                    plt.scatter(reduced_target_tsne[i, 0], reduced_target_tsne[i, 1],
                                c=[color_map_viz[name]],
                                label=f"Target: {name}" if name not in plt.gca().get_legend_handles_labels()[
                                    1] else None,
                                edgecolors='k', marker='X', s=150, linewidths=1.5)
            plt.title('SAE投影语义空间 vs 目标语义 (t-SNE 2D)')
            plt.xlabel('t-SNE Dim 1');
            plt.ylabel('t-SNE Dim 2')
            plt.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1)
            plt.grid(True, linestyle='--', alpha=0.6);
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig('sae_projected_vs_target_semantic_tsne.png');
            plt.close()
            print("SAE投影语义空间可视化图 (PCA & t-SNE) 已保存。")
        else:
            print("样本过少，跳过SAE投影语义空间t-SNE可视化。")

    def visualize_cnn_feature_distribution(self, data_dict, max_samples_per_class=200):
        """
        可视化CNN模型提取的特征分布 (SAE的输入)
        """
        print("\n==== 可视化CNN特征空间分布 (SAE输入) ====")


        self.cnn_model.eval()
        self.semantic_builder.autoencoder.eval()

        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_test, y_test = data_dict['X_test'], data_dict['y_test']

        single_idxs = [self.fault_types[k] for k in self.single_fault_types_ordered]
        comp_idxs = [self.fault_types[k] for k in self.compound_fault_types]

        def sample_by_label(X, y, label_list, max_s):
            samples_l, labels_l = [], []
            for lbl_idx in label_list:
                idxs = np.where(y == lbl_idx)[0]
                if len(idxs) == 0: continue
                choose = np.random.choice(idxs, min(len(idxs), max_s), replace=False)
                if X[choose].shape[0] > 0:  # Ensure something is chosen
                    samples_l.append(X[choose])
                    labels_l.extend([lbl_idx] * len(choose))
            # Check if samples_l is empty before vstack
            return np.vstack(samples_l) if samples_l else np.empty((0, X.shape[1])), np.array(labels_l)

        X_single, y_single = sample_by_label(X_train, y_train, single_idxs, max_samples_per_class)
        X_comp, y_comp = sample_by_label(X_test, y_test, comp_idxs, max_samples_per_class)


        X_all_cnn = []
        y_all_cnn = []
        if X_single.shape[0] > 0:
            X_all_cnn.append(X_single)
            y_all_cnn.append(y_single)
        if X_comp.shape[0] > 0:
            X_all_cnn.append(X_comp)
            y_all_cnn.append(y_comp)



        X_all_cnn = np.vstack(X_all_cnn)
        y_all_cnn = np.concatenate(y_all_cnn)

        names_all_cnn = [self.idx_to_fault.get(i, f"Unk_{i}") for i in y_all_cnn]

        cnn_features_extracted = []
        batch_sz_cnn_viz = self.batch_size
        with torch.no_grad():
            for i in range(0, len(X_all_cnn), batch_sz_cnn_viz):
                xbatch_sig = torch.FloatTensor(X_all_cnn[i:i + batch_sz_cnn_viz]).to(self.device)
                ae_in_cnn = xbatch_sig.squeeze(1) if xbatch_sig.dim() == 3 and xbatch_sig.shape[1] == 1 else xbatch_sig
                if ae_in_cnn.dim() != 2 or ae_in_cnn.shape[
                    1] != self.semantic_builder.autoencoder.input_length: continue

                sem_for_cnn = self.semantic_builder.autoencoder.encode(ae_in_cnn)
                feat_cnn = self.cnn_model(xbatch_sig, sem_for_cnn, return_features=True)
                cnn_features_extracted.append(feat_cnn.cpu().numpy())


        feats_cnn = np.vstack(cnn_features_extracted)

        configure_chinese_font()
        palette_cnn = sns.color_palette('Spectral', n_colors=len(set(names_all_cnn)))  # Changed palette

        # PCA 2D
        pca_cnn = PCA(n_components=2)
        pca2_cnn = pca_cnn.fit_transform(feats_cnn)
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=pca2_cnn[:, 0], y=pca2_cnn[:, 1], hue=names_all_cnn,
                        palette=palette_cnn, s=60, alpha=0.8, style=names_all_cnn,
                        markers={name: ('o' if '_' not in name else '^') for name in set(names_all_cnn)})
        plt.title('CNN 特征空间 PCA 2D 投影 (SAE输入)')
        plt.legend(loc='best', bbox_to_anchor=(1, 1));
        plt.tight_layout()
        plt.savefig('cnn_feature_space_pca2d_sae_input.png');
        plt.close()

        # t-SNE 2D
        tsne_cnn_perplexity = min(30, feats_cnn.shape[0] - 1 if feats_cnn.shape[0] > 1 else 1)
        if tsne_cnn_perplexity > 0:
            tsne_cnn = TSNE(n_components=2, perplexity=tsne_cnn_perplexity, n_iter=300, random_state=42,
                            learning_rate='auto', init='pca')
            tsne2_cnn = tsne_cnn.fit_transform(feats_cnn)
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x=tsne2_cnn[:, 0], y=tsne2_cnn[:, 1], hue=names_all_cnn,
                            palette=palette_cnn, s=60, alpha=0.8, style=names_all_cnn,
                            markers={name: ('o' if '_' not in name else '^') for name in set(names_all_cnn)})
            plt.title('CNN 特征空间 t-SNE 2D 投影 (SAE输入)')
            plt.legend(loc='best', bbox_to_anchor=(1, 1));
            plt.tight_layout()
            plt.savefig('cnn_feature_space_tsne2d_sae_input.png');
            plt.close()
            print("CNN 特征空间可视化 (SAE输入) 完成 (PCA & t-SNE)。")
        else:
            print("CNN特征可视化：样本过少，跳过t-SNE。")

    def run_pipeline(self):
        """基于语义自编码器(SAE)的零样本复合故障诊断流水线"""
        start_time = time.time()
        print("\n==== 步骤1: 数据加载 ====")
        data_dict = self.load_data()

        print("\n==== 步骤2: 语义构建 (AE训练和融合语义生成) ====")
        semantic_dict = self.build_semantics(data_dict)

        self.actual_ae_latent_dim = self.semantic_builder.actual_latent_dim
        self.fused_semantic_dim = self.semantic_builder.knowledge_dim + self.actual_ae_latent_dim

        self.visualize_semantic_space(data_dict, semantic_dict)

        print("\n==== 步骤3: 初始化并训练CNN模型 (带FiLM调制) ====")

        self.cnn_model = self.train_ae_data_semantic_cnn(
            data_dict=data_dict,
            semantic_dict=semantic_dict,
            epochs=CNN_EPOCHS,
            batch_size=self.batch_size,  # Use class batch_size
            lr=CNN_LR
        )


        self.visualize_cnn_feature_distribution(data_dict)

        print("\n==== 步骤4: 训练语义自编码器 (SAE) ====")
        sae_train_success = self.train_semantic_autoencoder(
            data_dict=data_dict,
            semantic_dict=semantic_dict,
            epochs=SAE_EPOCHS,
            lr=SAE_LR

        )
        self.visualize_sae_semantic_space(data_dict, semantic_dict)

        print("\n==== 步骤5: 获取复合故障的目标融合语义 (用于ZSL评估) ====")
        target_compound_semantics = self.get_target_compound_semantics_for_eval(semantic_dict)
        if not target_compound_semantics:
            print("未能获取复合故障的目标语义，无法进行ZSL评估，终止流水线。")
            return 0.0

        print("\n==== 步骤6: 零样本学习评估 ====")
        print("--- 评估方法1: SAE投影语义 + PCA降维 + 欧氏距离 ---")
        accuracy_pca, _ = self.evaluate_zero_shot_with_pca(
            data_dict,
            target_compound_semantics,
            pca_components=min(3, self.fused_semantic_dim)
        )

        print("\n--- 评估方法2: SAE投影语义 + 原始融合语义空间 + 余弦相似度 ---")
        accuracy_cosine, _ = self.evaluate_zero_shot_with_cosine_similarity(
            data_dict,
            target_compound_semantics
        )
        final_accuracy_to_report = accuracy_pca

        end_time = time.time()
        print(f"\n=== 流水线在 {(end_time - start_time) / 60:.2f} 分钟内完成 ===")
        print(f"零样本学习准确率 (PCA方法, 语义空间): {accuracy_pca:.2f}%")
        print(f"零样本学习准确率 (余弦相似度方法, 语义空间): {accuracy_cosine:.2f}%")

        return final_accuracy_to_report


if __name__ == "__main__":
    set_seed(42)
    data_path = "E:/研究生/CNN/HDU600D"

    if not os.path.isdir(data_path):
        if data_path == "./HDU600D" and not os.path.exists("./HDU600D"):
            os.makedirs("./HDU600D", exist_ok=True)
            dummy_data = {'normal_data': np.random.rand(100000)}
            sio.savemat("./HDU600D/normal.mat", dummy_data)

    if os.path.isdir(data_path):
        fault_diagnosis = ZeroShotCompoundFaultDiagnosis(
            data_path=data_path,
            sample_length=SEGMENT_LENGTH,
            latent_dim=AE_LATENT_DIM,  # AE's latent dim
            batch_size=DEFAULT_BATCH_SIZE,
            sae_mu=SAE_MU
        )
        final_accuracy = fault_diagnosis.run_pipeline()
        print(f"\n>>> 最终零样本诊断准确率 (基于PCA方法报告): {final_accuracy:.2f}% <<<")
    else:
        print("由于数据目录问题，流水线未运行。")
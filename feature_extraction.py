#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
零样本复合轴承故障诊断
实现基于知识语义和数据语义的零样本学习方法，用于复合轴承故障诊断
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, interpolate
from scipy.stats import iqr
import pywt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# 设置随机种子，确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 轴承参数
class BearingParameters:
    def __init__(self):
        self.inner_diameter = 17  # mm
        self.outer_diameter = 40  # mm
        self.width = 12  # mm
        self.ball_diameter = 6.75  # mm
        self.ball_number = 9
        self.contact_angle = 0  # 深沟球轴承接触角为0

        # 实验条件
        self.sampling_freq = 10000  # Hz
        self.load = 1000  # N
        self.rotation_speed = 1000  # rpm
        self.rotation_freq = self.rotation_speed / 60  # Hz

        # 计算特征频率
        self.calc_characteristic_frequencies()

    def calc_characteristic_frequencies(self):
        """计算轴承特征频率"""
        # 轴承节径
        pitch_diameter = (self.outer_diameter + self.inner_diameter) / 2

        # 接触角的余弦值
        cos_alpha = np.cos(np.radians(self.contact_angle))

        # 计算特征频率比例系数
        ball_pass_inner = self.ball_number / 2 * (1 + self.ball_diameter * cos_alpha / pitch_diameter)
        ball_pass_outer = self.ball_number / 2 * (1 - self.ball_diameter * cos_alpha / pitch_diameter)
        ball_spin = pitch_diameter / self.ball_diameter * (1 - (self.ball_diameter * cos_alpha / pitch_diameter) ** 2)
        cage = 1 / 2 * (1 - self.ball_diameter * cos_alpha / pitch_diameter)

        # 特征频率（相对于轴转速的倍数）
        self.bpfi = ball_pass_inner  # 内圈故障特征频率
        self.bpfo = ball_pass_outer  # 外圈故障特征频率
        self.bsf = ball_spin  # 滚动体故障特征频率
        self.ftf = cage  # 保持架故障特征频率

        # 实际特征频率（Hz）
        self.bpfi_hz = self.bpfi * self.rotation_freq
        self.bpfo_hz = self.bpfo * self.rotation_freq
        self.bsf_hz = self.bsf * self.rotation_freq
        self.ftf_hz = self.ftf * self.rotation_freq

        print(f"内圈故障特征频率: {self.bpfi_hz:.2f} Hz")
        print(f"外圈故障特征频率: {self.bpfo_hz:.2f} Hz")
        print(f"滚动体故障特征频率: {self.bsf_hz:.2f} Hz")
        print(f"保持架故障特征频率: {self.ftf_hz:.2f} Hz")


# 数据预处理类
class DataPreprocessor:
    def __init__(self, sample_length=10000):
        self.sample_length = sample_length

    def load_data(self, file_path):
        """加载CSV数据文件"""
        try:
            df = pd.read_csv(file_path, header=None)
            return df.values.flatten()
        except Exception as e:
            print(f"加载数据文件 {file_path} 时出错: {e}")
            return None

    def cubic_spline_interpolation(self, signal_data):
        """使用三次样条插值法处理缺失值"""
        # 检测缺失值
        missing_indices = np.isnan(signal_data)
        if not np.any(missing_indices):
            return signal_data

        # 获取非缺失值的索引和值
        valid_indices = np.where(~missing_indices)[0]
        valid_values = signal_data[valid_indices]

        # 所有索引
        all_indices = np.arange(len(signal_data))

        # 使用三次样条插值
        spline = interpolate.CubicSpline(valid_indices, valid_values)
        interpolated_signal = spline(all_indices)

        return interpolated_signal

    def remove_outliers(self, signal_data):
        """结合3σ准则与箱线图方法(IQR法则)剔除异常值"""
        # 计算均值和标准差
        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)

        # 3σ准则
        sigma_lower = mean_val - 3 * std_val
        sigma_upper = mean_val + 3 * std_val

        # 箱线图方法(IQR法则)
        q1 = np.percentile(signal_data, 25)
        q3 = np.percentile(signal_data, 75)
        iqr_val = iqr(signal_data)
        iqr_lower = q1 - 1.5 * iqr_val
        iqr_upper = q3 + 1.5 * iqr_val

        # 取两种方法的交集作为最终阈值
        lower_bound = max(sigma_lower, iqr_lower)
        upper_bound = min(sigma_upper, iqr_upper)

        # 替换异常值为边界值
        signal_data = np.clip(signal_data, lower_bound, upper_bound)

        return signal_data

    def wavelet_denoising(self, signal_data, wavelet='db4', level=4):
        """小波分解去噪"""
        # 小波分解
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)

        # 计算阈值（改进的SUREShrink阈值）
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        N = len(signal_data)

        # 对每个细节系数应用软阈值
        for j in range(1, len(coeffs)):
            # 计算层相关的阈值
            threshold = sigma * np.sqrt(2 * np.log(N)) * (1 / (1 + np.sqrt(np.log2(j + 1))))
            coeffs[j] = pywt.threshold(coeffs[j], threshold, mode='soft')

        # 小波重构
        denoised_signal = pywt.waverec(coeffs, wavelet)

        # 确保长度一致
        if len(denoised_signal) > len(signal_data):
            denoised_signal = denoised_signal[:len(signal_data)]

        return denoised_signal

    def butterworth_filter(self, signal_data, cutoff=10, order=4, fs=10000):
        """低通巴特沃斯滤波器"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = signal.filtfilt(b, a, signal_data)
        return filtered_signal

    def add_gaussian_noise(self, signal_data, snr_db=10):
        """添加高斯白噪声"""
        # 计算信号功率
        signal_power = np.mean(signal_data ** 2)

        # 计算噪声功率
        noise_power = signal_power / (10 ** (snr_db / 10))

        # 生成噪声
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal_data))

        # 添加噪声
        noisy_signal = signal_data + noise

        return noisy_signal

    def time_domain_augmentation(self, signal_data):
        """时域扩增：随机时移和随机缩放"""
        # 随机时移
        shift_percent = np.random.uniform(-0.05, 0.05)
        shift_samples = int(shift_percent * len(signal_data))
        shifted_signal = np.roll(signal_data, shift_samples)

        # 随机缩放
        scale_factor = np.random.uniform(0.95, 1.05)
        scaled_signal = signal_data * scale_factor

        return scaled_signal

    def segment_signal(self, signal_data):
        """将信号分割为固定长度的样本"""
        # 确保信号长度足够
        if len(signal_data) < self.sample_length:
            # 如果信号长度不足，则通过重复信号进行填充
            repeat_times = int(np.ceil(self.sample_length / len(signal_data)))
            signal_data = np.tile(signal_data, repeat_times)[:self.sample_length]

        # 计算可以分割的样本数量
        num_segments = len(signal_data) // self.sample_length

        # 分割信号
        segments = []
        for i in range(num_segments):
            start_idx = i * self.sample_length
            end_idx = start_idx + self.sample_length
            segment = signal_data[start_idx:end_idx]
            segments.append(segment)

        return np.array(segments)

    def preprocess_signal(self, signal_data, augment=False):
        """完整的信号预处理流程"""
        # 缺失值处理
        signal_data = self.cubic_spline_interpolation(signal_data)

        # 异常值剔除
        signal_data = self.remove_outliers(signal_data)

        # 小波分解去噪
        signal_data = self.wavelet_denoising(signal_data)

        # 低通巴特沃斯滤波
        signal_data = self.butterworth_filter(signal_data)

        # 数据增强（仅在训练时使用）
        if augment:
            # 添加高斯白噪声
            snr_db = np.random.uniform(5, 15)
            signal_data = self.add_gaussian_noise(signal_data, snr_db)

            # 时域扩增
            signal_data = self.time_domain_augmentation(signal_data)

        return signal_data

    def prepare_dataset(self, file_paths, labels, augment=False, augment_times=3):
        """准备数据集"""
        X = []
        y = []

        for file_path, label in zip(file_paths, labels):
            # 加载数据
            signal_data = self.load_data(file_path)
            if signal_data is None:
                continue

            # 预处理信号
            signal_data = self.preprocess_signal(signal_data, augment=False)

            # 分割信号
            segments = self.segment_signal(signal_data)

            # 添加到数据集
            X.append(segments)
            y.extend([label] * len(segments))

            # 数据增强
            if augment:
                for _ in range(augment_times):
                    # 预处理信号（包含增强）
                    augmented_signal = self.preprocess_signal(signal_data, augment=True)

                    # 分割信号
                    augmented_segments = self.segment_signal(augmented_signal)

                    # 添加到数据集
                    X.append(augmented_segments)
                    y.extend([label] * len(augmented_segments))

        # 合并所有分段
        X = np.vstack(X)
        y = np.array(y)

        return X, y


# 故障语义构建类
class FaultSemanticBuilder:
    def __init__(self, latent_dim=64):
        self.latent_dim = latent_dim
        self.fault_types = {
            'normal': [0, 0, 0],
            'inner': [1, 0, 0],
            'outer': [0, 1, 0],
            'ball': [0, 0, 1],
            'inner_ball': [1, 0, 1],
            'inner_outer': [1, 1, 0],
            'outer_ball': [0, 1, 1],
            'inner_outer_ball': [1, 1, 1]
        }

        # 轴承参数归一化
        bearing_params = BearingParameters()
        self.normalized_params = {
            'inner_diameter': bearing_params.inner_diameter / 100,  # 归一化到[0,1]
            'outer_diameter': bearing_params.outer_diameter / 100,
            'ball_diameter': bearing_params.ball_diameter / 100,
            'ball_number': bearing_params.ball_number / 20,  # 假设最大球数为20
            'bpfi': bearing_params.bpfi / 10,  # 归一化特征频率
            'bpfo': bearing_params.bpfo / 10,
            'bsf': bearing_params.bsf / 10
        }

        # 自编码器
        self.autoencoder = None

    def build_knowledge_semantic(self, fault_type):
        """构建知识语义向量"""
        # 基础故障位置编码
        fault_code = self.fault_types.get(fault_type, [0, 0, 0])

        # 添加轴承参数信息
        params = list(self.normalized_params.values())

        # 合并故障编码和参数信息
        knowledge_semantic = np.array(fault_code + params)

        return knowledge_semantic

    def build_data_semantic(self, signal_data, train=True):
        """使用自编码器构建数据语义向量"""
        if self.autoencoder is None or train:
            # 创建并训练自编码器
            self._train_autoencoder(signal_data)

        # 使用编码器提取特征
        with torch.no_grad():
            signal_tensor = torch.FloatTensor(signal_data).to(device)
            if len(signal_tensor.shape) == 2:
                signal_tensor = signal_tensor.unsqueeze(1)  # 添加通道维度
            data_semantic = self.autoencoder.encoder(signal_tensor).cpu().numpy()

        return data_semantic

    def _train_autoencoder(self, signal_data):
        """训练自编码器"""
        # 创建自编码器
        input_size = signal_data.shape[1]
        self.autoencoder = Autoencoder(input_size, self.latent_dim).to(device)

        # 准备数据
        signal_tensor = torch.FloatTensor(signal_data).to(device)
        if len(signal_tensor.shape) == 2:
            signal_tensor = signal_tensor.unsqueeze(1)  # 添加通道维度

        # 训练参数
        batch_size = 64
        epochs = 50
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 训练循环
        self.autoencoder.train()
        for epoch in range(epochs):
            # 随机打乱数据
            indices = torch.randperm(signal_tensor.size(0))
            total_loss = 0

            # 批次训练
            for i in range(0, signal_tensor.size(0), batch_size):
                # 获取批次数据
                idx = indices[i:i + batch_size]
                batch = signal_tensor[idx]

                # 前向传播
                optimizer.zero_grad()
                encoded, decoded = self.autoencoder(batch)

                # 计算损失
                # 重构损失
                recon_loss = criterion(decoded, batch)

                # 周期性信号的重构一致性约束
                # 对于振动信号，我们期望重构信号保持原始信号的周期性特征
                fft_orig = torch.fft.rfft(batch.squeeze(1), dim=1)
                fft_recon = torch.fft.rfft(decoded.squeeze(1), dim=1)
                consistency_loss = F.mse_loss(torch.abs(fft_orig), torch.abs(fft_recon))

                # 总损失
                loss = recon_loss + 0.1 * consistency_loss

                # 反向传播
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(indices):.6f}")

        # 设置为评估模式
        self.autoencoder.eval()

    def spectral_clustering(self, data_semantics, n_clusters=4):
        """对数据语义进行谱聚类"""
        # 应用谱聚类
        clustering = SpectralClustering(n_clusters=n_clusters,
                                        assign_labels='discretize',
                                        random_state=42,
                                        affinity='nearest_neighbors')
        cluster_labels = clustering.fit_predict(data_semantics)

        # 计算每个簇的中心
        cluster_centers = []
        for i in range(n_clusters):
            cluster_data = data_semantics[cluster_labels == i]
            center = np.mean(cluster_data, axis=0)
            cluster_centers.append(center)

        return np.array(cluster_centers), cluster_labels

    def synthesize_compound_fault_semantic(self, fault_semantics):
        """合成复合故障语义"""
        # 使用max运算合成复合故障语义
        compound_semantic = np.max(fault_semantics, axis=0)
        return compound_semantic


# 自编码器模型
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_dim):
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * (input_size // 16), latent_dim)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * (input_size // 16)),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, input_size // 16)),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # 输出范围限制在[-1, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# 语义嵌入网络
class SemanticEmbeddingNetwork(nn.Module):
    def __init__(self, semantic_dim, feature_dim):
        super(SemanticEmbeddingNetwork, self).__init__()

        # 三层MLP结构
        self.mlp = nn.Sequential(
            nn.Linear(semantic_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, feature_dim)
        )

        # 残差连接
        self.residual = nn.Linear(semantic_dim, feature_dim)

    def forward(self, x):
        # 主路径
        main_path = self.mlp(x)

        # 残差路径
        res_path = self.residual(x)

        # 残差连接
        output = main_path + res_path

        return output


# 注意力融合模块
class AttentionFusion(nn.Module):
    def __init__(self, knowledge_dim, data_dim, fused_dim=64):  # Added fused_dim
        super(AttentionFusion, self).__init__()

        # Projection layers to align dimensions
        self.knowledge_projection = nn.Linear(knowledge_dim, fused_dim)
        self.data_projection = nn.Linear(data_dim, fused_dim)

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(fused_dim * 2, 128),  # Input is concatenated projected semantics
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, knowledge_semantic, data_semantic):
        # Project semantics to common dimension
        knowledge_semantic = self.knowledge_projection(knowledge_semantic)
        data_semantic = self.data_projection(data_semantic)

        # Concatenate projected semantics
        combined = torch.cat([knowledge_semantic, data_semantic], dim=1)

        # Compute attention weights
        weights = self.attention(combined)

        # Weighted fusion
        knowledge_weight = weights[:, 0].unsqueeze(1)
        data_weight = weights[:, 1].unsqueeze(1)

        fused_semantic = knowledge_weight * knowledge_semantic + data_weight * data_semantic

        return fused_semantic, weights


# 双通道卷积网络
class DualChannelCNN(nn.Module):
    def __init__(self, input_size, semantic_dim, num_classes):
        super(DualChannelCNN, self).__init__()

        # 通道1：原始信号处理
        self.channel1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=14, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        # 计算通道1输出特征维度
        self.channel1_output_dim = self._get_channel1_output_dim(input_size)

        # 通道2：数据语义处理
        self.channel2_fc = nn.Sequential(
            nn.Linear(semantic_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 通道注意力（SE模块）
        self.se_module = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.Sigmoid()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(self.channel1_output_dim + 256, 512),  # Corrected to match actual dimension
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def _get_channel1_output_dim(self, input_size):
        size = input_size
        size = (size + 2 * 15 - 31) // 2 + 1  # Conv1
        size = size // 2  # Pool1
        size = (size + 2 * 14 - 29) // 1 + 1  # Conv2, dilation=2, effective kernel=29
        size = size // 2  # Pool2
        size = (size + 2 * 3 - 7) // 1 + 1  # Conv3
        size = size // 2  # Pool3
        size = (size + 2 * 1 - 3) // 1 + 1  # Conv4
        size = size // 2  # Pool4
        return 128 * size

    def forward(self, signal, semantic):
        # 通道1：处理原始信号
        x1 = self.channel1(signal)
        # 通道2：处理数据语义
        x2 = self.channel2_fc(semantic)
        # 应用通道注意力
        attention_weights = self.se_module(x2)
        x2 = x2 * attention_weights
        # 特征融合
        x = torch.cat([x1, x2], dim=1)
        output = self.fusion(x)
        return output


# 自定义损失函数
class CompositeLoss(nn.Module):
    def __init__(self, margin=0.5, temperature=0.07):
        super(CompositeLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, features, semantics, outputs, targets, epoch=0):
        # 交叉熵损失
        ce_loss = self.ce_loss(outputs, targets)

        # 特征-语义一致性损失
        cos_sim = F.cosine_similarity(features, semantics, dim=1).mean()
        consistency_loss = 1.0 - cos_sim + 0.01 * torch.norm(features, p=2, dim=1).mean()



        # 改进的对比损失
        batch_size = features.size(0)
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature

        # 创建标签掩码
        mask = torch.eq(targets.unsqueeze(1), targets.unsqueeze(0))

        # 对角线掩码
        diag_mask = torch.eye(batch_size, device=features.device)

        # 正样本掩码（同类别但非自身）
        pos_mask = mask.float() - diag_mask

        # 负样本掩码（不同类别）
        neg_mask = 1.0 - mask.float()

        # 计算正样本对的损失
        pos_similarity = similarity_matrix * pos_mask
        pos_similarity = pos_similarity.sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)

        # 计算负样本对的损失
        neg_similarity = similarity_matrix * neg_mask
        neg_similarity = neg_similarity.sum(dim=1) / (neg_mask.sum(dim=1) + 1e-8)

        # 对比损失
        contrastive_loss = torch.mean(torch.clamp(neg_similarity - pos_similarity + self.margin, min=0.0))

        # 动态调整权重 - 增加对比损失的权重以提高类别区分能力
        lambda_contrast = min(0.2 + 0.5 * epoch / 100, 0.7)  # 提高最大权重

        # 总损失 - 增加一致性损失的权重
        total_loss = ce_loss + 0.3 * consistency_loss + lambda_contrast * contrastive_loss

        return total_loss, ce_loss, consistency_loss, contrastive_loss


# 零样本轴承故障诊断模型
class ZeroShotBearingFaultDiagnosis:
    def __init__(self, sample_length=10000, latent_dim=64, feature_dim=64):
        self.sample_length = sample_length
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim

        # 数据预处理器
        self.preprocessor = DataPreprocessor(sample_length=sample_length)

        # 故障语义构建器
        self.semantic_builder = FaultSemanticBuilder(latent_dim=latent_dim)

        # 模型
        self.semantic_embedding_net = None
        self.attention_fusion = None
        self.dual_channel_cnn = None

        # 训练用故障类型（包含单一和复合故障）
        self.train_fault_types = {
            'normal': 0,
            'inner': 1,
            'outer': 2,
            'ball': 3,
            'inner_ball': 4,
            'inner_outer': 5,
            'outer_ball': 6,
            'inner_outer_ball': 7
        }

        # 测试用故障类型（仅复合故障）
        self.test_fault_types = {
            'inner_ball': 4,
            'inner_outer': 5,
            'outer_ball': 6,
            'inner_outer_ball': 7
        }

        # 反向映射
        self.train_idx_to_fault = {v: k for k, v in self.train_fault_types.items()}
        self.test_idx_to_fault = {v: k for k, v in self.test_fault_types.items()}

        # 单一故障类型（仅用于训练）
        self.single_fault_types = ['normal', 'inner', 'outer', 'ball']

        # 复合故障类型（用于测试）
        self.compound_fault_types = ['inner_ball', 'inner_outer', 'outer_ball', 'inner_outer_ball']

    def load_data(self, data_dir):
        """加载数据集"""
        # 单一故障数据（用于训练和验证）
        single_fault_files = []
        single_fault_labels = []

        for fault_type in self.single_fault_types:
            file_path = os.path.join(data_dir, f"{fault_type}.csv")
            if os.path.exists(file_path):
                single_fault_files.append(file_path)
                single_fault_labels.append(self.train_fault_types[fault_type])

        # 复合故障数据（用于测试）
        compound_fault_files = []
        compound_fault_labels = []

        for fault_type in self.compound_fault_types:
            file_path = os.path.join(data_dir, f"{fault_type}.csv")
            if os.path.exists(file_path):
                compound_fault_files.append(file_path)
                compound_fault_labels.append(self.test_fault_types[fault_type])

        # 准备训练集和验证集
        X_train, y_train = self.preprocessor.prepare_dataset(
            single_fault_files, single_fault_labels, augment=True, augment_times=3
        )

        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
        )

        # 准备测试集（仅复合故障）
        X_test, y_test = self.preprocessor.prepare_dataset(
            compound_fault_files, compound_fault_labels, augment=False
        )

        print(f"训练集大小: {X_train.shape}")
        print(f"验证集大小: {X_val.shape}")
        print(f"测试集大小: {X_test.shape}")

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def build_semantics(self, X_train, y_train):
        """构建故障语义"""
        # 知识语义
        knowledge_semantics = {}
        for fault_type, idx in self.fault_types.items():
            knowledge_semantic = self.semantic_builder.build_knowledge_semantic(fault_type)
            knowledge_semantics[idx] = knowledge_semantic

        # 数据语义
        # 按类别分组
        X_by_class = {}
        for i in range(len(y_train)):
            label = y_train[i]
            if label not in X_by_class:
                X_by_class[label] = []
            X_by_class[label].append(X_train[i])

        data_semantics = {}
        for label, data in X_by_class.items():
            data = np.array(data)
            data_semantic = self.semantic_builder.build_data_semantic(data, train=True)

            # 对数据语义进行谱聚类，获取聚类中心
            # 增加聚类数量以捕获更多样本分布特征
            cluster_centers, _ = self.semantic_builder.spectral_clustering(data_semantic, n_clusters=3)
            # 使用所有聚类中心的平均值作为最终数据语义
            data_semantics[label] = np.mean(cluster_centers, axis=0)

        # 合成复合故障语义
        compound_knowledge_semantics = {}
        compound_data_semantics = {}

        for fault_type in self.compound_fault_types:
            idx = self.fault_types[fault_type]

            # 分解复合故障
            fault_components = fault_type.split('_')
            component_indices = [self.fault_types[comp] for comp in fault_components if comp in self.fault_types]

            # 合成知识语义 - 使用加权合成而非简单的max操作
            component_knowledge_semantics = [knowledge_semantics[comp_idx] for comp_idx in component_indices]
            # 使用加权平均而非最大值，以保留更多信息
            weights = np.ones(len(component_knowledge_semantics)) / len(component_knowledge_semantics)
            compound_knowledge_semantic = np.zeros_like(component_knowledge_semantics[0])
            for i, semantic in enumerate(component_knowledge_semantics):
                compound_knowledge_semantic += weights[i] * semantic
            compound_knowledge_semantics[idx] = compound_knowledge_semantic

            # 合成数据语义 - 同样使用加权合成
            if all(comp_idx in data_semantics for comp_idx in component_indices):
                component_data_semantics = [data_semantics[comp_idx] for comp_idx in component_indices]
                # 使用加权平均
                compound_data_semantic = np.zeros_like(component_data_semantics[0])
                for i, semantic in enumerate(component_data_semantics):
                    compound_data_semantic += weights[i] * semantic
                compound_data_semantics[idx] = compound_data_semantic

        # 更新知识语义和数据语义
        knowledge_semantics.update(compound_knowledge_semantics)
        data_semantics.update(compound_data_semantics)

        # 增强复合故障与单一故障的区分度
        # 对复合故障语义进行额外的增强处理
        for fault_type in self.compound_fault_types:
            idx = self.fault_types[fault_type]
            if idx in knowledge_semantics and idx in data_semantics:
                # 增强知识语义中的复合特征
                fault_components = fault_type.split('_')
                for comp in fault_components:
                    if comp in self.fault_types:
                        comp_idx = self.fault_types[comp]
                        # 增强复合故障与单一故障的差异
                        diff_vector = knowledge_semantics[idx] - knowledge_semantics[comp_idx]
                        knowledge_semantics[idx] += 0.2 * diff_vector

                # 同样增强数据语义
                for comp in fault_components:
                    if comp in self.fault_types:
                        comp_idx = self.fault_types[comp]
                        if comp_idx in data_semantics:
                            diff_vector = data_semantics[idx] - data_semantics[comp_idx]
                            data_semantics[idx] += 0.2 * diff_vector

        return knowledge_semantics, data_semantics

    def initialize_models(self, input_size, num_classes, is_training=True):
        """初始化模型"""
        # 计算语义维度
        knowledge_dim = len(self.semantic_builder.build_knowledge_semantic('normal'))
        data_dim = self.latent_dim
        fused_dim = 64

        # 初始化语义嵌入网络
        self.semantic_embedding_net = SemanticEmbeddingNetwork(
            semantic_dim=fused_dim,
            feature_dim=self.feature_dim
        ).to(device)

        # 初始化注意力融合模块
        self.attention_fusion = AttentionFusion(
            knowledge_dim=knowledge_dim,
            data_dim=data_dim,
            fused_dim=fused_dim
        ).to(device)

        # 初始化双通道卷积网络
        # 训练时使用所有故障类型，测试时仅使用复合故障
        self.dual_channel_cnn = DualChannelCNN(
            input_size=input_size,
            semantic_dim=data_dim,
            num_classes=num_classes if is_training else len(self.test_fault_types)
        ).to(device)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, lr=0.001):
        """训练模型"""
        # 构建故障语义
        knowledge_semantics, data_semantics = self.build_semantics(X_train, y_train)

        # 初始化模型（训练时包含所有故障类型）
        input_size = X_train.shape[1]
        num_classes = len(self.train_fault_types)  # 8种故障
        self.initialize_models(input_size, num_classes, is_training=True)

        # 准备数据集
        train_dataset = BearingDataset(X_train, y_train, knowledge_semantics, data_semantics)
        val_dataset = BearingDataset(X_val, y_val, knowledge_semantics, data_semantics)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 优化器
        optimizer = optim.Adam([
            {'params': self.semantic_embedding_net.parameters(), 'lr': lr},
            {'params': self.attention_fusion.parameters(), 'lr': lr},
            {'params': self.dual_channel_cnn.parameters(), 'lr': lr * 1.5}  # 增加CNN学习率
        ], lr=lr)

        # 学习率调度器 - 使用更慢的衰减速率
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * 1.5, eta_min=lr / 20)

        # 损失函数
        criterion = CompositeLoss(margin=0.7)  # 增加margin以提高类别区分度

        # 训练循环
        best_val_acc = 0.0
        patience = 15  # 增加耐心值，允许模型训练更长时间
        patience_counter = 0

        # 渐进式训练策略
        for epoch in range(epochs):
            # 训练阶段
            if epoch < epochs // 3:
                # 阶段1：冻结语义嵌入网络，优化特征提取网络
                self.semantic_embedding_net.eval()
                for param in self.semantic_embedding_net.parameters():
                    param.requires_grad = False

                self.attention_fusion.train()
                self.dual_channel_cnn.train()

            elif epoch < 2 * epochs // 3:
                # 阶段2：冻结特征提取网络，优化语义嵌入网络
                self.semantic_embedding_net.train()
                for param in self.semantic_embedding_net.parameters():
                    param.requires_grad = True

                self.dual_channel_cnn.eval()
                for param in self.dual_channel_cnn.parameters():
                    param.requires_grad = False

                self.attention_fusion.train()

            else:
                # 阶段3：联合微调全网络
                self.semantic_embedding_net.train()
                for param in self.semantic_embedding_net.parameters():
                    param.requires_grad = True

                self.attention_fusion.train()

                self.dual_channel_cnn.train()
                for param in self.dual_channel_cnn.parameters():
                    param.requires_grad = True

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (signals, targets, k_semantics, d_semantics) in enumerate(train_loader):
                signals = signals.unsqueeze(1).float().to(device)  # 添加通道维度
                targets = targets.long().to(device)
                k_semantics = k_semantics.float().to(device)
                d_semantics = d_semantics.float().to(device)

                # 前向传播
                # 1. 语义融合
                fused_semantics, attention_weights = self.attention_fusion(k_semantics, d_semantics)

                # 2. 语义嵌入
                embedded_semantics = self.semantic_embedding_net(fused_semantics)

                # 3. 双通道卷积网络
                outputs = self.dual_channel_cnn(signals, d_semantics)

                # 计算损失
                loss, ce_loss, consistency_loss, contrastive_loss = criterion(
                    embedded_semantics, fused_semantics, outputs, targets, epoch
                )

                # 添加L2-SP正则化（阶段3）
                if epoch >= 2 * epochs // 3:
                    l2_sp_reg = 0.0
                    # 获取初始模型参数
                    if hasattr(self, 'initial_params'):
                        for name, param in self.semantic_embedding_net.named_parameters():
                            if name in self.initial_params:
                                l2_sp_reg += torch.sum((param - self.initial_params[name]) ** 2)

                        for name, param in self.dual_channel_cnn.named_parameters():
                            if name in self.initial_params:
                                l2_sp_reg += torch.sum((param - self.initial_params[name]) ** 2)

                        # 添加到总损失
                        loss += 0.01 * l2_sp_reg

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.semantic_embedding_net.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(self.attention_fusion.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(self.dual_channel_cnn.parameters(), max_norm=5.0)

                optimizer.step()

                # 统计
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

                # 打印批次进度
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                          f"Loss: {loss.item():.4f}, CE: {ce_loss.item():.4f}, "
                          f"Consist: {consistency_loss.item():.4f}, Contrast: {contrastive_loss.item():.4f}, "
                          f"Acc: {100. * train_correct / train_total:.2f}%")

            # 更新学习率
            scheduler.step()

            # 验证阶段
            self.semantic_embedding_net.eval()
            self.attention_fusion.eval()
            self.dual_channel_cnn.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for signals, targets, k_semantics, d_semantics in val_loader:
                    signals = signals.unsqueeze(1).float().to(device)
                    targets = targets.long().to(device)
                    k_semantics = k_semantics.float().to(device)
                    d_semantics = d_semantics.float().to(device)

                    # 前向传播
                    fused_semantics, _ = self.attention_fusion(k_semantics, d_semantics)
                    embedded_semantics = self.semantic_embedding_net(fused_semantics)
                    outputs = self.dual_channel_cnn(signals, d_semantics)

                    # 计算损失
                    loss, _, _, _ = criterion(embedded_semantics, fused_semantics, outputs, targets, epoch)

                    # 统计
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            # 计算验证准确率
            val_acc = 100. * val_correct / val_total

            # 打印训练和验证结果
            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Train Acc: {100. * train_correct / train_total:.2f}%, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, "
                  f"Val Acc: {val_acc:.2f}%")

            # 早停策略
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                # 保存最佳模型
                torch.save({
                    'semantic_embedding_net': self.semantic_embedding_net.state_dict(),
                    'attention_fusion': self.attention_fusion.state_dict(),
                    'dual_channel_cnn': self.dual_channel_cnn.state_dict()
                }, 'best_model.pth')

                print(f"保存最佳模型，验证准确率: {best_val_acc:.2f}%")

                # 保存初始参数（用于L2-SP正则化）
                if epoch == 0:
                    self.initial_params = {}
                    for name, param in self.semantic_embedding_net.named_parameters():
                        self.initial_params[name] = param.clone().detach()

                    for name, param in self.dual_channel_cnn.named_parameters():
                        self.initial_params[name] = param.clone().detach()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停：验证准确率连续 {patience} 轮未提升")
                    break

        # 加载最佳模型
        checkpoint = torch.load('best_model.pth')
        self.semantic_embedding_net.load_state_dict(checkpoint['semantic_embedding_net'])
        self.attention_fusion.load_state_dict(checkpoint['attention_fusion'])
        self.dual_channel_cnn.load_state_dict(checkpoint['dual_channel_cnn'])

        return best_val_acc

    def test(self, X_test, y_test, batch_size=64):
        """测试模型"""
        # 构建故障语义（仅针对复合故障）
        knowledge_semantics, data_semantics = self.build_semantics(X_test, y_test)

        # 重新初始化双通道CNN，仅输出复合故障类别
        input_size = X_test.shape[1]
        self.initialize_models(input_size, len(self.test_fault_types), is_training=False)

        # 加载最佳模型
        checkpoint = torch.load('best_model.pth')
        self.semantic_embedding_net.load_state_dict(checkpoint['semantic_embedding_net'])
        self.attention_fusion.load_state_dict(checkpoint['attention_fusion'])
        self.dual_channel_cnn.load_state_dict(checkpoint['dual_channel_cnn'], strict=False)  # 允许部分加载

        # 准备数据集
        test_dataset = BearingDataset(X_test, y_test, knowledge_semantics, data_semantics)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 评估模式
        self.semantic_embedding_net.eval()
        self.attention_fusion.eval()
        self.dual_channel_cnn.eval()

        # 测试
        test_correct = 0
        test_total = 0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for signals, targets, k_semantics, d_semantics in test_loader:
                signals = signals.unsqueeze(1).float().to(device)
                targets = targets.long().to(device)
                k_semantics = k_semantics.float().to(device)
                d_semantics = d_semantics.float().to(device)

                # 前向传播
                fused_semantics, attention_weights = self.attention_fusion(k_semantics, d_semantics)
                embedded_semantics = self.semantic_embedding_net(fused_semantics)
                outputs = self.dual_channel_cnn(signals, d_semantics)

                # 获取预测概率
                probs = F.softmax(outputs, dim=1)

                # 统计
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

                # 收集预测结果
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        # 合并所有预测概率
        all_probs = np.vstack(all_probs)

        # 计算测试准确率
        test_acc = 100. * test_correct / test_total
        print(f"复合故障测试准确率: {test_acc:.2f}%")

        # 计算每种复合故障类型的准确率
        for fault_type in self.compound_fault_types:
            fault_idx = self.test_fault_types[fault_type]
            fault_indices = [i for i, label in enumerate(all_targets) if label == fault_idx]
            if fault_indices:
                fault_preds = [all_preds[i] for i in fault_indices]
                fault_targets = [all_targets[i] for i in fault_indices]
                fault_acc = 100. * sum(p == t for p, t in zip(fault_preds, fault_targets)) / len(fault_indices)
                print(f"{fault_type} 故障准确率: {fault_acc:.2f}%")

        # 绘制混淆矩阵
        self.plot_confusion_matrix(all_targets, all_preds)

        # 特征可视化
        self.visualize_features(test_loader)

        # 计算预测置信度
        confidence_scores = np.max(all_probs, axis=1)
        avg_confidence = np.mean(confidence_scores)
        print(f"复合故障平均预测置信度: {avg_confidence:.4f}")

        # 绘制预测置信度分布
        plt.figure(figsize=(10, 6))
        sns.histplot(confidence_scores, bins=20, kde=True)
        plt.axvline(x=avg_confidence, color='r', linestyle='--', label=f'平均置信度: {avg_confidence:.4f}')
        plt.xlabel('预测置信度')
        plt.ylabel('样本数量')
        plt.title('复合故障预测置信度分布')
        plt.legend()
        plt.tight_layout()
        plt.savefig('confidence_distribution.png')
        plt.close()

        return test_acc

    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 标签名称（仅复合故障）
        labels = [self.test_idx_to_fault[i] for i in sorted(self.test_fault_types.values())]

        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('复合故障混淆矩阵')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

    def visualize_features(self, data_loader):
        """使用t-SNE可视化特征分布"""
        features = []
        labels = []

        self.semantic_embedding_net.eval()
        self.attention_fusion.eval()
        self.dual_channel_cnn.eval()

        with torch.no_grad():
            for signals, targets, k_semantics, d_semantics in data_loader:
                signals = signals.unsqueeze(1).float().to(device)
                k_semantics = k_semantics.float().to(device)
                d_semantics = d_semantics.float().to(device)

                # 前向传播获取特征
                fused_semantics, _ = self.attention_fusion(k_semantics, d_semantics)
                embedded_features = self.semantic_embedding_net(fused_semantics)

                # 收集特征和标签
                features.append(embedded_features.cpu().numpy())
                labels.extend(targets.numpy())

        # 合并特征
        features = np.vstack(features)

        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)

        # 绘制t-SNE图
        plt.figure(figsize=(10, 8))

        # 为复合故障类别设置不同的颜色和标记
        for i, fault_type in self.test_idx_to_fault.items():
            indices = [j for j, label in enumerate(labels) if label == i]
            if indices:
                plt.scatter(
                    features_2d[indices, 0],
                    features_2d[indices, 1],
                    label=fault_type,
                    alpha=0.7
                )

        plt.legend()
        plt.title('复合故障t-SNE特征可视化')
        plt.tight_layout()
        plt.savefig('tsne_visualization.png')
        plt.close()

    def plot_semantic_similarity(self, knowledge_semantics, data_semantics):
        """绘制语义相似度矩阵（仅复合故障）"""
        # 知识语义相似度矩阵
        k_sim_matrix = np.zeros((len(self.test_fault_types), len(self.test_fault_types)))
        for i, idx_i in enumerate(sorted(self.test_fault_types.values())):
            for j, idx_j in enumerate(sorted(self.test_fault_types.values())):
                if idx_i in knowledge_semantics and idx_j in knowledge_semantics:
                    k_sim_matrix[i, j] = np.dot(knowledge_semantics[idx_i], knowledge_semantics[idx_j]) / (
                            np.linalg.norm(knowledge_semantics[idx_i]) * np.linalg.norm(knowledge_semantics[idx_j])
                    )

        # 数据语义相似度矩阵
        d_sim_matrix = np.zeros((len(self.test_fault_types), len(self.test_fault_types)))
        for i, idx_i in enumerate(sorted(self.test_fault_types.values())):
            for j, idx_j in enumerate(sorted(self.test_fault_types.values())):
                if idx_i in data_semantics and idx_j in data_semantics:
                    d_sim_matrix[i, j] = np.dot(data_semantics[idx_i], data_semantics[idx_j]) / (
                            np.linalg.norm(data_semantics[idx_i]) * np.linalg.norm(data_semantics[idx_j])
                    )

        # 融合语义相似度矩阵
        f_sim_matrix = (k_sim_matrix + d_sim_matrix) / 2

        # 标签名称（仅复合故障）
        labels = [self.test_idx_to_fault[i] for i in sorted(self.test_fault_types.values())]

        # 绘制知识语义相似度矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(k_sim_matrix, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
        plt.xlabel('复合故障类型')
        plt.ylabel('复合故障类型')
        plt.title('复合故障知识语义相似度矩阵')
        plt.tight_layout()
        plt.savefig('knowledge_semantic_similarity.png')
        plt.close()

        # 绘制数据语义相似度矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(d_sim_matrix, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
        plt.xlabel('复合故障类型')
        plt.ylabel('复合故障类型')
        plt.title('复合故障数据语义相似度矩阵')
        plt.tight_layout()
        plt.savefig('data_semantic_similarity.png')
        plt.close()

        # 绘制融合语义相似度矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(f_sim_matrix, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
        plt.xlabel('复合故障类型')
        plt.ylabel('复合故障类型')
        plt.title('复合故障融合语义相似度矩阵')
        plt.tight_layout()
        plt.savefig('fused_semantic_similarity.png')
        plt.close()


# 轴承数据集
class BearingDataset(Dataset):
    def __init__(self, signals, labels, knowledge_semantics, data_semantics):
        self.signals = signals
        self.labels = labels
        self.knowledge_semantics = knowledge_semantics
        self.data_semantics = data_semantics

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        # 获取知识语义和数据语义
        k_semantic = self.knowledge_semantics.get(label, np.zeros(10))  # 默认值
        d_semantic = self.data_semantics.get(label, np.zeros(64))  # 默认值

        return signal, label, k_semantic, d_semantic


# 主函数
def main():
    # 设置数据目录
    data_dir = "E:\研究生\CNN\HUST1"

    # 创建零样本轴承故障诊断模型
    model = ZeroShotBearingFaultDiagnosis(sample_length=10000, latent_dim=64, feature_dim=64)

    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = model.load_data(data_dir)

    # 训练模型
    best_val_acc = model.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64, lr=0.001)
    print(f"最佳验证准确率: {best_val_acc:.2f}%")

    # 测试模型（仅复合故障）
    test_acc = model.test(X_test, y_test, batch_size=64)
    print(f"复合故障测试准确率: {test_acc:.2f}%")

    # 构建故障语义并绘制相似度矩阵（仅复合故障）
    knowledge_semantics, data_semantics = model.build_semantics(X_test, y_test)
    model.plot_semantic_similarity(knowledge_semantics, data_semantics)


if __name__ == "__main__":
    main()

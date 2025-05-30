import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
from scipy.interpolate import CubicSpline
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
from tqdm import tqdm
import warnings
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')


# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# 1. 数据预处理模块
class DataPreprocessor:
    def __init__(self, sample_length=1024, overlap=0.5, augment=True):
        self.sample_length = sample_length
        self.overlap = overlap
        self.augment = augment
        self.stride = int(sample_length * (1 - overlap))

    def cubic_spline_interpolation(self, signal_with_missing):
        """使用三次样条插值法处理缺失值"""
        mask = np.isnan(signal_with_missing)
        if np.any(mask):
            x = np.arange(len(signal_with_missing))
            x_known = x[~mask]
            y_known = signal_with_missing[~mask]
            cs = CubicSpline(x_known, y_known)
            signal_with_missing[mask] = cs(x[mask])
        return signal_with_missing

    def remove_outliers(self, signal_data):
        """结合3σ准则与箱线图方法(IQR法则)去除异常值"""
        q1, q3 = np.percentile(signal_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # 3σ准则
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        lower_sigma = mean - 3 * std
        upper_sigma = mean + 3 * std

        # 取两种方法的交集作为最终的阈值
        final_lower = max(lower_bound, lower_sigma)
        final_upper = min(upper_bound, upper_sigma)

        # 将异常值替换为阈值
        signal_data = np.clip(signal_data, final_lower, final_upper)
        return signal_data

    def wavelet_denoising(self, signal_data, wavelet='db4', level=3):
        """使用小波变换进行信号去噪"""
        # 小波分解
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)

        # 使用改进的SUREShrink阈值处理高频系数
        for i in range(1, len(coeffs)):
            N = len(signal_data)
            j = i
            sigma = np.median(np.abs(coeffs[i])) / 0.6745  # 噪声估计
            threshold = sigma * np.sqrt(2 * np.log(N)) * (1 / (1 + np.sqrt(np.log2(j + 1))))
            coeffs[i] = pywt.threshold(coeffs[i], threshold, 'soft')

        # 小波重构
        denoised_signal = pywt.waverec(coeffs, wavelet)

        # 确保输出长度与输入相同
        if len(denoised_signal) > len(signal_data):
            denoised_signal = denoised_signal[:len(signal_data)]

        return denoised_signal

    def butterworth_filter(self, signal_data, cutoff=10, fs=10000, order=4):
        """应用巴特沃斯低通滤波器"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low')
        filtered_signal = signal.filtfilt(b, a, signal_data)
        return filtered_signal

    def add_gaussian_noise(self, signal_data, snr=10):
        """添加高斯白噪声，SNR为信噪比(dB)"""
        signal_power = np.mean(signal_data ** 2)
        noise_power = signal_power / (10 ** (snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal_data))
        noisy_signal = signal_data + noise
        return noisy_signal

    def time_augmentation(self, signal_data):
        """时域扩增：随机时移和缩放"""
        # 随机时移 ±5%
        shift = np.random.randint(-int(0.05 * len(signal_data)), int(0.05 * len(signal_data)))
        shifted_signal = np.roll(signal_data, shift)

        # 随机缩放 0.95-1.05
        scale_factor = np.random.uniform(0.95, 1.05)
        scaled_signal = signal_data * scale_factor

        return scaled_signal

    def segment_signal(self, signal_data):
        """将长信号分割为固定长度的样本"""
        segments = []
        for i in range(0, len(signal_data) - self.sample_length + 1, self.stride):
            segment = signal_data[i:i + self.sample_length]
            segments.append(segment)
        return np.array(segments)

    def preprocess(self, signal_data, augmentation=False):
        """完整的预处理流程"""
        # 缺失值处理
        signal_data = self.cubic_spline_interpolation(signal_data)

        # 异常值处理
        signal_data = self.remove_outliers(signal_data)

        # 小波去噪
        signal_data = self.wavelet_denoising(signal_data)

        # 巴特沃斯滤波
        signal_data = self.butterworth_filter(signal_data)

        # 如果需要数据增强
        if augmentation and self.augment:
            aug_signals = []
            # 添加高斯噪声
            for snr in [5, 10, 15]:
                aug_signals.append(self.add_gaussian_noise(signal_data, snr))

            # 时域扩增（随机时移和缩放）
            for _ in range(3):
                aug_signals.append(self.time_augmentation(signal_data))

            # 将所有增强后的信号与原始信号合并
            aug_signals.append(signal_data)

            # 分割所有信号
            all_segments = []
            for sig in aug_signals:
                segments = self.segment_signal(sig)
                all_segments.append(segments)
            return np.vstack(all_segments)
        else:
            # 分割原始信号
            return self.segment_signal(signal_data)


# 2. 知识语义和数据语义构建模块
class FaultSemanticBuilder:
    def __init__(self, latent_dim=64, hidden_dim=128):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = None
        self.preprocessor = DataPreprocessor(sample_length=1024)
        # 增加中心损失参数
        self.center_weight = 0.01
        self.centers = None
        self.center_loss_weight = 0.5  # 中心损失权重

    def build_knowledge_semantics(self):
        """构建基于轴承故障位置和尺寸的知识语义"""
        # 定义故障位置的one-hot编码
        # [内圈, 外圈, 滚动体]
        fault_location = {
            'normal': [0, 0, 0],
            'inner': [1, 0, 0],
            'outer': [0, 1, 0],
            'ball': [0, 0, 1],
            'inner_outer': [1, 1, 0],
            'inner_ball': [1, 0, 1],
            'outer_ball': [0, 1, 1],
            'inner_outer_ball': [1, 1, 1]
        }

        # 轴承参数的归一化取值
        bearing_params = {
            'inner_diameter': 17 / 40,  # 归一化到[0,1]
            'outer_diameter': 1.0,  # 40/40 = 1.0
            'width': 12 / 40,  # 12/40
            'ball_diameter': 6.75 / 40,  # 6.75/40
            'ball_number': 9 / 20  # 假设最大为20个球
        }

        # 构建完整的知识语义向量
        knowledge_semantics = {}
        for fault_type, location_encoding in fault_location.items():
            # 连接故障位置编码和轴承参数
            semantics = np.array(location_encoding + list(bearing_params.values()))
            knowledge_semantics[fault_type] = semantics

        return knowledge_semantics

    def _enhanced_contrastive_loss(self, z, labels, temperature=0.1, margin=1.0):
        """
        增强的对比损失函数，使用更强的约束让不同类别更加分散

        Args:
            z: 潜在表示
            labels: 故障类型标签
            temperature: 温度参数，调整相似度分布的平滑程度(降低会使差异更明显)
            margin: 不同类别间的最小距离边界
        """
        batch_size = z.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=z.device)

        # L2标准化潜在向量
        z_norm = F.normalize(z, p=2, dim=1)

        # 计算余弦相似度矩阵
        sim_matrix = torch.mm(z_norm, z_norm.t()) / temperature

        # 创建同类样本掩码(正样本对)
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        # 排除自身相似度
        pos_mask.fill_diagonal_(0)

        # 创建异类样本掩码(负样本对)
        neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()

        # 计算正样本损失 - 推动同类样本更近
        pos_similarity = sim_matrix * pos_mask
        pos_loss = torch.sum(1.0 - pos_similarity) / (pos_mask.sum() + 1e-8)

        # 计算负样本损失 - 使不同类样本距离超过margin
        neg_similarity = sim_matrix * neg_mask
        # 应用margin: 只有当相似度大于-margin时才产生损失
        neg_loss = torch.sum(torch.clamp(neg_similarity + margin, min=0)) / (neg_mask.sum() + 1e-8)

        # 总对比损失
        contrastive_loss = pos_loss + neg_loss

        return contrastive_loss

    def _center_loss(self, z, labels, centers):
        """
        中心损失函数，强制同类样本聚集在类中心周围

        Args:
            z: 潜在表示
            labels: 故障类型标签
            centers: 各类别中心点
        """
        batch_size = z.size(0)
        unique_labels = torch.unique(labels)

        # 计算每个样本到其类别中心的距离
        distances = torch.zeros(batch_size, device=z.device)
        for i in range(batch_size):
            label = labels[i].item()
            if label < len(centers):  # 确保标签索引有效
                center = centers[label]
                distances[i] = torch.norm(z[i] - center, p=2)

        # 类内聚集损失 - 平均距离
        center_loss = torch.mean(distances)

        return center_loss

    def _create_autoencoder(self, input_dim):
        """创建增强的自编码器模型"""

        class EnhancedAutoEncoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super(EnhancedAutoEncoder, self).__init__()

                # 编码器 - 增加深度和跳连接以提高特征提取能力
                self.encoder_layers = nn.ModuleList([
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),

                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.BatchNorm1d(hidden_dim // 2),
                    nn.LeakyReLU(0.2),

                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.BatchNorm1d(hidden_dim // 4),
                    nn.LeakyReLU(0.2),
                ])

                # 添加潜在空间投影层
                self.latent_proj = nn.Linear(hidden_dim // 4, latent_dim)

                # 增加一个L2归一化层以帮助对比学习
                self.l2_norm = lambda x: F.normalize(x, p=2, dim=1)

                # 解码器 - 与编码器对称
                self.decoder_layers = nn.ModuleList([
                    nn.Linear(latent_dim, hidden_dim // 4),
                    nn.BatchNorm1d(hidden_dim // 4),
                    nn.LeakyReLU(0.2),

                    nn.Linear(hidden_dim // 4, hidden_dim // 2),
                    nn.BatchNorm1d(hidden_dim // 2),
                    nn.LeakyReLU(0.2),

                    nn.Linear(hidden_dim // 2, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),

                    nn.Linear(hidden_dim, input_dim)
                ])

                # 添加降噪正则化，减少过拟合
                self.dropout = nn.Dropout(0.2)

                # 特征融合门机制，更好地保留原始信息
                self.fusion_gate = nn.Parameter(torch.tensor([0.5]))

            def encode(self, x):
                # 逐层编码
                h = x
                skip_connections = []

                for i, layer in enumerate(self.encoder_layers):
                    h = layer(h)
                    # 存储每一层的输出用于跳连接
                    if i % 3 == 2:  # 每个LeakyReLU后
                        skip_connections.append(h)

                # 投影到潜在空间
                z = self.latent_proj(h)

                # 返回原始潜在表示和L2归一化后的表示
                z_norm = self.l2_norm(z)

                return z, z_norm, skip_connections

            def decode(self, z, skip_connections=None):
                # 从潜在空间解码
                h = z

                # 将跳连接反向添加回解码器
                skip_idx = len(skip_connections) - 1 if skip_connections else -1

                for i, layer in enumerate(self.decoder_layers):
                    h = layer(h)

                    # 应用跳跃连接
                    if i % 3 == 2 and skip_idx >= 0:  # 每个LeakyReLU后
                        # 融合门机制动态调整跳连接权重
                        gate = torch.sigmoid(self.fusion_gate)
                        h = gate * h + (1 - gate) * skip_connections[skip_idx]
                        skip_idx -= 1

                    # 在中间层添加dropout防止过拟合
                    if i < len(self.decoder_layers) - 1:  # 最后输出层不使用dropout
                        h = self.dropout(h)

                return h

            def forward(self, x):
                # 编码
                z, z_norm, skip_connections = self.encode(x)
                # 解码
                x_recon = self.decode(z, skip_connections)
                return x_recon, z, z_norm

        self.autoencoder = EnhancedAutoEncoder(input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        return self.autoencoder

    def _reconstruction_loss(self, x_recon, x):
        """增强的重构损失，结合多种信息"""
        # 均方误差损失
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')

        # 频域损失 - 保留频谱特性
        x_fft = torch.fft.rfft(x, dim=1)
        x_recon_fft = torch.fft.rfft(x_recon, dim=1)
        freq_loss = F.mse_loss(torch.abs(x_fft), torch.abs(x_recon_fft), reduction='mean')

        # 相位损失 - 保留相位信息
        phase_x = torch.angle(x_fft)
        phase_recon = torch.angle(x_recon_fft)
        phase_loss = torch.mean(1 - torch.cos(phase_x - phase_recon))

        # 总重构损失 - 权重可调整
        total_loss = mse_loss + 0.2 * freq_loss + 0.1 * phase_loss

        return total_loss

    def _mmd_loss(self, z, batch_size):
        """最大均值差异损失，使潜在特征分布更接近高斯分布"""

        # 多尺度RBF核函数，提高MMD估计的准确性
        def gaussian_kernel_matrix(x, y, sigma_values=[0.01, 0.1, 1, 10, 100]):
            dist = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), p=2, dim=2)
            kernels = torch.zeros_like(dist)

            for sigma in sigma_values:
                gamma = 1.0 / (2.0 * sigma ** 2)
                kernels += torch.exp(-gamma * dist)

            return kernels / len(sigma_values)

        # 生成标准正态分布样本
        z_normal = torch.randn_like(z)

        # 计算批次内样本间的高斯核
        xx = gaussian_kernel_matrix(z, z)

        # 计算随机正态分布样本间的高斯核
        yy = gaussian_kernel_matrix(z_normal, z_normal)

        # 计算交叉核
        xy = gaussian_kernel_matrix(z, z_normal)

        # 计算MMD^2
        mmd_loss = xx.mean() + yy.mean() - 2 * xy.mean()

        return mmd_loss

    def _inter_class_variance_loss(self, z, labels):
        """类间方差损失，增大不同类之间的距离"""
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)

        if num_classes <= 1:
            return torch.tensor(0.0, device=z.device)

        # 计算每个类的中心
        class_centers = []
        for label in unique_labels:
            indices = (labels == label).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                center = torch.mean(z[indices], dim=0)
                class_centers.append(center)

        class_centers = torch.stack(class_centers)

        # 计算类中心之间的平均距离倒数
        total_inv_distance = 0
        count = 0

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                # 计算欧氏距离的倒数（加上小常数防止除零）
                dist = torch.norm(class_centers[i] - class_centers[j], p=2)
                total_inv_distance += 1.0 / (dist + 1e-5)  # 使用倒数，距离越大，值越小
                count += 1

        # 平均倒数距离
        if count > 0:
            avg_inv_distance = total_inv_distance / count
            return avg_inv_distance

        return torch.tensor(0.0, device=z.device)

    def _label_smoothing_loss(self, z, labels, num_classes):
        """标签平滑交叉熵损失，增加分类器的鲁棒性"""
        # 构建分类器头
        if not hasattr(self, 'classifier_head'):
            self.classifier_head = nn.Linear(z.size(1), num_classes).to(z.device)

        # 预测类别
        logits = self.classifier_head(z)

        # 标签平滑参数
        epsilon = 0.1

        # 创建平滑标签
        n_classes = num_classes
        one_hot = torch.zeros_like(logits).scatter(1, labels.unsqueeze(1), 1)
        smooth_labels = one_hot * (1 - epsilon) + epsilon / n_classes

        # 计算交叉熵
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_labels * log_probs).sum(dim=1).mean()

        return loss

    def train_autoencoder(self, X_train, labels=None, epochs=100, batch_size=64, lr=0.001):
        """训练自编码器提取数据语义特征，增加对比损失和类内聚集约束"""
        # 创建自编码器
        self._create_autoencoder(X_train.shape[1])

        # 创建数据加载器
        if labels is not None:
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(labels))
        else:
            train_dataset = TensorDataset(torch.FloatTensor(X_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 定义优化器
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=lr / 20)

        # 如果有标签，初始化中心点向量
        unique_labels = np.unique(labels) if labels is not None else []
        num_classes = len(unique_labels)

        if labels is not None:
            self.centers = torch.zeros(num_classes, self.latent_dim).to(self.device)
            # 使用动量更新类中心
            center_momentum = 0.9

        # 训练循环
        best_loss = float('inf')
        patience_counter = 0
        patience = 10  # 早停参数

        for epoch in range(epochs):
            self.autoencoder.train()
            total_loss = 0
            total_recon_loss = 0
            total_contr_loss = 0
            total_mmd_loss = 0
            total_center_loss = 0
            total_inter_class_loss = 0

            # 动态调整损失权重
            epoch_ratio = epoch / epochs
            contrastive_weight = 0.1 + 0.4 * epoch_ratio  # 随着训练进行逐渐增大对比损失权重
            mmd_weight = 0.01 * (1.0 - 0.5 * epoch_ratio)  # 随着训练进行逐渐减小MMD权重

            # 动态调整温度参数 - 随着训练进行降低温度使得对比效果更加明显
            temperature = max(0.05, 0.2 - 0.15 * epoch_ratio)
            margin = min(2.0, 0.5 + 1.5 * epoch_ratio)  # 随着训练进行逐渐增大类间距离

            for data in train_loader:
                # 处理数据，支持带标签和不带标签的情况
                if labels is not None:
                    x, batch_labels = data
                    x = x.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                else:
                    x = data[0].to(self.device)
                    batch_labels = None

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播
                x_recon, z, z_norm = self.autoencoder(x)

                # 计算重构损失
                recon_loss = self._reconstruction_loss(x_recon, x)

                # 计算分布对齐损失
                mmd = self._mmd_loss(z, x.size(0))

                # 总损失初始化
                loss = recon_loss + mmd_weight * mmd

                # 如果提供了标签，计算和添加各种类别相关的损失
                if batch_labels is not None:
                    # 增强的对比损失
                    contr_loss = self._enhanced_contrastive_loss(z_norm, batch_labels,
                                                                 temperature=temperature,
                                                                 margin=margin)

                    # 计算类间方差损失
                    inter_class_loss = self._inter_class_variance_loss(z, batch_labels)

                    # 更新类中心 (使用动量更新)
                    with torch.no_grad():
                        for label in torch.unique(batch_labels):
                            label_idx = label.item()
                            indices = (batch_labels == label).nonzero(as_tuple=True)[0]
                            if len(indices) > 0:
                                class_center = torch.mean(z[indices], dim=0)
                                if label_idx < len(self.centers):
                                    self.centers[label_idx] = center_momentum * self.centers[label_idx] + \
                                                              (1 - center_momentum) * class_center

                    # 计算中心损失
                    center_loss = self._center_loss(z, batch_labels, self.centers)

                    # 总损失 - 加入所有损失项
                    # 总损失 - 调整各损失项权重
                    loss = recon_loss + mmd_weight * mmd + \
                           contrastive_weight * contr_loss + \
                           0.1 * center_loss + \
                           0.01 * inter_class_loss  # 大幅减小类间损失权重

                    # 累加损失项
                    total_contr_loss += contr_loss.item() * x.size(0)
                    total_center_loss += center_loss.item() * x.size(0)
                    total_inter_class_loss += inter_class_loss.item() * x.size(0)

                # 反向传播
                loss.backward()

                # 优化器步进
                optimizer.step()

                # 累加损失
                total_loss += loss.item() * x.size(0)
                total_recon_loss += recon_loss.item() * x.size(0)
                total_mmd_loss += mmd.item() * x.size(0)

            # 更新学习率
            scheduler.step()

            # 计算平均损失
            avg_loss = total_loss / len(train_loader.dataset)
            avg_recon_loss = total_recon_loss / len(train_loader.dataset)
            avg_mmd_loss = total_mmd_loss / len(train_loader.dataset)

            # 打印每个损失组件
            loss_components = f"Recon: {avg_recon_loss:.4f}, MMD: {avg_mmd_loss:.4f}"

            if labels is not None:
                avg_contr_loss = total_contr_loss / len(train_loader.dataset)
                avg_center_loss = total_center_loss / len(train_loader.dataset)
                avg_inter_class_loss = total_inter_class_loss / len(train_loader.dataset)
                loss_components += f", Contr: {avg_contr_loss:.4f}, Center: {avg_center_loss:.4f}, Inter: {avg_inter_class_loss:.4f}"

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f} ({loss_components})")

            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.autoencoder.state_dict(), 'best_autoencoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # 加载最佳模型
        self.autoencoder.load_state_dict(torch.load('best_autoencoder.pth'))
        self.autoencoder.eval()

    def extract_data_semantics(self, X, fault_labels=None):
        """从样本中提取数据语义向量"""
        self.autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            # 使用编码器部分提取语义
            z, z_norm, _ = self.autoencoder.encode(X_tensor)
            # 使用归一化后的表示作为语义向量
            data_semantics = z_norm.cpu().numpy()

        # 如果提供了故障标签，则按类别聚类
        if fault_labels is not None:
            unique_faults = np.unique(fault_labels)
            prototype_semantics = {}

            for fault in unique_faults:
                # 获取该故障类型的所有样本索引
                indices = np.where(fault_labels == fault)[0]

                if len(indices) > 0:
                    fault_semantics = data_semantics[indices]

                    # 使用谱聚类优化特征
                    if len(indices) > 10:  # 确保有足够样本进行聚类
                        n_clusters = min(5, len(indices) // 2)  # 动态设置聚类数
                        spectral = SpectralClustering(n_clusters=n_clusters,
                                                      affinity='nearest_neighbors',
                                                      random_state=42)
                        cluster_labels = spectral.fit_predict(fault_semantics)

                        # 找到最大簇作为原型
                        largest_cluster = np.argmax(np.bincount(cluster_labels))
                        prototype_idx = np.where(cluster_labels == largest_cluster)[0]
                        prototype = np.mean(fault_semantics[prototype_idx], axis=0)
                    else:
                        # 样本太少，直接取平均
                        prototype = np.mean(fault_semantics, axis=0)

                    prototype_semantics[fault] = prototype

            return data_semantics, prototype_semantics

        return data_semantics

    def synthesize_compound_semantics(self, single_fault_semantics):
        """使用增强的复合故障语义合成策略"""
        compound_semantics = {}

        # 定义可能的复合故障组合
        compound_combinations = {
            'inner_outer': ['inner', 'outer'],
            'inner_ball': ['inner', 'ball'],
            'outer_ball': ['outer', 'ball'],
            'inner_outer_ball': ['inner', 'outer', 'ball']
        }

        # 合成复合故障语义
        for compound_type, components in compound_combinations.items():
            if all(comp in single_fault_semantics for comp in components):
                # 使用加权最大值合成复合故障语义
                semantics = [single_fault_semantics[comp] for comp in components]
                semantics_array = np.array(semantics)

                # 1. 先找出每个维度上的最大值（特征结合）
                max_features = np.max(semantics_array, axis=0)

                # 2. 增强主导特征，降低次要特征 (非线性变换)
                enhanced = np.sign(max_features) * (np.abs(max_features) ** 0.8)

                # 3. 确保语义向量范数一致（L2归一化）
                compound_semantics[compound_type] = enhanced / np.linalg.norm(enhanced)

        return compound_semantics


# 3. 双通道卷积网络模型
class DualChannelCNN(nn.Module):
    def __init__(self, input_length=1024, semantic_dim=64, num_classes=8):
        super(DualChannelCNN, self).__init__()

        # 通道1：原始信号处理 - 增加深度和膨胀卷积
        self.channel1 = nn.Sequential(
            # 第一层：多尺度特征提取
            nn.Conv1d(1, 64, kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),

            # 第二层：添加膨胀卷积，增大感受野
            nn.Conv1d(64, 128, kernel_size=32, stride=1, padding=16 * 2, dilation=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),

            # 第三层：添加膨胀卷积，进一步增大感受野
            nn.Conv1d(128, 256, kernel_size=16, stride=1, padding=16 * 4, dilation=4),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=2),

            # 第四层：标准卷积提取精细特征
            nn.Conv1d(256, 512, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(8)
        )

        # 计算通道1的展平后特征维度
        self.channel1_flattened = 512 * 8

        # 通道2：数据语义处理 - 更深的网络结构
        self.channel2_fc = nn.Sequential(
            nn.Linear(semantic_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2)
        )

        # 增强的通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 512),
            nn.Sigmoid()
        )

        # 添加空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=7, padding=3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.channel1_flattened + 512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
        )

        # 分类头
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, semantic=None, return_features=False):
        # 通道1处理
        x = x.unsqueeze(1)  # 添加通道维度 [batch_size, 1, length]
        channel1_out = self.channel1(x)
        channel1_out = channel1_out.reshape(channel1_out.size(0), -1)  # 展平

        # 如果没有提供语义向量，只用通道1的特征
        if semantic is None:
            if return_features:
                return channel1_out
            # 使用通道1特征进行分类
            fusion_out = self.fusion(channel1_out)
            logits = self.classifier(fusion_out)
            return logits, fusion_out

        # 通道2处理
        channel2_out = self.channel2_fc(semantic)

        # 应用通道注意力
        channel_attn = self.channel_attention(channel2_out)
        attended_semantic = channel2_out * channel_attn

        # 特征融合
        concatenated = torch.cat([channel1_out, attended_semantic], dim=1)
        fusion_out = self.fusion(concatenated)

        if return_features:
            return fusion_out

        # 分类
        logits = self.classifier(fusion_out)

        return logits, fusion_out


# 4. 语义嵌入网络
class SemanticEmbeddingNetwork(nn.Module):
    def __init__(self, semantic_dim, feature_dim=256):
        super(SemanticEmbeddingNetwork, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(semantic_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(256, feature_dim)
        )

        # 添加残差连接
        self.residual = nn.Linear(semantic_dim, feature_dim)

    def forward(self, semantic):
        embedded = self.embedding(semantic)
        residual = self.residual(semantic)
        output = embedded + residual  # 残差连接
        return output


# 5. 损失函数设计
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_value=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.lambda_value = lambda_value

    def forward(self, features, labels):
        batch_size = features.size(0)

        # 计算特征之间的余弦相似度
        features = F.normalize(features, dim=1)  # L2标准化
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建标签掩码
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

        # 排除对角线元素
        identity_mask = torch.eye(batch_size, device=features.device)
        mask = mask.float() - identity_mask

        # 计算正样本对的对比损失
        pos_mask = mask
        neg_mask = 1.0 - mask

        # 正样本损失：同类样本的相似度应该高
        pos_similarity = similarity_matrix * pos_mask
        pos_similarity = pos_similarity.sum(1) / (pos_mask.sum(1) + 1e-8)
        pos_loss = torch.mean((1.0 - pos_similarity) ** 2)

        # 负样本损失：不同类样本的相似度应该低
        neg_similarity = similarity_matrix * neg_mask
        neg_loss = torch.mean((neg_similarity) ** 2) * self.lambda_value

        # 总损失
        loss = pos_loss + neg_loss

        return loss


class FeatureSemanticConsistencyLoss(nn.Module):
    def __init__(self, beta=0.01):
        super(FeatureSemanticConsistencyLoss, self).__init__()
        self.beta = beta
        self.projection = None  # 动态创建的映射层

    def forward(self, features, semantics, weights=None):
        # 检查并创建映射层（如果需要）
        if self.projection is None or self.projection.in_features != semantics.size(
                1) or self.projection.out_features != features.size(1):
            self.projection = nn.Linear(semantics.size(1), features.size(1)).to(features.device)
            # 初始化投影矩阵
            nn.init.orthogonal_(self.projection.weight)

        # 将语义映射到特征维度
        projected_semantics = self.projection(semantics)

        # 标准化特征和语义向量
        features = F.normalize(features, dim=1)
        projected_semantics = F.normalize(projected_semantics, dim=1)

        # 计算加权的余弦相似度
        if weights is not None:
            # 将权重应用于特征
            weighted_features = features * weights.unsqueeze(0)
            similarity = torch.sum(weighted_features * projected_semantics, dim=1)
        else:
            # 标准余弦相似度
            similarity = torch.sum(features * projected_semantics, dim=1)

        # 一致性损失
        consistency_loss = 1.0 - torch.mean(similarity)

        # 添加L2正则化
        l2_reg = torch.mean(torch.norm(features, p=2, dim=1) ** 2)

        # 总损失
        loss = consistency_loss + self.beta * l2_reg

        return loss


# 6. 主程序：模型训练和评估
class ZeroShotCompoundFaultDiagnosis:
    def __init__(self, data_path, sample_length=1024, latent_dim=64, batch_size=64):
        self.data_path = data_path
        self.sample_length = sample_length
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 初始化组件
        self.preprocessor = DataPreprocessor(sample_length=sample_length)
        self.semantic_builder = FaultSemanticBuilder(latent_dim=latent_dim)

        # 故障类别映射
        self.fault_types = {
            'normal': 0,
            'inner': 1,
            'outer': 2,
            'ball': 3,
            'inner_outer': 4,
            'inner_ball': 5,
            'outer_ball': 6,
            'inner_outer_ball': 7
        }

        # 逆向映射
        self.idx_to_fault = {v: k for k, v in self.fault_types.items()}

        # 复合故障类别
        self.compound_fault_types = ['inner_outer', 'inner_ball', 'outer_ball', 'inner_outer_ball']

    def visualize_data_semantics_distribution(self, data_dict, semantic_dict):
        """Visualize the distribution of data semantics using t-SNE"""
        print("Visualizing data semantics distribution...")

        # Extract data semantics for training data
        data_semantics, _ = self.semantic_builder.extract_data_semantics(
            data_dict['X_train'],
            data_dict['y_train']
        )

        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        semantics_2d = tsne.fit_transform(data_semantics)

        # Prepare data for plotting
        fault_labels = [self.idx_to_fault[label] for label in data_dict['y_train']]
        unique_faults = sorted(list(set(fault_labels)))

        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'x': semantics_2d[:, 0],
            'y': semantics_2d[:, 1],
            'fault_type': fault_labels
        })

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='fault_type', palette='deep', s=100, alpha=0.7)
        plt.title('t-SNE Visualization of Data Semantics Distribution')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Fault Type', loc='best')
        plt.tight_layout()
        plt.savefig('data_semantics_distribution.png')
        plt.close()

        print("Data semantics distribution saved to 'data_semantics_distribution.png'")

    def load_data(self):
        """加载和预处理数据"""
        print("Loading and preprocessing data...")

        single_fault_data = {}
        single_fault_labels = []
        single_fault_features = []

        compound_fault_data = {}
        compound_fault_labels = []
        compound_fault_features = []

        # 加载单一故障数据（用于训练）
        single_fault_files = ['normal.mat', 'inner.mat', 'outer.mat', 'ball.mat']

        for file_name in single_fault_files:
            fault_type = file_name.split('.')[0]
            file_path = os.path.join(self.data_path, file_name)

            try:
                # 加载.mat文件
                mat_data = sio.loadmat(file_path)
                # 提取数据字段 (通常是'Data'或'X'或类似的字段)
                signal_data = None
                for key in mat_data.keys():
                    if isinstance(mat_data[key], np.ndarray) and not key.startswith('__'):
                        if mat_data[key].size > 1000:  # 假设数据字段是较大的数组
                            signal_data = mat_data[key]
                            break

                if signal_data is None:
                    print(f"No suitable data found in {file_name}")
                    continue

                # 确保信号是一维的
                if len(signal_data.shape) > 1:
                    signal_data = signal_data.ravel()

                # 截取指定长度
                if len(signal_data) > 200000:
                    signal_data = signal_data[:200000]

                # 数据预处理
                processed_segments = self.preprocessor.preprocess(signal_data, augmentation=True)

                # 存储处理后的数据和标签
                single_fault_data[fault_type] = processed_segments
                label_idx = self.fault_types[fault_type]
                single_fault_labels.extend([label_idx] * len(processed_segments))
                single_fault_features.append(processed_segments)

                print(f"Processed {fault_type}: {processed_segments.shape}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        # 加载复合故障数据（用于测试）
        compound_fault_files = ['inner_ball.mat', 'inner_outer.mat', 'outer_ball.mat', 'inner_outer_ball.mat']

        for file_name in compound_fault_files:
            fault_type = file_name.split('.')[0]
            file_path = os.path.join(self.data_path, file_name)

            try:
                # 加载.mat文件
                mat_data = sio.loadmat(file_path)
                # 提取数据字段
                signal_data = None
                for key in mat_data.keys():
                    if isinstance(mat_data[key], np.ndarray) and not key.startswith('__'):
                        if mat_data[key].size > 1000:
                            signal_data = mat_data[key]
                            break

                if signal_data is None:
                    print(f"No suitable data found in {file_name}")
                    continue

                # 确保信号是一维的
                if len(signal_data.shape) > 1:
                    signal_data = signal_data.ravel()

                # 截取指定长度
                if len(signal_data) > 200000:
                    signal_data = signal_data[:200000]

                # 数据预处理（不进行增强）
                processed_segments = self.preprocessor.preprocess(signal_data, augmentation=False)

                # 存储处理后的数据和标签
                compound_fault_data[fault_type] = processed_segments
                label_idx = self.fault_types[fault_type]
                compound_fault_labels.extend([label_idx] * len(processed_segments))
                compound_fault_features.append(processed_segments)

                print(f"Processed {fault_type}: {processed_segments.shape}")

            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        # 将特征和标签转换为数组
        single_fault_features = np.vstack(single_fault_features) if single_fault_features else np.array([])
        single_fault_labels = np.array(single_fault_labels)

        compound_fault_features = np.vstack(compound_fault_features) if compound_fault_features else np.array([])
        compound_fault_labels = np.array(compound_fault_labels)

        # 划分训练集和验证集
        if len(single_fault_features) > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                single_fault_features, single_fault_labels, test_size=0.3, random_state=42, stratify=single_fault_labels
            )

            return {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': compound_fault_features,
                'y_test': compound_fault_labels,
                'single_fault_data': single_fault_data,
                'compound_fault_data': compound_fault_data
            }
        else:
            print("No single fault data loaded.")
            return None

    def build_semantics(self, data_dict):
        """构建知识语义和数据语义 (双通道CNN仅使用数据语义，但语义映射使用融合语义)"""
        print("Building fault semantics...")

        # 构建知识语义
        knowledge_semantics = self.semantic_builder.build_knowledge_semantics()

        # 训练自编码器提取数据语义
        print("Training autoencoder for data semantics...")
        self.semantic_builder.train_autoencoder(
            data_dict['X_train'],
            labels=data_dict['y_train'],  # Pass the labels
            epochs=100,
            batch_size=self.batch_size
        )

        # 提取单一故障的数据语义
        _, single_fault_prototypes = self.semantic_builder.extract_data_semantics(
            data_dict['X_train'],
            data_dict['y_train']
        )

        # 将数字标签转换为故障类型名称
        prototype_semantics = {}
        for label_idx, prototype in single_fault_prototypes.items():
            fault_type = self.idx_to_fault[label_idx]
            prototype_semantics[fault_type] = prototype

        # 合成复合故障的数据语义
        compound_data_semantics = self.semantic_builder.synthesize_compound_semantics(prototype_semantics)

        # 准备CNN训练用的数据语义 (仅使用数据语义)
        data_only_semantics = {}

        # 添加单一故障的数据语义
        for fault_type in prototype_semantics.keys():
            data_only_semantics[fault_type] = prototype_semantics[fault_type]

        # 添加复合故障的数据语义
        for fault_type, semantics in compound_data_semantics.items():
            data_only_semantics[fault_type] = semantics

        # 准备用于语义映射和特征空间的融合语义
        fused_semantics = {}
        for fault_type in list(knowledge_semantics.keys()):
            if fault_type in prototype_semantics:
                # 单一故障：融合知识语义和数据语义
                knowledge_vec = knowledge_semantics[fault_type]
                data_vec = prototype_semantics[fault_type]
                fused_semantics[fault_type] = np.concatenate([knowledge_vec, data_vec])
            elif fault_type in compound_data_semantics:
                # 复合故障：融合知识语义和合成的数据语义
                knowledge_vec = knowledge_semantics[fault_type]
                data_vec = compound_data_semantics[fault_type]
                fused_semantics[fault_type] = np.concatenate([knowledge_vec, data_vec])

        return {
            'knowledge_semantics': knowledge_semantics,
            'data_semantics': prototype_semantics,
            'compound_data_semantics': compound_data_semantics,
            'data_only_semantics': data_only_semantics,  # CNN训练时使用
            'fused_semantics': fused_semantics  # 语义映射时使用
        }

    def visualize_semantics(self, semantic_dict):
        """可视化语义相似度矩阵"""
        print("Visualizing semantic similarity matrices...")

        def compute_similarity_matrix(semantics_dict):
            # 提取所有故障类型和对应的语义向量
            fault_types = list(semantics_dict.keys())
            vectors = [semantics_dict[ft] for ft in fault_types]

            # 计算相似度矩阵
            n = len(vectors)
            sim_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    # 使用余弦相似度
                    vec_i = vectors[i] / np.linalg.norm(vectors[i])
                    vec_j = vectors[j] / np.linalg.norm(vectors[j])
                    sim_matrix[i, j] = np.dot(vec_i, vec_j)

            return sim_matrix, fault_types

        # 计算数据语义的相似度矩阵
        d_sim_mat, d_labels = compute_similarity_matrix(semantic_dict['data_only_semantics'])

        # 计算融合语义的相似度矩阵
        f_sim_mat, f_labels = compute_similarity_matrix(semantic_dict['fused_semantics'])

        # 绘制相似度矩阵热图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.heatmap(d_sim_mat, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=d_labels,
                    yticklabels=d_labels, ax=axes[0])
        axes[0].set_title('Data Semantics Similarity (Used for CNN)')

        sns.heatmap(f_sim_mat, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=f_labels,
                    yticklabels=f_labels, ax=axes[1])
        axes[1].set_title('Fused Semantics Similarity (Used for Mapping)')

        plt.tight_layout()
        plt.savefig('semantic_similarity_matrices.png')
        plt.close()

        print("Semantic similarity matrices saved to 'semantic_similarity_matrices.png'")

    def train_dual_channel_cnn(self, data_dict, semantic_dict, epochs=100, lr=0.001):
        """训练双通道CNN模型 (仅使用数据语义)"""
        print("Training dual channel CNN model with data semantics only...")

        # 准备数据
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']

        # 获取数据语义向量 (不使用知识语义)
        data_semantics = semantic_dict['data_only_semantics']

        # 创建语义向量字典 - 标签索引到语义向量的映射
        semantic_vectors = {}
        for fault_type, idx in self.fault_types.items():
            if fault_type in data_semantics:
                semantic_vectors[idx] = data_semantics[fault_type]

        # 初始化模型
        input_length = X_train.shape[1]
        semantic_dim = next(iter(data_semantics.values())).shape[0]  # 获取数据语义向量的维度

        self.cnn_model = DualChannelCNN(
            input_length=input_length,
            semantic_dim=semantic_dim,  # 使用数据语义的维度
            num_classes=len(self.fault_types)
        ).to(self.device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        contrastive_loss = ContrastiveLoss().to(self.device)
        consistency_loss = FeatureSemanticConsistencyLoss().to(self.device)

        optimizer = optim.Adam(self.cnn_model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        # 训练循环
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10  # 早停参数

        for epoch in range(epochs):
            # 训练阶段
            self.cnn_model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            # 动态调整对比损失权重
            epoch_ratio = epoch / epochs
            lambda_value = 0.1 + 0.5 * epoch_ratio  # 从0.1线性增加到0.6，增加对比效果
            contrastive_loss.lambda_value = lambda_value

            # 动态调整温度参数，随训练进程降低温度
            contrastive_loss.temperature = max(0.02, 0.07 - 0.05 * epoch_ratio)

            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 为每个样本获取对应的语义向量 (仅数据语义)
                batch_semantics = []
                for label in labels.cpu().numpy():
                    if label in semantic_vectors:
                        batch_semantics.append(semantic_vectors[label])
                    else:
                        # 如果没有找到对应的语义向量，使用零向量
                        batch_semantics.append(np.zeros(semantic_dim))

                batch_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                # 前向传播
                logits, features = self.cnn_model(inputs, batch_semantics)

                # 计算分类损失
                ce_loss = criterion(logits, labels)

                # 计算对比损失 - 增强类间差异
                contr_loss = contrastive_loss(features, labels)

                # 计算特征-语义一致性损失 - 确保特征与语义对齐
                consist_loss = consistency_loss(features, batch_semantics)

                # 总损失 - 动态调整权重
                consist_weight = 0.05 + 0.15 * epoch_ratio  # 从0.05增加到0.2
                loss = ce_loss + lambda_value * contr_loss + consist_weight * consist_loss

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=1.0)

                optimizer.step()

                # 计算训练准确率
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_loss += loss.item() * inputs.size(0)

            # 更新学习率
            scheduler.step()

            # 计算平均训练损失和准确率
            avg_train_loss = train_loss / len(train_loader.dataset)
            train_accuracy = 100.0 * correct / total

            # 验证阶段
            self.cnn_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # 为每个样本获取对应的语义向量
                    batch_semantics = []
                    for label in labels.cpu().numpy():
                        if label in semantic_vectors:
                            batch_semantics.append(semantic_vectors[label])
                        else:
                            batch_semantics.append(np.zeros(semantic_dim))

                    batch_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)

                    # 前向传播
                    logits, _ = self.cnn_model(inputs, batch_semantics)

                    # 计算损失
                    loss = criterion(logits, labels)
                    val_loss += loss.item() * inputs.size(0)

                    # 计算准确率
                    _, predicted = torch.max(logits, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            # 计算平均验证损失和准确率
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = 100.0 * val_correct / val_total

            print(f"Epoch [{epoch + 1}/{epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                  f"λ: {lambda_value:.2f}, Temp: {contrastive_loss.temperature:.3f}")

            # 早停检查
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.cnn_model.state_dict(), 'best_cnn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # 加载最佳模型
        self.cnn_model.load_state_dict(torch.load('best_cnn_model.pth'))
        self.cnn_model.eval()
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")

    def train_semantic_embedding(self, semantic_dict, data_dict):
        """训练语义嵌入网络，将语义向量映射到特征空间 (使用融合语义)"""
        print("Training semantic embedding network with fused semantics...")

        # 获取融合语义向量 (使用知识语义和数据语义融合)
        fused_semantics = semantic_dict['fused_semantics']

        # 准备数据：仅使用单一故障的样本
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']

        # 首先获取CNN模型输出的特征维度
        with torch.no_grad():
            sample_x = torch.FloatTensor(X_train[:1]).to(self.device)
            sample_features = self.cnn_model(sample_x, return_features=True)
            feature_dim = sample_features.size(1)

        print(f"CNN feature dimension: {feature_dim}")

        # 初始化语义嵌入网络，使用正确的特征维度
        semantic_dim = next(iter(fused_semantics.values())).shape[0]

        self.embedding_net = SemanticEmbeddingNetwork(
            semantic_dim=semantic_dim,
            feature_dim=feature_dim  # 使用CNN模型的特征维度
        ).to(self.device)

        # 定义损失函数和优化器
        mse_loss = nn.MSELoss()
        cosine_loss = nn.CosineEmbeddingLoss(margin=0.1)  # 增加margin以增强分离
        triplet_loss = nn.TripletMarginLoss(margin=0.5)  # 添加三元组损失增强语义差异
        optimizer = optim.Adam(self.embedding_net.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # 训练嵌入网络
        epochs = 150  # 增加训练轮数
        batch_size = 32

        # 将数据集分批次处理
        num_samples = len(X_train)
        num_batches = int(np.ceil(num_samples / batch_size))

        for epoch in range(epochs):
            total_loss = 0.0

            # 随机打乱索引
            indices = np.random.permutation(num_samples)

            for i in range(num_batches):
                # 获取当前批次的索引
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # 获取当前批次的样本
                batch_x = X_train[batch_indices]
                batch_y = y_train[batch_indices]

                # 构建语义向量批次 (使用融合语义)
                batch_semantics = []
                for label in batch_y:
                    fault_type = self.idx_to_fault[label]
                    if fault_type in fused_semantics:
                        batch_semantics.append(fused_semantics[fault_type])
                    else:
                        # 如果没有找到对应的语义向量，使用零向量
                        batch_semantics.append(np.zeros(semantic_dim))

                batch_semantics = torch.FloatTensor(np.array(batch_semantics)).to(self.device)
                batch_x = torch.FloatTensor(batch_x).to(self.device)

                # 使用CNN模型提取样本特征 (CNN训练时使用的是数据语义)
                with torch.no_grad():
                    features = self.cnn_model(batch_x, return_features=True)

                # 将融合语义向量映射到特征空间
                embedded_semantics = self.embedding_net(batch_semantics)

                # 计算MSE损失
                loss_mse = mse_loss(embedded_semantics, features)

                # 计算余弦相似度损失
                loss_cos = cosine_loss(
                    embedded_semantics,
                    features,
                    torch.ones(len(batch_indices)).to(self.device)
                )

                # 计算三元组损失（仅当有足够不同的标签时）
                unique_labels = torch.unique(torch.tensor(batch_y))
                if len(unique_labels) >= 2:
                    loss_triplet = 0.0
                    num_triplets = 0

                    # 为每个样本找到正样本和负样本
                    for j, label in enumerate(batch_y):
                        # 找同类样本作为正样本
                        positive_indices = np.where(batch_y == label)[0]
                        positive_indices = positive_indices[positive_indices != j]  # 排除自己

                        if len(positive_indices) > 0:
                            # 找不同类样本作为负样本
                            negative_indices = np.where(batch_y != label)[0]

                            if len(negative_indices) > 0:
                                # 随机选择一个正样本和一个负样本
                                pos_idx = np.random.choice(positive_indices)
                                neg_idx = np.random.choice(negative_indices)

                                # 提取对应的嵌入向量
                                anchor = embedded_semantics[j]
                                positive = features[pos_idx]
                                negative = features[neg_idx]

                                # 累加三元组损失
                                loss_triplet += triplet_loss(anchor.unsqueeze(0),
                                                             positive.unsqueeze(0),
                                                             negative.unsqueeze(0))
                                num_triplets += 1

                    if num_triplets > 0:
                        loss_triplet = loss_triplet / num_triplets
                else:
                    loss_triplet = torch.tensor(0.0).to(self.device)

                # 总损失 - 动态调整权重
                epoch_ratio = epoch / epochs
                triplet_weight = 0.1 + 0.3 * epoch_ratio  # 从0.1增加到0.4
                cos_weight = 0.3

                loss = loss_mse + cos_weight * loss_cos + triplet_weight * loss_triplet

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.embedding_net.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item() * len(batch_indices)

            # 计算平均损失
            avg_loss = total_loss / num_samples
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}")

            # 更新学习率
            scheduler.step(avg_loss)

            # 每30个epoch保存一次检查点
            if (epoch + 1) % 30 == 0:
                torch.save(self.embedding_net.state_dict(), f'semantic_embedding_net_ep{epoch + 1}.pth')

        # 保存嵌入网络
        torch.save(self.embedding_net.state_dict(), 'semantic_embedding_net.pth')
        print("Semantic embedding network training completed.")

    def generate_compound_fault_projections(self, semantic_dict):
        """生成复合故障的投影特征 (使用融合语义)"""
        print("Generating compound fault projections using fused semantics...")

        # 获取复合故障融合语义
        compound_semantics = {}
        fused_semantics = semantic_dict['fused_semantics']  # 使用融合语义

        for fault_type in self.compound_fault_types:
            if fault_type in fused_semantics:
                compound_semantics[fault_type] = fused_semantics[fault_type]

        # 将复合故障语义映射到特征空间
        self.embedding_net.eval()

        compound_projections = {}

        with torch.no_grad():
            for fault_type, semantic_vec in compound_semantics.items():
                semantic_tensor = torch.FloatTensor(semantic_vec).unsqueeze(0).to(self.device)
                projected_feature = self.embedding_net(semantic_tensor)
                compound_projections[fault_type] = projected_feature.cpu().numpy().squeeze()

        # 可视化复合故障投影的相似度
        self._visualize_compound_projections(compound_projections)

        return compound_projections

    def _visualize_compound_projections(self, compound_projections):
        """可视化复合故障投影向量的相似度"""
        # 提取故障类型和对应的投影向量
        fault_types = list(compound_projections.keys())
        vectors = [compound_projections[ft] for ft in fault_types]

        # 计算相似度矩阵
        n = len(vectors)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # 使用余弦相似度
                vec_i = vectors[i] / np.linalg.norm(vectors[i])
                vec_j = vectors[j] / np.linalg.norm(vectors[j])
                sim_matrix[i, j] = np.dot(vec_i, vec_j)

        # 绘制相似度热图
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    xticklabels=fault_types, yticklabels=fault_types)
        plt.title('Compound Fault Projection Similarity Matrix')
        plt.tight_layout()
        plt.savefig('compound_projection_similarity.png')
        plt.close()

        print("Compound projection similarity matrix saved to 'compound_projection_similarity.png'")

    def evaluate_zero_shot(self, data_dict, compound_projections):
        """零样本评估复合故障分类性能"""
        print("Evaluating zero-shot compound fault classification...")

        # 获取测试数据
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']

        # 仅保留复合故障的测试样本
        compound_indices = []
        for i, label in enumerate(y_test):
            fault_type = self.idx_to_fault[label]
            if fault_type in self.compound_fault_types:
                compound_indices.append(i)

        X_compound = X_test[compound_indices]
        y_compound = y_test[compound_indices]

        # 创建复合故障标签到索引的映射
        compound_to_idx = {fault: idx for idx, fault in enumerate(self.compound_fault_types)}

        # 将原始标签转换为新的标签（仅针对复合故障）
        y_true = np.array([compound_to_idx[self.idx_to_fault[label]] for label in y_compound])

        # 预测结果
        y_pred = []
        similarity_scores = []  # 存储相似度分数用于可视化
        batch_size = 32
        num_samples = len(X_compound)
        num_batches = int(np.ceil(num_samples / batch_size))

        self.cnn_model.eval()

        # 计算每个测试样本与每个复合故障投影的相似度
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                batch_x = torch.FloatTensor(X_compound[start_idx:end_idx]).to(self.device)

                # 提取特征
                batch_features = self.cnn_model(batch_x, return_features=True)

                # 计算与每个复合故障投影的相似度
                batch_pred = []
                batch_scores = []

                for j in range(batch_features.size(0)):
                    feature = batch_features[j].cpu().numpy()

                    # 计算余弦相似度
                    similarities = []
                    for fault_type in self.compound_fault_types:
                        proj = compound_projections[fault_type]
                        similarity = np.dot(feature, proj) / (np.linalg.norm(feature) * np.linalg.norm(proj))
                        similarities.append(similarity)

                    # 存储相似度分数
                    batch_scores.append(similarities)

                    # 选择最相似的复合故障类型
                    pred_idx = np.argmax(similarities)
                    batch_pred.append(pred_idx)

                y_pred.extend(batch_pred)
                similarity_scores.extend(batch_scores)

        # 转换为numpy数组
        y_pred = np.array(y_pred)
        similarity_scores = np.array(similarity_scores)

        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred) * 100

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_true, y_pred)

        print(f"Zero-shot compound fault classification accuracy: {accuracy:.2f}%")

        # 可视化混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.compound_fault_types,
                    yticklabels=self.compound_fault_types)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix for Compound Faults')
        plt.tight_layout()
        plt.savefig('compound_fault_confusion_matrix.png')
        plt.close()

        # 可视化相似度分数分布
        self._visualize_similarity_scores(similarity_scores, y_true)

        print("Confusion matrix saved to 'compound_fault_confusion_matrix.png'")

        return accuracy, conf_matrix

    def _visualize_similarity_scores(self, similarity_scores, true_labels):
        """可视化测试样本与各复合故障原型的相似度分数分布"""
        plt.figure(figsize=(12, 8))

        # 为每种故障类型计算每个复合故障相似度的平均值
        for i, fault_type in enumerate(self.compound_fault_types):
            # 过滤出该类型故障的样本
            indices = np.where(true_labels == i)[0]
            if len(indices) > 0:
                class_scores = similarity_scores[indices]
                mean_scores = np.mean(class_scores, axis=0)

                # 创建柱状图
                x = np.arange(len(self.compound_fault_types))
                plt.bar(x + i * 0.2, mean_scores, width=0.2,
                        label=f'True: {fault_type}')

        plt.xlabel('Compound Fault Types')
        plt.ylabel('Average Similarity Score')
        plt.title('Similarity Scores between Test Samples and Compound Fault Prototypes')
        plt.xticks(np.arange(len(self.compound_fault_types)) + 0.3,
                   self.compound_fault_types, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('compound_fault_similarity_scores.png')
        plt.close()

        print("Similarity scores visualization saved to 'compound_fault_similarity_scores.png'")

    def run_pipeline(self):
        """运行完整的零样本复合轴承故障诊断流程"""
        # 1. 加载和预处理数据
        data_dict = self.load_data()
        if data_dict is None:
            print("Data loading failed. Exiting.")
            return

        # 2. 构建知识语义和数据语义 (生成两种语义)
        semantic_dict = self.build_semantics(data_dict)

        # 3. 可视化语义相似度矩阵
        self.visualize_semantics(semantic_dict)

        # 4. 可视化数据语义分布
        self.visualize_data_semantics_distribution(data_dict, semantic_dict)

        # 5. 训练双通道CNN模型（使用单一故障数据+仅数据语义）
        self.train_dual_channel_cnn(data_dict, semantic_dict)

        # 6. 训练语义嵌入网络 (使用融合语义)
        self.train_semantic_embedding(semantic_dict, data_dict)

        # 7. 生成复合故障投影特征 (使用融合语义)
        compound_projections = self.generate_compound_fault_projections(semantic_dict)

        # 8. 零样本评估复合故障分类性能
        accuracy, _ = self.evaluate_zero_shot(data_dict, compound_projections)

        return accuracy


# 主程序
if __name__ == "__main__":
    # 设置随机种子
    set_seed(42)

    # 数据路径，需要根据实际情况修改
    data_path = "E:/研究生/CNN/HDUCESHI"

    # 创建零样本复合轴承故障诊断实例
    fault_diagnosis = ZeroShotCompoundFaultDiagnosis(
        data_path=data_path,
        sample_length=1024,
        latent_dim=64,
        batch_size=64
    )

    # 运行完整流程
    accuracy = fault_diagnosis.run_pipeline()

    print(f"Final zero-shot compound fault classification accuracy: {accuracy:.2f}%")
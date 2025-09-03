# -*- coding: utf-8 -*-
"""
CNN到VAE语义空间投影分析系统
CNN to VAE Semantic Space Projection Analysis System
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import pywt
from scipy.signal import butter, sosfiltfilt
import time
import warnings

warnings.filterwarnings('ignore')

# Import VAE from existing claude.py
try:
    from claude import BetaVAE_H, kl_divergence, reparametrize
    VAE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import VAE components from claude.py: {e}")
    VAE_AVAILABLE = False
    
    # Define fallback VAE components
    import torch.nn as nn
    import torch
    from torch.autograd import Variable
    
    def reparametrize(mu, logvar):
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps
    
    def kl_divergence(mu, logvar):
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        total_kld = klds.sum(1).mean(0, True)
        return total_kld
    
    class View(nn.Module):
        def __init__(self, size):
            super(View, self).__init__()
            self.size = size
        
        def forward(self, tensor):
            return tensor.view(self.size)
    
    class BetaVAE_H(nn.Module):
        def __init__(self, z_dim=32, nc=3):
            super(BetaVAE_H, self).__init__()
            self.z_dim = z_dim
            self.nc = nc
            
            # 编码器网络
            self.encoder = nn.Sequential(
                nn.Conv2d(nc, 32, 4, 2, 1), nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),
                nn.Conv2d(64, 256, 4, 1), nn.ReLU(True),
                View((-1, 256*1*1)),
                nn.Linear(256, z_dim*2),
            )
            
            # 解码器网络
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, 256),
                View((-1, 256, 1, 1)),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 64, 4), nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True),
                nn.ConvTranspose2d(32, nc, 4, 2, 1),
            )

        def forward(self, x):
            distributions = self.encoder(x)
            mu = distributions[:, :self.z_dim]
            logvar = distributions[:, self.z_dim:]
            z = reparametrize(mu, logvar)
            x_recon = self.decoder(z)
            return x_recon, mu, logvar

# Import configurations
try:
    from config import (
        VAE_CONFIG, PROJECTION_CONFIG, CWT_CONFIG, 
        VISUALIZATION_CONFIG, SAVE_CONFIG, CNN_FEATURE_DIM
    )
except ImportError:
    print("Warning: Could not import configurations, using defaults")
    VAE_CONFIG = {'z_dim': 32, 'nc': 3, 'beta': 4.0, 'epochs': 50, 'batch_size': 64, 'learning_rate': 1e-4}
    PROJECTION_CONFIG = {'projector_hidden_dims': [512, 256, 128], 'projector_epochs': 30, 'projector_lr': 1e-3}
    CWT_CONFIG = {'sampling_rate': 12000, 'freq_range': (800, 4000), 'output_size': (64, 64)}
    VISUALIZATION_CONFIG = {'plot_projection_results': True}
    SAVE_CONFIG = {'results_dir': 'projection_results'}
    CNN_FEATURE_DIM = 512


class ProjectionNetwork(nn.Module):
    """
    投影网络：将CNN特征映射到VAE语义空间
    Projection Network: Maps CNN features to VAE semantic space
    """
    
    def __init__(self, input_dim=CNN_FEATURE_DIM, output_dim=VAE_CONFIG['z_dim'], 
                 hidden_dims=None):
        super(ProjectionNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = PROJECTION_CONFIG.get('projector_hidden_dims', [512, 256, 128])
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建投影网络层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        return self.projection(x)


class CWTImageGenerator:
    """
    CWT图像生成器：为VAE训练生成时频图像
    CWT Image Generator: Generates time-frequency images for VAE training
    """
    
    def __init__(self, config=None):
        self.config = config or CWT_CONFIG
        self.sampling_rate = self.config['sampling_rate']
        self.freq_range = self.config['freq_range']
        self.output_size = self.config['output_size']
        self.wavelet = self.config.get('wavelet', 'cmor1.5-1.0')
    
    def preprocess_signal(self, signal, cutoff_hz=None):
        """信号预处理：滤波"""
        if cutoff_hz is None:
            cutoff_hz = self.freq_range[0]
        
        def filter_signal(s, btype):
            try:
                sos = butter(N=4, Wn=cutoff_hz / (0.5 * self.sampling_rate), 
                           btype=btype, analog=False, output='sos')
                filtered = sosfiltfilt(sos, s)
                return filtered
            except:
                return s
        
        high_freq_signal = filter_signal(signal, 'high')
        return high_freq_signal
    
    def generate_cwt_image(self, signal):
        """
        生成CWT时频图像
        Generate CWT time-frequency image
        """
        try:
            # 预处理信号
            processed_signal = self.preprocess_signal(signal)
            
            # 计算CWT的尺度参数
            freq_min, freq_max = self.freq_range
            num_scales = 64
            
            # 生成尺度数组
            scales = np.logspace(
                np.log10(self.sampling_rate / freq_max), 
                np.log10(self.sampling_rate / freq_min), 
                num_scales
            )
            
            # 执行CWT变换
            coefficients, frequencies = pywt.cwt(processed_signal, scales, self.wavelet, 
                                                sampling_period=1.0/self.sampling_rate)
            
            # 获取幅度
            cwt_magnitude = np.abs(coefficients)
            
            # 归一化
            cwt_magnitude = (cwt_magnitude - np.min(cwt_magnitude)) / \
                          (np.max(cwt_magnitude) - np.min(cwt_magnitude) + 1e-8)
            
            # 调整到目标尺寸
            from scipy.ndimage import zoom
            target_h, target_w = self.output_size
            current_h, current_w = cwt_magnitude.shape
            
            zoom_factors = (target_h / current_h, target_w / current_w)
            cwt_resized = zoom(cwt_magnitude, zoom_factors, order=1)
            
            # 转换为3通道图像（RGB）
            cwt_image = np.stack([cwt_resized] * 3, axis=0)  # [3, H, W]
            
            return cwt_image.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: CWT generation failed: {e}")
            # 返回随机图像作为备选
            return np.random.rand(3, *self.output_size).astype(np.float32)


class CNNToVAEProjectionSystem:
    """
    CNN到VAE语义空间投影分析系统
    CNN to VAE Semantic Space Projection Analysis System
    """
    
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Projection system using device: {self.device}")
        
        # 初始化组件
        self.vae_model = None
        self.projection_net = None
        self.cwt_generator = CWTImageGenerator()
        
        # 存储特征和数据
        self.cnn_features = None
        self.vae_features = None
        self.projected_features = None
        self.labels = None
        
        # 性能指标
        self.alignment_metrics = {}
        
        # 创建结果目录
        self.results_dir = SAVE_CONFIG.get('results_dir', 'projection_results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    def prepare_vae_data(self, signal_data, labels):
        """
        为VAE训练准备CWT图像数据
        Prepare CWT image data for VAE training
        """
        print("Preparing CWT images for VAE training...")
        
        cwt_images = []
        valid_labels = []
        
        for i, signal in enumerate(tqdm(signal_data, desc="Generating CWT images")):
            try:
                cwt_image = self.cwt_generator.generate_cwt_image(signal)
                if cwt_image is not None and np.all(np.isfinite(cwt_image)):
                    cwt_images.append(cwt_image)
                    valid_labels.append(labels[i])
            except Exception as e:
                print(f"Warning: Failed to generate CWT image for signal {i}: {e}")
        
        if not cwt_images:
            raise ValueError("No valid CWT images generated")
        
        cwt_images = np.array(cwt_images)
        valid_labels = np.array(valid_labels)
        
        print(f"Generated {len(cwt_images)} CWT images for VAE training")
        return cwt_images, valid_labels
    
    def train_vae(self, signal_data, labels):
        """
        训练VAE模型
        Train VAE model
        """
        print("Training VAE model...")
        
        # 准备CWT图像数据
        cwt_images, valid_labels = self.prepare_vae_data(signal_data, labels)
        
        # 初始化VAE模型
        vae_config = VAE_CONFIG
        self.vae_model = BetaVAE_H(
            z_dim=vae_config['z_dim'], 
            nc=vae_config['nc']
        ).to(self.device)
        
        # 数据加载器
        dataset = TensorDataset(
            torch.FloatTensor(cwt_images), 
            torch.LongTensor(valid_labels)
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=vae_config['batch_size'], 
            shuffle=True, 
            drop_last=True
        )
        
        # 优化器
        optimizer = optim.Adam(
            self.vae_model.parameters(), 
            lr=vae_config['learning_rate']
        )
        
        # 训练循环
        self.vae_model.train()
        for epoch in range(vae_config['epochs']):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            
            for batch_idx, (data, _) in enumerate(dataloader):
                data = data.to(self.device)
                
                optimizer.zero_grad()
                
                # VAE前向传播
                recon_data, mu, logvar = self.vae_model(data)
                
                # 损失计算
                recon_loss = F.mse_loss(recon_data, data, reduction='sum')
                kl_loss = kl_divergence(mu, logvar)
                total_loss = recon_loss + vae_config['beta'] * kl_loss
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                avg_recon = epoch_recon_loss / len(dataloader)
                avg_kl = epoch_kl_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{vae_config['epochs']}] "
                      f"Loss: {avg_loss:.4f} "
                      f"(Recon: {avg_recon:.4f}, KL: {avg_kl:.4f})")
        
        # 保存VAE模型
        if SAVE_CONFIG.get('save_vae_model', True):
            model_path = os.path.join(self.results_dir, 'vae_model.pth')
            torch.save(self.vae_model.state_dict(), model_path)
            print(f"VAE model saved to {model_path}")
        
        self.vae_model.eval()
        return cwt_images, valid_labels
    
    def extract_vae_features(self, cwt_images):
        """
        提取VAE特征
        Extract VAE features
        """
        print("Extracting VAE features...")
        
        if self.vae_model is None:
            raise ValueError("VAE model not trained")
        
        self.vae_model.eval()
        vae_features = []
        
        batch_size = VAE_CONFIG.get('batch_size', 64)
        
        with torch.no_grad():
            for i in range(0, len(cwt_images), batch_size):
                batch = cwt_images[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                _, mu, _ = self.vae_model(batch_tensor)
                vae_features.append(mu.cpu().numpy())
        
        vae_features = np.vstack(vae_features)
        self.vae_features = vae_features
        
        print(f"Extracted VAE features shape: {vae_features.shape}")
        return vae_features
    
    def train_projection_network(self, cnn_features, vae_features, labels):
        """
        训练投影网络
        Train projection network
        """
        print("Training projection network...")
        
        # 确保特征维度匹配
        if len(cnn_features) != len(vae_features):
            min_len = min(len(cnn_features), len(vae_features))
            cnn_features = cnn_features[:min_len]
            vae_features = vae_features[:min_len]
            labels = labels[:min_len]
        
        # 初始化投影网络
        self.projection_net = ProjectionNetwork(
            input_dim=cnn_features.shape[1],
            output_dim=vae_features.shape[1]
        ).to(self.device)
        
        # 数据加载器
        dataset = TensorDataset(
            torch.FloatTensor(cnn_features),
            torch.FloatTensor(vae_features),
            torch.LongTensor(labels)
        )
        
        batch_size = PROJECTION_CONFIG.get('projector_batch_size', 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # 优化器
        optimizer = optim.Adam(
            self.projection_net.parameters(),
            lr=PROJECTION_CONFIG.get('projector_lr', 1e-3)
        )
        
        # 损失函数
        mse_loss = nn.MSELoss()
        cosine_loss = nn.CosineEmbeddingLoss()
        
        # 训练循环
        epochs = PROJECTION_CONFIG.get('projector_epochs', 30)
        self.projection_net.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_align_loss = 0.0
            epoch_cosine_loss = 0.0
            
            for cnn_batch, vae_batch, label_batch in dataloader:
                cnn_batch = cnn_batch.to(self.device)
                vae_batch = vae_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # 投影预测
                projected = self.projection_net(cnn_batch)
                
                # 对齐损失（MSE）
                align_loss = mse_loss(projected, vae_batch)
                
                # 余弦相似性损失
                cos_loss = cosine_loss(
                    projected, vae_batch, 
                    torch.ones(len(projected)).to(self.device)
                )
                
                # 总损失
                total_loss = (PROJECTION_CONFIG.get('alignment_loss_weight', 1.0) * align_loss + 
                             PROJECTION_CONFIG.get('contrastive_loss_weight', 0.5) * cos_loss)
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_align_loss += align_loss.item()
                epoch_cosine_loss += cos_loss.item()
            
            # 打印训练进度
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / len(dataloader)
                avg_align = epoch_align_loss / len(dataloader)
                avg_cos = epoch_cosine_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Loss: {avg_loss:.4f} "
                      f"(Align: {avg_align:.4f}, Cosine: {avg_cos:.4f})")
        
        # 保存投影网络
        if SAVE_CONFIG.get('save_projection_model', True):
            model_path = os.path.join(self.results_dir, 'projection_network.pth')
            torch.save(self.projection_net.state_dict(), model_path)
            print(f"Projection network saved to {model_path}")
        
        self.projection_net.eval()
    
    def project_features(self, cnn_features):
        """
        使用训练好的投影网络投影CNN特征
        Project CNN features using trained projection network
        """
        if self.projection_net is None:
            raise ValueError("Projection network not trained")
        
        print("Projecting CNN features to VAE semantic space...")
        
        self.projection_net.eval()
        projected_features = []
        
        batch_size = PROJECTION_CONFIG.get('projector_batch_size', 32)
        
        with torch.no_grad():
            for i in range(0, len(cnn_features), batch_size):
                batch = cnn_features[i:i+batch_size]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                projected = self.projection_net(batch_tensor)
                projected_features.append(projected.cpu().numpy())
        
        projected_features = np.vstack(projected_features)
        self.projected_features = projected_features
        
        print(f"Projected features shape: {projected_features.shape}")
        return projected_features
    
    def evaluate_projection_quality(self):
        """
        评估投影质量
        Evaluate projection quality
        """
        if self.projected_features is None or self.vae_features is None:
            raise ValueError("Projected or VAE features not available")
        
        print("Evaluating projection quality...")
        
        # 确保特征长度匹配
        min_len = min(len(self.projected_features), len(self.vae_features))
        projected = self.projected_features[:min_len]
        vae_feats = self.vae_features[:min_len]
        
        # MSE损失
        mse = mean_squared_error(vae_feats, projected)
        
        # 余弦相似性
        cosine_similarities = []
        for i in range(len(projected)):
            cos_sim = cosine_similarity(
                projected[i:i+1], vae_feats[i:i+1]
            )[0, 0]
            cosine_similarities.append(cos_sim)
        
        avg_cosine_sim = np.mean(cosine_similarities)
        
        # 特征空间对齐度（基于t-SNE）
        alignment_score = self._compute_alignment_score(projected, vae_feats)
        
        self.alignment_metrics = {
            'mse_loss': mse,
            'cosine_similarity': avg_cosine_sim,
            'alignment_score': alignment_score
        }
        
        print(f"Projection Quality Metrics:")
        print(f"  MSE Loss: {mse:.6f}")
        print(f"  Cosine Similarity: {avg_cosine_sim:.4f}")
        print(f"  Alignment Score: {alignment_score:.4f}")
        
        return self.alignment_metrics
    
    def _compute_alignment_score(self, projected, vae_feats):
        """计算特征空间对齐分数"""
        try:
            # 使用t-SNE降维
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(projected)//4))
            
            projected_2d = tsne.fit_transform(projected)
            vae_2d = tsne.fit_transform(vae_feats)
            
            # 计算对应点之间的距离
            distances = np.sqrt(np.sum((projected_2d - vae_2d)**2, axis=1))
            alignment_score = 1.0 / (1.0 + np.mean(distances))
            
            return alignment_score
            
        except Exception as e:
            print(f"Warning: Alignment score computation failed: {e}")
            return 0.0
    
    def visualize_projection_results(self):
        """
        可视化投影结果
        Visualize projection results
        """
        if not VISUALIZATION_CONFIG.get('plot_projection_results', True):
            return
        
        print("Generating projection visualization...")
        
        if self.projected_features is None or self.vae_features is None:
            print("Warning: Features not available for visualization")
            return
        
        # 确保特征长度匹配
        min_len = min(len(self.projected_features), len(self.vae_features))
        projected = self.projected_features[:min_len]
        vae_feats = self.vae_features[:min_len]
        labels = self.labels[:min_len] if self.labels is not None else None
        
        try:
            # t-SNE可视化
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(projected)//4))
            
            # 合并特征进行t-SNE
            combined_features = np.vstack([projected, vae_feats])
            combined_2d = tsne.fit_transform(combined_features)
            
            projected_2d = combined_2d[:len(projected)]
            vae_2d = combined_2d[len(projected):]
            
            # 绘制对比图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 投影特征t-SNE
            scatter1 = axes[0].scatter(projected_2d[:, 0], projected_2d[:, 1], 
                                     c=labels if labels is not None else 'blue', 
                                     cmap='tab10', alpha=0.7, s=30)
            axes[0].set_title('Projected CNN Features (t-SNE)')
            axes[0].set_xlabel('t-SNE Dimension 1')
            axes[0].set_ylabel('t-SNE Dimension 2')
            axes[0].grid(True, alpha=0.3)
            
            # VAE特征t-SNE
            scatter2 = axes[1].scatter(vae_2d[:, 0], vae_2d[:, 1], 
                                     c=labels if labels is not None else 'red', 
                                     cmap='tab10', alpha=0.7, s=30)
            axes[1].set_title('VAE Features (t-SNE)')
            axes[1].set_xlabel('t-SNE Dimension 1')
            axes[1].set_ylabel('t-SNE Dimension 2')
            axes[1].grid(True, alpha=0.3)
            
            # 对齐可视化
            for i in range(0, len(projected_2d), max(1, len(projected_2d)//50)):
                axes[2].plot([projected_2d[i, 0], vae_2d[i, 0]], 
                           [projected_2d[i, 1], vae_2d[i, 1]], 
                           'k-', alpha=0.3, linewidth=0.5)
            
            axes[2].scatter(projected_2d[:, 0], projected_2d[:, 1], 
                          c='blue', alpha=0.7, s=30, label='Projected CNN')
            axes[2].scatter(vae_2d[:, 0], vae_2d[:, 1], 
                          c='red', alpha=0.7, s=30, label='VAE')
            axes[2].set_title('Feature Space Alignment')
            axes[2].set_xlabel('t-SNE Dimension 1')
            axes[2].set_ylabel('t-SNE Dimension 2')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图像
            save_path = os.path.join(self.results_dir, 'projection_visualization.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Projection visualization saved to {save_path}")
            
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
    
    def save_projection_results(self):
        """
        保存投影分析结果
        Save projection analysis results
        """
        print("Saving projection analysis results...")
        
        # 保存特征
        if SAVE_CONFIG.get('save_projection_features', True):
            if self.projected_features is not None:
                np.save(
                    os.path.join(self.results_dir, 'projected_features.npy'),
                    self.projected_features
                )
            
            if self.vae_features is not None:
                np.save(
                    os.path.join(self.results_dir, 'vae_features.npy'),
                    self.vae_features
                )
            
            if self.cnn_features is not None:
                np.save(
                    os.path.join(self.results_dir, 'cnn_features.npy'),
                    self.cnn_features
                )
        
        # 保存评估指标
        if self.alignment_metrics:
            import json
            # Convert numpy float32 to regular float for JSON serialization
            serializable_metrics = {}
            for key, value in self.alignment_metrics.items():
                if isinstance(value, np.float32) or isinstance(value, np.float64):
                    serializable_metrics[key] = float(value)
                else:
                    serializable_metrics[key] = value
                    
            metrics_path = os.path.join(self.results_dir, 'alignment_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
        
        print(f"Results saved to {self.results_dir}")
    
    def run_projection_analysis(self, signal_data, cnn_features, labels):
        """
        运行完整的投影分析流程
        Run complete projection analysis pipeline
        """
        print("\n=== CNN到VAE语义空间投影分析 ===")
        print("=== CNN to VAE Semantic Space Projection Analysis ===")
        
        start_time = time.time()
        
        try:
            # 存储输入数据
            self.cnn_features = cnn_features
            self.labels = labels
            
            # 步骤1：训练VAE
            print("\n步骤1: 训练VAE模型")
            cwt_images, vae_labels = self.train_vae(signal_data, labels)
            
            # 步骤2：提取VAE特征
            print("\n步骤2: 提取VAE特征")
            vae_features = self.extract_vae_features(cwt_images)
            
            # 步骤3：训练投影网络
            print("\n步骤3: 训练投影网络")
            self.train_projection_network(cnn_features, vae_features, labels)
            
            # 步骤4：执行特征投影
            print("\n步骤4: 执行特征投影")
            projected_features = self.project_features(cnn_features)
            
            # 步骤5：评估投影质量
            print("\n步骤5: 评估投影质量")
            metrics = self.evaluate_projection_quality()
            
            # 步骤6：可视化结果
            print("\n步骤6: 生成可视化结果")
            self.visualize_projection_results()
            
            # 步骤7：保存结果
            print("\n步骤7: 保存分析结果")
            self.save_projection_results()
            
            elapsed_time = time.time() - start_time
            print(f"\n投影分析完成，耗时: {elapsed_time:.2f}秒")
            print("Projection analysis completed successfully!")
            
            return {
                'projected_features': projected_features,
                'vae_features': vae_features,
                'alignment_metrics': metrics,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in projection analysis: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
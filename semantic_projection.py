#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CAE特征到VAE语义空间的投影网络实现

实现一个语义投影网络，将CAE提取的512维图像特征投影到VAE的语义空间中。
包含训练、特征对齐、可视化功能。

Author: Generated for liuliiya project
Date: 2025-01-12
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import random
import json
from typing import Dict, List, Tuple, Optional, Union

# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 配置参数
CONFIG = {
    'cae_feature_dim': 512,    # CAE特征维度
    'vae_semantic_dim': 16,    # VAE语义空间维度（基于实际文件分析）
    'batch_size': 64,          # 批处理大小
    'learning_rate': 1e-3,     # 学习率
    'epochs': 200,             # 训练轮数
    'dropout_rate': 0.3,       # Dropout比例
    'early_stopping_patience': 20,  # 早停耐心值
    'weight_decay': 1e-5,      # 权重衰减
    'random_seed': 42,         # 随机种子
    'tsne_perplexity': 30,     # t-SNE困惑度
    'tsne_n_iter': 1000,       # t-SNE迭代次数
    'save_model_path': 'models',  # 模型保存路径
    'figure_save_path': 'figures',  # 图片保存路径
}

# 创建目录
os.makedirs(CONFIG['save_model_path'], exist_ok=True)
os.makedirs(CONFIG['figure_save_path'], exist_ok=True)


class SemanticProjection(nn.Module):
    """
    语义投影网络：将CAE特征投影到VAE语义空间
    
    网络架构：
    - CAE特征 (512维) → 中间层 (256维) → 中间层 (128维) → VAE语义空间 (16维)
    - 使用ReLU激活函数和Dropout防止过拟合
    - 支持残差连接（可选）
    """
    
    def __init__(self, 
                 cae_dim: int = CONFIG['cae_feature_dim'],
                 vae_dim: int = CONFIG['vae_semantic_dim'],
                 dropout_rate: float = CONFIG['dropout_rate'],
                 use_residual: bool = True):
        super(SemanticProjection, self).__init__()
        
        self.cae_dim = cae_dim
        self.vae_dim = vae_dim
        self.use_residual = use_residual
        
        # 主投影网络
        self.projection = nn.Sequential(
            nn.Linear(cae_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, vae_dim),
        )
        
        # 残差连接（如果维度不匹配，使用线性变换）
        if self.use_residual:
            if cae_dim != vae_dim:
                self.residual_projection = nn.Linear(cae_dim, vae_dim)
            else:
                self.residual_projection = nn.Identity()
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, cae_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            cae_features: CAE特征张量 [batch_size, cae_dim]
            
        Returns:
            projected_features: 投影后的特征张量 [batch_size, vae_dim]
        """
        # 主投影路径
        projected = self.projection(cae_features)
        
        # 残差连接
        if self.use_residual:
            residual = self.residual_projection(cae_features)
            projected = projected + residual
        
        return projected


class SemanticProjectionTrainer:
    """语义投影网络训练器"""
    
    def __init__(self, 
                 model: SemanticProjection,
                 device: torch.device = DEVICE,
                 config: Dict = CONFIG):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # 优化器和调度器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for cae_features, vae_features in tqdm(train_loader, desc="Training"):
            cae_features = cae_features.to(self.device)
            vae_features = vae_features.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predicted_features = self.model(cae_features)
            
            # 计算损失
            loss = self.criterion(predicted_features, vae_features)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for cae_features, vae_features in val_loader:
                cae_features = cae_features.to(self.device)
                vae_features = vae_features.to(self.device)
                
                predicted_features = self.model(cae_features)
                loss = self.criterion(predicted_features, vae_features)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              verbose: bool = True) -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            verbose: 是否打印详细信息
            
        Returns:
            训练历史字典
        """
        print(f"开始训练语义投影网络...")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # 早停检查
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_model('best_model.pth')
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config['early_stopping_patience']:
                        print(f"早停触发，在epoch {epoch+1}")
                        break
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                          f"Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, "
                          f"LR: {lr:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                          f"Train Loss: {train_loss:.6f}, "
                          f"LR: {lr:.6f}")
        
        total_time = time.time() - start_time
        print(f"训练完成，总用时: {total_time:.2f}秒")
        
        # 保存最终模型
        self.save_model('final_model.pth')
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_time': total_time
        }
    
    def save_model(self, filename: str):
        """保存模型"""
        model_path = os.path.join(self.config['save_model_path'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, model_path)
        print(f"模型已保存到: {model_path}")
    
    def load_model(self, filename: str):
        """加载模型"""
        model_path = os.path.join(self.config['save_model_path'], filename)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"模型已从 {model_path} 加载")


class DataGenerator:
    """数据生成器，用于生成模拟的CAE和VAE特征"""
    
    def __init__(self, config: Dict = CONFIG):
        self.config = config
        self.fault_types = ['normal', 'inner', 'outer', 'ball', 
                           'inner_outer', 'inner_ball', 'outer_ball', 'inner_outer_ball']
        
        # 加载现有的VAE语义特征
        self.vae_semantics = self._load_vae_semantics()
        
    def _load_vae_semantics(self) -> Dict[str, np.ndarray]:
        """加载VAE语义特征"""
        try:
            data = np.load('semantic_features_16d.npz', allow_pickle=True)
            semantic_dict = data['semantic_dict'].item()
            print(f"成功加载VAE语义特征，包含 {len(semantic_dict)} 个故障类型")
            return semantic_dict
        except FileNotFoundError:
            print("未找到VAE语义特征文件，将生成模拟数据")
            return self._generate_mock_vae_semantics()
    
    def _generate_mock_vae_semantics(self) -> Dict[str, np.ndarray]:
        """生成模拟的VAE语义特征"""
        semantics = {}
        np.random.seed(42)  # 确保可复现
        
        # 为每个故障类型生成特征向量
        for i, fault_type in enumerate(self.fault_types):
            # 生成具有一定模式的特征向量
            base_vector = np.random.normal(0, 1, self.config['vae_semantic_dim'])
            
            # 为不同故障类型添加特定的模式
            if fault_type == 'normal':
                base_vector[:4] = [1, 0, 0, 0]  # 正常状态的模式
            elif fault_type == 'inner':
                base_vector[:4] = [0, 1, 0, 0]  # 内圈故障模式
            elif fault_type == 'outer':
                base_vector[:4] = [0, 0, 1, 0]  # 外圈故障模式
            elif fault_type == 'ball':
                base_vector[:4] = [0, 0, 0, 1]  # 球体故障模式
            elif fault_type == 'inner_outer':
                base_vector[:4] = [0, 0.7, 0.7, 0]  # 复合故障模式
            elif fault_type == 'inner_ball':
                base_vector[:4] = [0, 0.7, 0, 0.7]
            elif fault_type == 'outer_ball':
                base_vector[:4] = [0, 0, 0.7, 0.7]
            elif fault_type == 'inner_outer_ball':
                base_vector[:4] = [0, 0.6, 0.6, 0.6]
            
            # 标准化
            base_vector = base_vector / np.linalg.norm(base_vector)
            semantics[fault_type] = base_vector.astype(np.float32)
        
        return semantics
    
    def generate_cae_features(self, 
                            fault_type: str, 
                            num_samples: int = 1000) -> np.ndarray:
        """
        生成模拟的CAE特征
        
        Args:
            fault_type: 故障类型
            num_samples: 样本数量
            
        Returns:
            CAE特征数组 [num_samples, cae_feature_dim]
        """
        if fault_type not in self.fault_types:
            raise ValueError(f"未知故障类型: {fault_type}")
        
        # 获取对应的VAE语义特征作为参考
        vae_semantic = self.vae_semantics[fault_type]
        
        # 生成CAE特征：基于VAE语义特征扩展到高维空间
        cae_features = []
        
        for _ in range(num_samples):
            # 基础特征：重复VAE语义特征
            base_feature = np.tile(vae_semantic, 
                                 self.config['cae_feature_dim'] // len(vae_semantic))
            
            # 补齐维度
            if len(base_feature) < self.config['cae_feature_dim']:
                pad_length = self.config['cae_feature_dim'] - len(base_feature)
                base_feature = np.concatenate([base_feature, vae_semantic[:pad_length]])
            
            # 添加噪声
            noise = np.random.normal(0, 0.1, self.config['cae_feature_dim'])
            feature = base_feature + noise
            
            # 标准化
            feature = feature / np.linalg.norm(feature)
            cae_features.append(feature)
        
        return np.array(cae_features, dtype=np.float32)
    
    def generate_dataset(self, 
                        num_samples_per_type: int = 1000,
                        train_split: float = 0.8) -> Tuple[TensorDataset, TensorDataset]:
        """
        生成完整的数据集
        
        Args:
            num_samples_per_type: 每个故障类型的样本数量
            train_split: 训练集比例
            
        Returns:
            (训练数据集, 验证数据集)
        """
        all_cae_features = []
        all_vae_features = []
        all_labels = []
        
        for fault_type in self.fault_types:
            # 生成CAE特征
            cae_features = self.generate_cae_features(fault_type, num_samples_per_type)
            
            # 获取对应的VAE语义特征
            vae_semantic = self.vae_semantics[fault_type]
            vae_features = np.tile(vae_semantic, (num_samples_per_type, 1))
            
            # 收集数据
            all_cae_features.append(cae_features)
            all_vae_features.append(vae_features)
            all_labels.extend([fault_type] * num_samples_per_type)
        
        # 合并数据
        all_cae_features = np.vstack(all_cae_features)
        all_vae_features = np.vstack(all_vae_features)
        
        # 打乱数据
        indices = np.random.permutation(len(all_cae_features))
        all_cae_features = all_cae_features[indices]
        all_vae_features = all_vae_features[indices]
        all_labels = [all_labels[i] for i in indices]
        
        # 分割训练集和验证集
        split_idx = int(len(all_cae_features) * train_split)
        
        train_cae = all_cae_features[:split_idx]
        train_vae = all_vae_features[:split_idx]
        train_labels = all_labels[:split_idx]
        
        val_cae = all_cae_features[split_idx:]
        val_vae = all_vae_features[split_idx:]
        val_labels = all_labels[split_idx:]
        
        # 创建数据集
        train_dataset = TensorDataset(
            torch.FloatTensor(train_cae),
            torch.FloatTensor(train_vae)
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(val_cae),
            torch.FloatTensor(val_vae)
        )
        
        print(f"数据集生成完成:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  故障类型: {self.fault_types}")
        
        return train_dataset, val_dataset, train_labels, val_labels


class SemanticVisualizer:
    """语义特征可视化器"""
    
    def __init__(self, config: Dict = CONFIG):
        self.config = config
        self.fault_types = ['normal', 'inner', 'outer', 'ball', 
                           'inner_outer', 'inner_ball', 'outer_ball', 'inner_outer_ball']
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def visualize_training_history(self, 
                                 train_losses: List[float],
                                 val_losses: List[float] = None,
                                 save_path: str = None):
        """可视化训练历史"""
        plt.figure(figsize=(12, 5))
        
        # 训练损失
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学习率变化（如果有的话）
        plt.subplot(1, 2, 2)
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', linewidth=2, color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.title('Validation Loss')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到: {save_path}")
        
        plt.show()
    
    def visualize_semantic_space(self,
                               vae_features: np.ndarray,
                               cae_projected_features: np.ndarray,
                               labels: List[str],
                               save_path: str = None,
                               title: str = "语义空间可视化"):
        """
        可视化语义空间中的特征分布
        
        Args:
            vae_features: VAE直接提取的语义特征
            cae_projected_features: CAE投影后的语义特征
            labels: 标签列表
            save_path: 保存路径
            title: 图片标题
        """
        # 合并特征进行t-SNE降维
        all_features = np.vstack([vae_features, cae_projected_features])
        
        # 创建标签
        vae_labels = [f"VAE_{label}" for label in labels]
        cae_labels = [f"CAE_{label}" for label in labels]
        all_labels = vae_labels + cae_labels
        
        # 创建来源标签
        source_labels = ['VAE'] * len(vae_features) + ['CAE'] * len(cae_projected_features)
        
        # 创建故障类型标签
        fault_labels = labels + labels
        
        # 执行t-SNE
        print("执行t-SNE降维...")
        tsne = TSNE(n_components=2, 
                   random_state=42, 
                   perplexity=min(self.config['tsne_perplexity'], len(all_features)//4),
                   max_iter=self.config['tsne_n_iter'])
        
        features_2d = tsne.fit_transform(all_features)
        
        # 绘图
        plt.figure(figsize=(15, 12))
        
        # 分割VAE和CAE特征的2D坐标
        vae_coords = features_2d[:len(vae_features)]
        cae_coords = features_2d[len(vae_features):]
        
        # 获取唯一故障类型
        unique_faults = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_faults)))
        
        # 绘制VAE特征（圆形）
        for i, fault_type in enumerate(unique_faults):
            vae_mask = np.array([label == fault_type for label in labels])
            if np.any(vae_mask):
                plt.scatter(vae_coords[vae_mask, 0], vae_coords[vae_mask, 1], 
                           c=[colors[i]], s=100, alpha=0.7, marker='o', 
                           label=f'VAE_{fault_type}', edgecolors='black', linewidth=0.5)
        
        # 绘制CAE特征（三角形）
        for i, fault_type in enumerate(unique_faults):
            cae_mask = np.array([label == fault_type for label in labels])
            if np.any(cae_mask):
                plt.scatter(cae_coords[cae_mask, 0], cae_coords[cae_mask, 1], 
                           c=[colors[i]], s=100, alpha=0.7, marker='^', 
                           label=f'CAE_{fault_type}', edgecolors='black', linewidth=0.5)
        
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 添加说明
        plt.figtext(0.02, 0.02, 
                   "○ = VAE直接提取的语义特征\n△ = CAE投影后的语义特征\n相同颜色代表相同故障类型",
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"语义空间可视化图已保存到: {save_path}")
        
        plt.show()
    
    def visualize_feature_alignment(self,
                                  vae_features: np.ndarray,
                                  cae_projected_features: np.ndarray,
                                  labels: List[str],
                                  save_path: str = None):
        """
        可视化特征对齐情况
        
        Args:
            vae_features: VAE语义特征
            cae_projected_features: CAE投影后的特征
            labels: 标签列表
            save_path: 保存路径
        """
        # 计算每个故障类型的特征对齐度
        unique_faults = list(set(labels))
        alignment_scores = []
        
        for fault_type in unique_faults:
            mask = np.array([label == fault_type for label in labels])
            if np.any(mask):
                vae_fault_features = vae_features[mask]
                cae_fault_features = cae_projected_features[mask]
                
                # 计算余弦相似度
                similarities = []
                for i in range(len(vae_fault_features)):
                    vae_feat = vae_fault_features[i]
                    cae_feat = cae_fault_features[i]
                    
                    # 余弦相似度
                    similarity = np.dot(vae_feat, cae_feat) / (
                        np.linalg.norm(vae_feat) * np.linalg.norm(cae_feat) + 1e-8)
                    similarities.append(similarity)
                
                alignment_scores.append({
                    'fault_type': fault_type,
                    'mean_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities),
                    'similarities': similarities
                })
        
        # 绘制对齐度分析
        plt.figure(figsize=(15, 10))
        
        # 子图1：平均对齐度
        plt.subplot(2, 2, 1)
        fault_names = [score['fault_type'] for score in alignment_scores]
        mean_sims = [score['mean_similarity'] for score in alignment_scores]
        std_sims = [score['std_similarity'] for score in alignment_scores]
        
        bars = plt.bar(fault_names, mean_sims, yerr=std_sims, capsize=5, alpha=0.7)
        plt.ylabel('平均余弦相似度')
        plt.title('各故障类型特征对齐度')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 为每个柱子添加数值标签
        for bar, mean_sim in zip(bars, mean_sims):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{mean_sim:.3f}', ha='center', va='bottom')
        
        # 子图2：对齐度分布
        plt.subplot(2, 2, 2)
        all_similarities = []
        for score in alignment_scores:
            all_similarities.extend(score['similarities'])
        
        plt.hist(all_similarities, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('余弦相似度')
        plt.ylabel('频数')
        plt.title('特征对齐度分布')
        plt.grid(True, alpha=0.3)
        
        # 子图3：每个故障类型的对齐度箱线图
        plt.subplot(2, 2, 3)
        similarities_by_fault = [score['similarities'] for score in alignment_scores]
        plt.boxplot(similarities_by_fault, labels=fault_names)
        plt.ylabel('余弦相似度')
        plt.title('各故障类型对齐度分布')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 子图4：特征距离分析
        plt.subplot(2, 2, 4)
        distances = []
        for fault_type in unique_faults:
            mask = np.array([label == fault_type for label in labels])
            if np.any(mask):
                vae_fault_features = vae_features[mask]
                cae_fault_features = cae_projected_features[mask]
                
                # 计算欧氏距离
                fault_distances = []
                for i in range(len(vae_fault_features)):
                    distance = np.linalg.norm(vae_fault_features[i] - cae_fault_features[i])
                    fault_distances.append(distance)
                distances.append(fault_distances)
        
        plt.boxplot(distances, labels=fault_names)
        plt.ylabel('欧氏距离')
        plt.title('VAE与CAE特征距离分布')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"特征对齐分析图已保存到: {save_path}")
        
        plt.show()
        
        # 打印对齐度统计信息
        print("\n=== 特征对齐度分析 ===")
        for score in alignment_scores:
            print(f"{score['fault_type']}: "
                  f"平均相似度={score['mean_similarity']:.4f} ± {score['std_similarity']:.4f}")
        
        overall_mean = np.mean([score['mean_similarity'] for score in alignment_scores])
        print(f"整体平均对齐度: {overall_mean:.4f}")
        
        return alignment_scores


def main():
    """主函数：执行完整的语义投影网络训练和评估流程"""
    print("=== CAE特征到VAE语义空间的投影网络实现 ===")
    
    # 1. 生成数据
    print("\n1. 生成数据集...")
    data_generator = DataGenerator(CONFIG)
    train_dataset, val_dataset, train_labels, val_labels = data_generator.generate_dataset(
        num_samples_per_type=500,  # 每个故障类型500个样本
        train_split=0.8
    )
    
    # 2. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                          shuffle=False, num_workers=0)
    
    # 3. 创建模型
    print("\n2. 创建语义投影网络...")
    model = SemanticProjection(
        cae_dim=CONFIG['cae_feature_dim'],
        vae_dim=CONFIG['vae_semantic_dim'],
        dropout_rate=CONFIG['dropout_rate']
    )
    
    # 4. 创建训练器
    trainer = SemanticProjectionTrainer(model, DEVICE, CONFIG)
    
    # 5. 训练模型
    print("\n3. 开始训练...")
    training_history = trainer.train(train_loader, val_loader, verbose=True)
    
    # 6. 加载最佳模型
    print("\n4. 加载最佳模型进行评估...")
    trainer.load_model('best_model.pth')
    
    # 7. 评估模型
    print("\n5. 评估模型性能...")
    model.eval()
    
    # 获取测试数据
    test_cae_features = []
    test_vae_features = []
    test_labels = []
    
    with torch.no_grad():
        for cae_feat, vae_feat in val_loader:
            cae_feat = cae_feat.to(DEVICE)
            projected_feat = model(cae_feat)
            
            test_cae_features.append(projected_feat.cpu().numpy())
            test_vae_features.append(vae_feat.numpy())
    
    # 合并测试数据
    test_cae_features = np.vstack(test_cae_features)
    test_vae_features = np.vstack(test_vae_features)
    
    # 8. 可视化结果
    print("\n6. 生成可视化结果...")
    visualizer = SemanticVisualizer(CONFIG)
    
    # 训练历史
    visualizer.visualize_training_history(
        training_history['train_losses'],
        training_history['val_losses'],
        save_path=os.path.join(CONFIG['figure_save_path'], 'training_history.png')
    )
    
    # 语义空间可视化
    visualizer.visualize_semantic_space(
        test_vae_features,
        test_cae_features,
        val_labels,
        save_path=os.path.join(CONFIG['figure_save_path'], 'semantic_space.png'),
        title="CAE特征到VAE语义空间的投影效果"
    )
    
    # 特征对齐分析
    alignment_scores = visualizer.visualize_feature_alignment(
        test_vae_features,
        test_cae_features,
        val_labels,
        save_path=os.path.join(CONFIG['figure_save_path'], 'feature_alignment.png')
    )
    
    # 9. 保存结果
    print("\n7. 保存结果...")
    results = {
        'config': CONFIG,
        'training_history': training_history,
        'alignment_scores': alignment_scores,
        'model_info': {
            'parameters': sum(p.numel() for p in model.parameters()),
            'architecture': str(model)
        }
    }
    
    with open(os.path.join(CONFIG['figure_save_path'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n=== 语义投影网络训练和评估完成 ===")
    print(f"模型文件保存在: {CONFIG['save_model_path']}")
    print(f"结果图片保存在: {CONFIG['figure_save_path']}")
    
    return model, trainer, visualizer


if __name__ == "__main__":
    # 运行主程序
    model, trainer, visualizer = main()
    
    print("\n程序执行完成！")
    print("您可以通过以下方式使用训练好的模型:")
    print("1. 使用 model.forward(cae_features) 进行特征投影")
    print("2. 使用 trainer.load_model('best_model.pth') 加载最佳模型")
    print("3. 使用 visualizer 的各种方法进行可视化分析")
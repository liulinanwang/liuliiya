# -*- coding: utf-8 -*-
"""
Configuration file for bearing fault diagnosis system
"""

# =============================================================================
# 主系统配置 (Main System Configuration)
# =============================================================================

# 数据处理配置
SEGMENT_LENGTH = 1024
OVERLAP = 0.5
STEP = int(SEGMENT_LENGTH * (1 - OVERLAP))
if STEP < 1: 
    STEP = 1

# 小波相关配置
DEFAULT_WAVELET = 'db4'
DEFAULT_WAVELET_LEVEL = 3

# =============================================================================
# CNN到VAE语义空间投影分析配置 (CNN to VAE Projection Analysis Configuration)
# =============================================================================

# 投影分析控制开关
ENABLE_PROJECTION_ANALYSIS = True  # 是否启用投影分析

# VAE模型配置
VAE_CONFIG = {
    'z_dim': 32,                    # VAE潜在空间维度
    'nc': 3,                        # VAE输入通道数（用于CWT图像）
    'beta': 4.0,                    # Beta-VAE的beta参数
    'epochs': 50,                   # VAE训练轮数
    'batch_size': 64,               # VAE训练批次大小
    'learning_rate': 1e-4,          # VAE学习率
    'image_size': 64,               # 输入图像尺寸
}

# 投影器配置
PROJECTION_CONFIG = {
    'projector_hidden_dims': [512, 256, 128],  # 投影器隐藏层维度
    'projector_epochs': 30,                     # 投影器训练轮数
    'projector_lr': 1e-3,                      # 投影器学习率
    'projector_batch_size': 32,                # 投影器训练批次大小
    'alignment_loss_weight': 1.0,              # 对齐损失权重
    'reconstruction_loss_weight': 0.1,         # 重构损失权重
    'contrastive_loss_weight': 0.5,            # 对比损失权重
    'temperature': 0.07,                       # 对比学习温度参数
}

# CWT图像生成配置（用于VAE训练）
CWT_CONFIG = {
    'sampling_rate': 12000,          # 采样率
    'freq_range': (800, 4000),       # 频率范围
    'num_scales': 64,                # 尺度数量
    'wavelet': 'cmor1.5-1.0',       # CWT小波基
    'output_size': (64, 64),         # 输出图像尺寸
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'plot_projection_results': True,     # 是否绘制投影结果
    'plot_alignment_metrics': True,      # 是否绘制对齐指标
    'plot_tsne_comparison': True,        # 是否绘制t-SNE对比
    'save_intermediate_results': True,   # 是否保存中间结果
}

# 结果保存配置
SAVE_CONFIG = {
    'save_projection_model': True,       # 是否保存投影模型
    'save_vae_model': True,              # 是否保存VAE模型
    'save_projection_features': True,    # 是否保存投影特征
    'results_dir': 'projection_results', # 结果保存目录
    'model_dir': 'saved_models',         # 模型保存目录
}

# =============================================================================
# 现有系统配置保持不变 (Existing System Configuration)
# =============================================================================

# AE相关配置
AE_LATENT_DIM = 64
AE_EPOCHS = 100
AE_LR = 0.001
AE_BATCH_SIZE = 64
AE_CONTRASTIVE_WEIGHT = 1.2
AE_NOISE_STD = 0.05

# CNN/SEN配置
CNN_EPOCHS = 30
CNN_LR = 0.0005
SEN_EPOCHS = 20
SEN_LR = 0.001
CNN_FEATURE_DIM = 512
DEFAULT_BATCH_SIZE = 64

# 故障类型映射
FAULT_TYPES = {
    'normal': 0, 'inner': 1, 'outer': 2, 'ball': 3,
    'inner_outer': 4, 'inner_ball': 5, 'outer_ball': 6, 'inner_outer_ball': 7
}

# 数据路径配置（需要根据实际情况修改）
DEFAULT_DATA_PATH = "E:/研究生/CNN/HDU1000-600"

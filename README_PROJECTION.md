# CNN到VAE语义空间投影分析系统集成

## 概述

本项目成功将CNN到VAE语义空间投影分析功能整合到现有的轴承故障诊断系统中，实现了无缝集成的可选分析模块。该功能可以将CNN提取的特征映射到VAE的语义空间，提供更深入的特征空间分析能力。

## 功能特性

### 1. 无缝集成
- 作为可选功能添加到主脚本中，不影响原有功能
- 通过配置开关控制是否启用投影分析
- 与现有CNN和VAE分析结果完美整合

### 2. 数据流整合
- 利用现有的CNN特征和信号数据
- 自动生成CWT时频图像用于VAE训练
- 完整的特征提取和投影流程

### 3. 配置控制
- 集中化配置管理
- 详细的参数调节选项
- 用户交互控制

### 4. 结果保存与可视化
- 投影分析结果与其他分析结果统一保存
- t-SNE可视化对比
- 详细的评估指标

## 文件结构

```
liuliiya/
├── main.py                    # 主程序（已集成投影分析）
├── config.py                  # 配置文件（新增）
├── projection_system.py       # 投影分析系统（新增）
├── test_integration.py        # 集成测试（新增）
├── claude.py                  # 原有VAE实现
├── utils.py                   # 工具函数
└── README_PROJECTION.md       # 本文档
```

## 配置说明

### 主要配置项

```python
# 投影分析控制开关
ENABLE_PROJECTION_ANALYSIS = True  # 是否启用投影分析

# VAE模型配置
VAE_CONFIG = {
    'z_dim': 32,                    # VAE潜在空间维度
    'nc': 3,                        # VAE输入通道数
    'beta': 4.0,                    # Beta-VAE的beta参数
    'epochs': 50,                   # VAE训练轮数
    'batch_size': 64,               # VAE训练批次大小
    'learning_rate': 1e-4,          # VAE学习率
}

# 投影器配置
PROJECTION_CONFIG = {
    'projector_hidden_dims': [512, 256, 128],  # 投影器隐藏层维度
    'projector_epochs': 30,                     # 投影器训练轮数
    'projector_lr': 1e-3,                      # 投影器学习率
    'projector_batch_size': 32,                # 投影器训练批次大小
    'alignment_loss_weight': 1.0,              # 对齐损失权重
    'contrastive_loss_weight': 0.5,            # 对比损失权重
}
```

## 使用方法

### 1. 基本使用

```python
from main import ZeroShotCompoundFaultDiagnosis

# 初始化诊断系统
fault_diagnosis = ZeroShotCompoundFaultDiagnosis(
    data_path="your_data_path",
    sample_length=1024,
    latent_dim=64,
    batch_size=64
)

# 运行完整管道（包含投影分析）
results = fault_diagnosis.run_pipeline()

# 查看结果
print(f"ZSL准确率: {results['zsl_accuracy']:.2f}%")
if results['projection_results']:
    metrics = results['projection_results']['alignment_metrics']
    print(f"投影质量 - MSE: {metrics['mse_loss']:.6f}")
    print(f"投影质量 - 余弦相似性: {metrics['cosine_similarity']:.4f}")
```

### 2. 独立使用投影分析

```python
from projection_system import CNNToVAEProjectionSystem

# 初始化投影系统
projection_system = CNNToVAEProjectionSystem()

# 准备数据（信号数据、CNN特征、标签）
signal_data = your_signal_data  # [N, signal_length]
cnn_features = your_cnn_features  # [N, feature_dim]
labels = your_labels  # [N]

# 运行投影分析
results = projection_system.run_projection_analysis(
    signal_data=signal_data,
    cnn_features=cnn_features,
    labels=labels
)
```

### 3. 配置调节

```python
# 修改配置文件config.py中的参数
# 或在运行时动态修改
import config

# 禁用投影分析
config.ENABLE_PROJECTION_ANALYSIS = False

# 调整VAE参数
config.VAE_CONFIG['epochs'] = 100
config.VAE_CONFIG['z_dim'] = 64

# 调整投影器参数
config.PROJECTION_CONFIG['projector_epochs'] = 50
config.PROJECTION_CONFIG['projector_lr'] = 5e-4
```

## 核心组件

### 1. CNNToVAEProjectionSystem

主要的投影分析系统类，包含完整的分析流程：

- **VAE训练**: 基于CWT时频图像训练Beta-VAE模型
- **特征提取**: 从VAE编码器提取语义特征
- **投影网络训练**: 训练CNN特征到VAE语义空间的映射
- **质量评估**: 计算对齐指标和相似性度量
- **可视化**: 生成t-SNE对比图和对齐可视化

### 2. ProjectionNetwork

CNN特征到VAE语义空间的投影网络：

```python
ProjectionNetwork(
    input_dim=512,      # CNN特征维度
    output_dim=32,      # VAE语义空间维度
    hidden_dims=[512, 256, 128]  # 隐藏层配置
)
```

### 3. CWTImageGenerator

连续小波变换图像生成器，用于VAE训练：

- 信号预处理和滤波
- CWT时频分析
- 图像归一化和尺寸调整
- 多通道图像生成

## 技术特性

### 1. 鲁棒性设计
- 自动处理依赖缺失（如PyEMD）
- 提供VAE组件的备用实现
- 全面的错误处理和异常捕获
- 输入数据验证和清理

### 2. 内存优化
- 批处理大数据集
- 渐进式特征提取
- 及时释放中间结果
- GPU/CPU自适应

### 3. 可扩展性
- 模块化设计
- 配置驱动的参数调节
- 易于添加新的评估指标
- 支持自定义损失函数

## 评估指标

### 1. 投影质量指标

- **MSE损失**: 投影特征与VAE特征的均方误差
- **余弦相似性**: 特征向量的余弦相似度
- **对齐分数**: 基于t-SNE的特征空间对齐度

### 2. 可视化分析

- **投影特征t-SNE**: CNN投影特征的二维可视化
- **VAE特征t-SNE**: VAE原始特征的二维可视化  
- **对齐可视化**: 对应特征点的连接线图

## 结果输出

### 1. 模型文件
- `vae_model.pth`: 训练好的VAE模型
- `projection_network.pth`: 训练好的投影网络

### 2. 特征文件
- `cnn_features.npy`: CNN提取的特征
- `vae_features.npy`: VAE编码的特征
- `projected_features.npy`: 投影后的特征

### 3. 分析结果
- `alignment_metrics.json`: 对齐评估指标
- `projection_visualization.png`: 可视化结果图

## 性能优化建议

### 1. 训练效率
- 根据数据集大小调整批次大小
- 使用GPU加速（如果可用）
- 适当减少训练轮数进行快速测试

### 2. 内存使用
- 处理大数据集时使用较小的批次大小
- 及时清理不需要的中间变量
- 考虑使用数据加载器的多进程功能

### 3. 准确性提升
- 增加VAE和投影器的训练轮数
- 调节损失函数权重
- 使用数据增强技术

## 故障排除

### 1. 常见问题

**Q: 投影分析无法启动**
A: 检查配置中的`ENABLE_PROJECTION_ANALYSIS`是否为True，确认依赖库已正确安装

**Q: VAE训练失败** 
A: 检查CWT图像生成是否成功，确认输入信号数据格式正确

**Q: 内存不足**
A: 减小批次大小，调整`batch_size`和`projector_batch_size`参数

**Q: 投影质量差**
A: 增加训练轮数，调节学习率，检查CNN特征质量

### 2. 调试模式

启用详细日志输出：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或者在config.py中添加
DEBUG_MODE = True
```

## 扩展开发

### 1. 添加新的评估指标

```python
def custom_alignment_metric(projected_features, vae_features):
    # 实现自定义对齐指标
    return metric_value

# 在CNNToVAEProjectionSystem中添加
self.alignment_metrics['custom_metric'] = custom_alignment_metric(
    projected, vae_feats
)
```

### 2. 自定义损失函数

```python
class CustomProjectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, projected, target):
        # 实现自定义损失
        return loss

# 在投影网络训练中使用
custom_loss = CustomProjectionLoss()
```

### 3. 新的可视化方法

```python
def custom_visualization(projected_features, vae_features, labels):
    # 实现自定义可视化
    plt.figure()
    # 绘制代码
    plt.savefig('custom_visualization.png')

# 在可视化阶段调用
custom_visualization(self.projected_features, self.vae_features, self.labels)
```

## 测试

运行集成测试：

```bash
python test_integration.py
```

测试包括：
- 投影系统基本功能测试
- 主管道集成测试
- 合成数据端到端测试

## 参考文献

1. Higgins, I., et al. "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." ICLR 2017.
2. Chen, T., et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.
3. van der Maaten, L., & Hinton, G. "Visualizing Data using t-SNE." JMLR 2008.

## 版本历史

- **v1.0**: 初始实现，基本投影功能
- **v1.1**: 添加配置管理和用户交互
- **v1.2**: 集成到主管道，添加可视化
- **v1.3**: 性能优化和鲁棒性改进

## 支持与反馈

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者
- 参与项目讨论